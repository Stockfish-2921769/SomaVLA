#!/usr/bin/env python3
"""Stage A: train the VLA-lite low-level policy on teacher subgoals.

One field source per run (nca/cnn/euclid/gt/none/zero/random); everything
else is shared.  Uses precomputed frozen SigLIP tokens.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from soma.vla_lite.determinism import (enable_determinism, loader_generator,
                                       seed_everything)
from soma.vla_lite.policy import VLALiteLowLevel


class LowLevelData(Dataset):
    def __init__(self, path, feat_path, split, limit=0):
        d = np.load(path)
        # Materialize once: NpzFile decompresses a whole array on every access.
        split_arr = d["split"]
        self.states = d["states"]
        self.obstacles = d["obstacles32"]
        self.plans = d["plans32"]
        self.ch_mean = d["ch_mean"].astype(np.float32)
        self.ch_std = d["ch_std"].astype(np.float32)
        self.subgoals_raw = (d["subgoals"] * self.ch_std +
                             self.ch_mean).astype(np.float32)
        self.actions_raw = d["actions"].astype(np.float32)
        self.feat = np.load(feat_path, mmap_mode="r")
        idx = np.flatnonzero(split_arr == split)
        if limit:
            idx = idx[:limit]
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        return dict(
            feat=np.asarray(self.feat[i], dtype=np.float32),
            state=self.states[i].astype(np.float32),
            subgoal=self.subgoals_raw[i],
            action=self.actions_raw[i],
            obstacle=self.obstacles[i].astype(np.float32),
            plan=self.plans[i].astype(np.float32),
        )


def make_policy(args, norm):
    pol = VLALiteLowLevel(field_source=args.arm, grid=args.grid,
                          chunk=8, action_dim=7,
                          readout_mode=args.readout,
                          cnn_layers=args.cnn_layers)
    pol.set_norm(norm["st_mean"], norm["st_std"],
                 norm["ch_mean"], norm["ch_std"])
    pol.nca.k_steps = args.k_infer
    return pol


def resolve_k(args):
    """Random-K training (R9 Route A): K ~ U[k_lo, k_hi] per step pushes the
    rule toward a fixed point instead of a K-specific transient. Inference K
    defaults to k_hi; eval scales it with grid size (K ∝ G)."""
    if args.k_steps:                       # fixed-K ablation (e.g. K=12)
        args.k_range = [args.k_steps, args.k_steps]
    lo, hi = args.k_range
    if not 1 <= lo <= hi:
        raise ValueError(f"bad --k-range {args.k_range}")
    args.k_infer = hi


def batch_loss(pol, batch, args, k_steps=None):
    act_mean = torch.as_tensor(args.act_mean, device=batch["action"].device)
    act_std = torch.as_tensor(args.act_std, device=batch["action"].device)
    action_norm = (batch["action"] - act_mean) / act_std
    out = pol(vis_tokens=batch["feat"], states=batch["state"],
              subgoals=batch["subgoal"], actions_norm=action_norm,
              gt_plan=batch["plan"], gt_obstacle=batch["obstacle"],
              field_dropout=args.field_dropout,
              goal_dropout=args.goal_dropout, k_steps=k_steps)
    loss = out["loss"]
    plan_loss = torch.zeros((), device=loss.device)
    mask = (batch["obstacle"] < 0.5).float()
    # zero arm trains its NCA with the same plan supervision as nca, so the
    # only difference between them is whether the decoder sees the field
    if args.plan_weight > 0 and args.arm in {"nca", "cnn", "zero"}:
        pred_plan = out["plan"]
        if pred_plan.ndim == 4:
            pred_plan = pred_plan[:, 0]
        plan_loss = (((pred_plan - batch["plan"]) ** 2) * mask).sum() / \
            mask.sum().clamp(min=1.0)
        loss = loss + args.plan_weight * plan_loss
    if args.fp_weight > 0 and out["nca_delta"] is not None:
        fp = ((out["nca_delta"] ** 2) * mask).sum() / \
            mask.sum().clamp(min=1.0)
        loss = loss + args.fp_weight * fp
    obs_loss = F.binary_cross_entropy_with_logits(
        out["obstacle_logits"][:, 0], batch["obstacle"])
    loss = loss + args.obs_weight * obs_loss
    return loss, out["loss"].detach(), plan_loss.detach(), obs_loss.detach()


@torch.no_grad()
def validate(pol, loader, args):
    pol.eval()
    tot = np.zeros(3)
    n = 0
    for batch in loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        loss, act, plan, obs = batch_loss(pol, batch, args)
        tot += np.array([act.item(), plan.item(), obs.item()])
        n += 1
    pol.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vla_lite_2w.npz")
    ap.add_argument("--feat", default="data/feat_low_fp16.npy")
    ap.add_argument("--arm", default="nca",
                    choices=["nca", "cnn", "euclid", "gt", "vis", "none",
                             "zero", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--k-steps", type=int, default=0,
                    help="fixed K (ablation); overrides --k-range")
    ap.add_argument("--k-range", type=int, nargs=2, default=[16, 64],
                    metavar=("LO", "HI"),
                    help="random-K training range; inference K = HI")
    ap.add_argument("--fp-weight", type=float, default=0.0,
                    help="fixed-point residual |s_K - s_{K-1}|^2 on plan")
    ap.add_argument("--readout", choices=["full", "plan"], default="full")
    ap.add_argument("--cnn-layers", type=int, default=12,
                    help="CNN depth; 12 = param/RF-matched to K=12, "
                         ">= G = global-RF control")
    ap.add_argument("--plan-weight", type=float, default=0.1)
    ap.add_argument("--obs-weight", type=float, default=0.5)
    ap.add_argument("--field-dropout", type=float, default=0.1)
    ap.add_argument("--goal-dropout", type=float, default=0.0,
                    help="per-sample prob. of hiding subgoal xy from the "
                         "decoder (field still gets it), so the goal must "
                         "be read from the field")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic CUDA kernels (pilot §6.2)")
    ap.add_argument("--lr-adapter", type=float, default=3e-4)
    ap.add_argument("--lr-nca", type=float, default=1e-3)
    ap.add_argument("--lr-decoder", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ckpts/vla_lite/nca_s0")
    args = ap.parse_args()
    resolve_k(args)

    if args.deterministic:
        enable_determinism()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tr = LowLevelData(args.data, args.feat, 0, args.limit)
    va = LowLevelData(args.data, args.feat, 1, args.limit)
    d = np.load(args.data)
    norm = dict(st_mean=d["st_mean"], st_std=d["st_std"],
                ch_mean=d["ch_mean"], ch_std=d["ch_std"])
    args.act_mean = d["act_mean"]
    args.act_std = d["act_std"]
    pol = make_policy(args, norm).to(args.device)
    n_tr = sum(p.numel() for p in pol.parameters() if p.requires_grad)
    print(f"arm={args.arm} train={len(tr)} val={len(va)} "
          f"trainable={n_tr/1e6:.2f}M", flush=True)

    groups = [
        dict(params=pol.adapter.parameters(), lr=args.lr_adapter),
        dict(params=pol.readout.parameters(), lr=args.lr_decoder),
        dict(params=pol.decoder.parameters(), lr=args.lr_decoder),
        dict(params=pol.language.parameters(), lr=args.lr_adapter),
    ]
    if args.arm in {"nca", "zero"}:
        groups.append(dict(params=pol.nca.parameters(), lr=args.lr_nca))
    if args.arm == "vis":
        groups.append(dict(params=pol.vis_field.parameters(),
                           lr=args.lr_adapter))
    if args.arm == "cnn":
        groups.append(dict(params=pol.cnn.parameters(), lr=args.lr_nca))
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def sched(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, sched)

    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True,
                           drop_last=True,
                           generator=loader_generator(args.seed),
                           persistent_workers=args.num_workers > 0)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    log_path = os.path.join(args.out, "train_log.jsonl")
    log_f = open(log_path, "w")
    t0 = time.time()
    it = iter(tr_loader)
    running = np.zeros(4)
    nrun = 0
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tr_loader)
            batch = next(it)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        pol.train()
        k = random.randint(*args.k_range)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            loss, act, plan, obs = batch_loss(pol, batch, args, k_steps=k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), args.clip)
        opt.step()
        lr_sched.step()
        running += np.array([loss.item(), act.item(), plan.item(), obs.item()])
        nrun += 1
        if step % args.log_every == 0:
            r = running / nrun
            rec = dict(step=step, loss=r[0], action=r[1], plan=r[2],
                       obs=r[3], lr=lr_sched.get_last_lr()[0],
                       sec=time.time() - t0)
            print(f"step {step:6d} loss {r[0]:.4f} act {r[1]:.4f} "
                  f"plan {r[2]:.4f} obs {r[3]:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            running[:] = 0
            nrun = 0
        if args.val_every and step % args.val_every == 0:
            v = validate(pol, va_loader, args)
            print(f"  val step {step}: act {v[0]:.4f} plan {v[1]:.4f} "
                  f"obs {v[2]:.4f}", flush=True)
            log_f.write(json.dumps(dict(step=step, val_act=v[0],
                                        val_plan=v[1], val_obs=v[2])) + "\n")
            log_f.flush()

    ckpt = dict(model=pol.state_dict(), args=vars(args),
                norm={k: np.asarray(v) for k, v in norm.items()},
                step=args.steps)
    torch.save(ckpt, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    log_f.close()
    print(f"saved {args.out}/last.pt", flush=True)


if __name__ == "__main__":
    main()
