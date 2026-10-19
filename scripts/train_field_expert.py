#!/usr/bin/env python3
"""E0/E1: train SmolVLA's action expert + field encoder on GT-field teacher data.

Arms differ ONLY in the plan channel fed at training time:
  no-field      no field tokens at all
  zero-plan     field tokens, plan channel zero
  random-plan   smooth low-frequency random plan
  wrong-goal    GT plan to a random free goal at similar distance
  gt-plan       exact geodesic plan to the visit subgoal (main arm)

The action labels, data, high-level subgoals, expert init, and budget are
shared across arms.  Fixed 15K steps; no MSE-based model selection.

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_field_expert.py \
      --data data/multi_lowlevel_2w.npz --arm gt-plan --seed 0 \
      --steps 15000 --batch 16 --out ckpts/e0/gt-plan_s0
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from field_common import (build_field_policy, save_field_ckpt, ACTION_DIM,
                          STATE_DIM, SUBGOAL_DIM, CHUNK)


def load_data(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    print(f"data {path}: images {out['images'].shape} actions "
          f"{out['actions'].shape} fields {out['fields'].shape}")
    return out


def make_norm(d, subset=None):
    idx = np.arange(len(d["states"])) if subset is None else subset
    st = d["states"][idx]
    act = d["actions"][idx].reshape(-1, ACTION_DIM)
    st_mean = st.mean(0).astype(np.float32)
    st_std = np.maximum(st.std(0), 1e-3).astype(np.float32)
    act_mean = act.mean(0).astype(np.float32)
    act_std = np.maximum(act.std(0), 1e-3).astype(np.float32)
    return st_mean, st_std, act_mean, act_std


def batch_tensors(d, idx, st_mean, st_std, act_mean, act_std, arm,
                  device, field_dropout, rng):
    im = torch.from_numpy(d["images"][idx]).float().div_(255.0).to(device)
    st = (d["states"][idx] - st_mean) / st_std
    st[:, 3:6] = 0.0                      # rot is constant in this task
    state = np.concatenate([st, d["subgoals"][idx]], 1).astype(np.float32)
    act = (d["actions"][idx] - act_mean) / act_std
    field = None
    if arm != "no-field":
        field = torch.from_numpy(np.asarray(d["fields"][idx], np.float32))
        if arm == "zero-plan":
            field[:, 1] = 0.0
        elif arm == "random-plan":
            field[:, 1] = torch.from_numpy(
                np.asarray(d["plans_random"][idx], np.float32))
        elif arm == "wrong-goal":
            field[:, 1] = torch.from_numpy(
                np.asarray(d["plans_wrong"][idx], np.float32))
        if field_dropout > 0:
            drop = torch.rand(field.shape[0]) < field_dropout
            field[drop] = 0.0
        field = field.to(device)
    return {
        "observation.images.camera1": im,
        "observation.state": torch.from_numpy(state).to(device),
        "action": torch.from_numpy(act).to(device),
        "observation.language.tokens": None,   # filled by caller
        "observation.language.attention_mask": None,
    }, field


def loss_for_batch(wrapper, batch):
    policy = wrapper.policy
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(actions.shape[0], actions.device)
    losses = policy.model.forward(
        images, img_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state, actions, noise=noise, time=time)
    return losses[:, :, :ACTION_DIM].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/multi_lowlevel_2w.npz")
    ap.add_argument("--arm", default="gt-plan",
                    choices=["no-field", "zero-plan", "random-plan",
                             "wrong-goal", "gt-plan"])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--field-lr", type=float, default=3e-4)
    ap.add_argument("--field-dropout", type=float, default=0.15)
    ap.add_argument("--subset", type=int, default=0,
                    help="train on the first N samples (pilot overfit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="ckpts/e0/gt-plan_s0")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out, exist_ok=True)

    d = load_data(args.data)
    n_all = len(d["states"])
    pool = np.arange(n_all if args.subset <= 0 else min(args.subset, n_all))
    train_idx = pool.copy()
    st_mean, st_std, act_mean, act_std = make_norm(d, train_idx)
    print(f"train samples {len(train_idx)}/{n_all}  "
          f"st_std {np.round(st_std,4)}  act_std {np.round(act_std,4)}")

    wrapper, lang_ids, lang_mask = build_field_policy(device, args.arm)
    policy = wrapper.policy
    trainable = [p for p in policy.parameters() if p.requires_grad]
    field_params = ([] if wrapper.field_encoder is None
                    else list(wrapper.field_encoder.parameters()))
    n_tr = sum(p.numel() for p in trainable)
    n_f = sum(p.numel() for p in field_params)
    print(f"arm {args.arm}  expert-trainable {n_tr/1e6:.2f}M  "
          f"field encoder {n_f/1e3:.1f}K")

    groups = [{"params": trainable, "lr": args.lr}]
    if field_params:
        groups.append({"params": field_params, "lr": args.field_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-5)
    warmup = min(500, max(1, args.steps // 20))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup)),
         torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, max(1, args.steps - warmup))],
        milestones=[warmup])

    def run_batch(idx):
        batch, field = batch_tensors(
            d, idx, st_mean, st_std, act_mean, act_std, args.arm,
            device, args.field_dropout, rng)
        batch["observation.language.tokens"] = lang_ids.expand(len(idx), -1)
        batch["observation.language.attention_mask"] = \
            lang_mask.expand(len(idx), -1)
        if wrapper.field_encoder is not None:
            wrapper.set_field(field)
        else:
            wrapper.clear_field()
        return loss_for_batch(wrapper, batch)

    t0 = time.time()
    losses = []
    field_gn = float("nan")
    print(f"{'step':>6} {'loss':>9} {'fieldGN':>9} {'lr':>9} {'wall':>6}",
          flush=True)
    for step in range(1, args.steps + 1):
        wrapper.train()
        policy.model.vlm_with_expert.vlm.eval()   # keep frozen VLM deterministic
        idx = train_idx[rng.choice(len(train_idx), args.batch, replace=False)]
        loss = run_batch(idx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad]
            + field_params, 1.0)
        if field_params and step == 50:
            field_gn = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in field_params
                if p.grad is not None)))
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % args.log_every == 0 or step == 1:
            print(f"{step:>6} {np.mean(losses[-args.log_every:]):>9.4f} "
                  f"{field_gn:>9.2e} {opt.param_groups[0]['lr']:>9.1e} "
                  f"{time.time()-t0:>6.0f}", flush=True)

    extra = {
        "arm": args.arm,
        "args": vars(args),
        "step": args.steps,
        "final_loss": float(np.mean(losses[-200:])),
        "field_grad_norm_step50": field_gn,
        "st_mean": st_mean, "st_std": st_std,
        "act_mean": act_mean, "act_std": act_std,
        "ch_mean": d["ch_mean"], "ch_std": d["ch_std"],
        "data_size": os.path.getsize(args.data),
        "data_mtime": os.path.getmtime(args.data),
        "data_sha1_stats": hashlib.sha1(np.concatenate(
            [st_mean, st_std, act_mean, act_std]).tobytes()).hexdigest(),
        "train_samples": int(len(train_idx)),
        "elapsed_s": time.time() - t0,
    }
    save_field_ckpt(os.path.join(args.out, "last.pt"), wrapper, extra)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in extra.items()}, f, indent=2)
    print(f"saved {args.out}/last.pt  final loss {extra['final_loss']:.4f}  "
          f"field grad {field_gn:.2e}")


if __name__ == "__main__":
    main()
