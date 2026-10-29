#!/usr/bin/env python3
"""Stage B: train the VLA-lite high-level visit policy on frozen SigLIP tokens.

Predicts the normalized visit command [gx, gy, gz, open] at hover.  Only the
high-level transformer and language slot train; everything upstream is frozen.
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

from soma.vla_lite.high_level import HighLevelPolicy
from soma.vla_lite.language import LanguageSlot


class VisitData(Dataset):
    def __init__(self, path, feat_path, split, limit=0):
        d = np.load(path)
        split_arr = d["split"]
        self.states = d["states"].astype(np.float32)
        self.targets = d["chunks"][:, 0].astype(np.float32)
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
            state=self.states[i],
            target=self.targets[i],
        )


def batch_loss(hl, lang, batch, ch_mean, ch_std):
    b = batch["state"].shape[0]
    _, lang_tok = lang(batch_size=b, device=batch["state"].device)
    pred = hl(batch["feat"], batch["state"], lang_tok)
    xyz = F.l1_loss(pred[:, :3], batch["target"][:, :3])
    open_t = batch["target"][:, 3] * ch_std[3] + ch_mean[3]
    open_loss = F.binary_cross_entropy_with_logits(pred[:, 3], open_t)
    return xyz + open_loss, xyz.detach(), open_loss.detach(), pred


@torch.no_grad()
def validate(hl, lang, loader, ch_mean, ch_std):
    hl.eval()
    l1 = torch.zeros((), device=ch_mean.device)
    correct = total = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(ch_mean.device) for k, v in batch.items()}
        _, lang_tok = lang(batch_size=batch["state"].shape[0],
                           device=ch_mean.device)
        pred = hl(batch["feat"], batch["state"], lang_tok)
        l1 += F.l1_loss(pred[:, :3], batch["target"][:, :3], reduction="sum")
        open_t = batch["target"][:, 3] * ch_std[3] + ch_mean[3]
        correct += int(((pred[:, 3] > 0) == (open_t > 0.5)).sum())
        total += int(open_t.numel())
        n += batch["state"].shape[0]
    hl.train()
    return (l1 / (n * 3)).item(), correct / max(total, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vla_lite_visit_2w.npz")
    ap.add_argument("--feat", default="data/feat_visit_fp16.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ckpts/vla_lite/high_s0")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tr = VisitData(args.data, args.feat, 0, args.limit)
    va = VisitData(args.data, args.feat, 1, args.limit)
    d = np.load(args.data)
    ch_mean = torch.as_tensor(d["ch_mean"], dtype=torch.float32,
                              device=args.device)
    ch_std = torch.as_tensor(d["ch_std"], dtype=torch.float32,
                             device=args.device)

    hl = HighLevelPolicy(use_field=False).to(args.device)
    lang = LanguageSlot().to(args.device)
    n_tr = sum(p.numel() for p in hl.parameters() if p.requires_grad)
    print(f"high: train={len(tr)} val={len(va)} trainable={n_tr/1e6:.2f}M",
          flush=True)

    params = list(hl.parameters()) + list(lang.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def sched(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, sched)

    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True,
                           drop_last=True,
                           persistent_workers=args.num_workers > 0)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "w")
    t0 = time.time()
    it = iter(tr_loader)
    running = np.zeros(2)
    nrun = 0
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tr_loader)
            batch = next(it)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        hl.train()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            loss, xyz, op, _ = batch_loss(hl, lang, batch, ch_mean, ch_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        lr_sched.step()
        running += np.array([xyz.item(), op.item()])
        nrun += 1
        if step % args.log_every == 0:
            r = running / nrun
            print(f"step {step:6d} xyz {r[0]:.4f} open {r[1]:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            log_f.write(json.dumps(dict(step=step, xyz=r[0], open=r[1],
                                        lr=lr_sched.get_last_lr()[0],
                                        sec=time.time() - t0)) + "\n")
            log_f.flush()
            running[:] = 0
            nrun = 0
        if args.val_every and step % args.val_every == 0:
            v = validate(hl, lang, va_loader, ch_mean, ch_std)
            print(f"  val step {step}: xyz {v[0]:.4f} open_acc {v[1]:.3f}",
                  flush=True)
            log_f.write(json.dumps(dict(step=step, val_xyz=v[0],
                                        val_open_acc=v[1])) + "\n")
            log_f.flush()

    torch.save(dict(model=hl.state_dict(), language=lang.state_dict(),
                    args=vars(args),
                    norm={k: np.asarray(d[k]) for k in
                          ("ch_mean", "ch_std", "st_mean", "st_std")},
                    step=args.steps), os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    log_f.close()
    print(f"saved {args.out}/last.pt", flush=True)


if __name__ == "__main__":
    main()
