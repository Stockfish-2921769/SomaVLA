#!/usr/bin/env python3
"""Decision B — data-efficiency curve for the SigLIP planner head.

The small-model advantage claim needs a data-budget axis: how many synthetic
scenes must the tiny 0.27M head see before its regression crosses the
closed-loop-safety threshold (hard-band m/μ rel err small + hard-cell sim ok)?
Train from scratch on the FIRST N of the fixed 10800-scene train set, fixed
val = the same 1200 scenes, ~18 epochs per N (epoch budget constant → N is the
only lever). Reports per N: val scene xy MAE (mm), μ/mass MAE, hard-band m/μ rel
err, under-estimate fraction, best val MSE, wall-clock.

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 python scripts/sweep_planner_data.py \
      --sizes 1200 2400 4800 9600 10800 --epochs 18 --out ckpts/planner_sweep
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch
import torch.nn as nn

from train_planner import SigLIPPlanner
from train_vla_tiny import count_params


def train_one(im, lab, vim, vlab, steps, batch, out_path, device):
    """Frozen SigLIP + head, MSE on per-dim-normalized labels (same recipe as
    train_planner.py main). Returns (best_val, report_dict)."""
    mean = lab.mean(axis=0)
    std = lab.std(axis=0) + 1e-6
    lab_n = (lab - mean) / std
    vlab_n = (vlab - mean) / std
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    planner = SigLIPPlanner().to(device)
    opt = torch.optim.AdamW(planner.head.parameters(), lr=5e-4, weight_decay=1e-4)
    warmup = min(200, steps // 10)
    lin = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - warmup)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [lin, cos], milestones=[warmup])

    n = len(im)
    def batch_at(i):
        sl = slice(i * batch, (i + 1) * batch)
        return (torch.from_numpy(im[sl]).to(device),
                torch.from_numpy(lab_n[sl]).to(device))

    steps_per_ep = max(1, n // batch)
    best_val = float("inf")
    os.makedirs(out_path, exist_ok=True)
    for s in range(1, steps + 1):
        planner.train()
        img, yy = batch_at(s % steps_per_ep)
        loss = nn.functional.mse_loss(planner(img), yy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.head.parameters(), 5.0)
        opt.step()
        sched.step()
        if s % max(200, steps // 10) == 0 or s == steps:
            planner.eval()
            with torch.no_grad():
                vp = planner(torch.from_numpy(vim).to(device))
                vl = nn.functional.mse_loss(
                    vp, torch.from_numpy(vlab_n).to(device)).item()
            if vl < best_val:
                best_val = vl
                torch.save({"head": planner.head.state_dict(),
                            "norm": {"mean": mean_t.cpu(), "std": std_t.cpu()}},
                           os.path.join(out_path, "best.pt"))

    # Final per-dim val MAE (denormalized) + m/μ attribution on the best head.
    planner.head.load_state_dict(
        torch.load(os.path.join(out_path, "best.pt"), map_location=device,
                   weights_only=True)["head"])
    planner.eval()
    with torch.no_grad():
        vp = planner(torch.from_numpy(vim).to(device)).cpu().numpy()
    vp = vp * std + mean          # normalized preds → physical units
    vt = vlab                      # raw labels (already physical units)
    mae = np.abs(vp - vt).mean(axis=0)
    scene_mm = mae[:4].mean() * 1000.0
    r_true = vt[:, 5] / vt[:, 4]
    r_pred = vp[:, 5] / vp[:, 4]
    rel = (r_pred - r_true) / r_true
    hard = r_true >= 1.16
    hard_rel = np.abs(rel[hard]).mean() if hard.any() else float("nan")
    under = (rel[hard] < -0.1).mean() if hard.any() else float("nan")
    rep = {"scene_mm": scene_mm,
           "obj_mm": mae[0] * 1000.0, "place_mm": mae[2] * 1000.0,
           "mu": mae[4], "mass": mae[5],
           "hard_mur": hard_rel, "under": under, "best_val": best_val}
    return best_val, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/planner_train.npz")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1200, 2400, 4800,
                                                             9600, 10800])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=str, default="ckpts/planner_sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    d = np.load(args.data)
    im, lab = d["images"], d["labels"]
    print(f"data: {im.shape} {lab.shape}")
    vim, vlab = im[int(0.9 * len(im)):], lab[int(0.9 * len(im)):]
    train_im, train_lab = im[:int(0.9 * len(im))], lab[:int(0.9 * len(im))]
    n_max = len(train_im)

    print(f"{'N':>6} {'steps':>6} {'sceneMAE':>9} {'obj':>6} {'place':>7} "
          f"{'μ':>6} {'m':>6} {'hard m/μ rel':>12} {'under%':>7} "
          f"{'valMSE':>8} {'wall_s':>7}")
    results = {}
    for n in args.sizes:
        if n > n_max:
            print(f"skip {n} > train max {n_max}")
            continue
        steps = max(200, int(n / args.batch) * args.epochs)
        t0 = time.time()
        tag = os.path.join(args.out, f"N{n}")
        bv, rep = train_one(train_im[:n], train_lab[:n], vim, vlab, steps,
                            args.batch, tag, device)
        wall = time.time() - t0
        rep.update({"n": n, "steps": steps, "wall": wall})
        results[n] = rep
        print(f"{n:>6} {steps:>6} {rep['scene_mm']:>9.2f} {rep['obj_mm']:>6.2f} "
              f"{rep['place_mm']:>7.2f} {rep['mu']:>6.3f} {rep['mass']:>6.3f} "
              f"{rep['hard_mur'] * 100:>11.1f}% {rep['under'] * 100:>6.0f}% "
              f"{rep['best_val']:>8.4f} {wall:>7.0f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
