#!/usr/bin/env python3
"""Corrected per-N report for the planner data-efficiency sweep.

Reloads each ckpts/planner_sweep/N{n}/best.pt (head + per-N norm) and reports
denormalized val metrics on the FIXED 1200-scene val set (last of the 12k),
so the curve is a clean function of train data budget N. Includes the original
full run (ckpts/planner/best.pt, N=10800) as the top of the curve.

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 python scripts/report_planner_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from train_planner import SigLIPPlanner


def report(ckpt_path, vim, vlab, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    planner = SigLIPPlanner().to(device).eval()
    planner.head.load_state_dict(ck["head"])
    mean = np.asarray(ck["norm"]["mean"].cpu())
    std = np.asarray(ck["norm"]["std"].cpu())
    with torch.no_grad():
        vp = planner(torch.from_numpy(vim).to(device)).cpu().numpy()
    vp = vp * std + mean
    mae = np.abs(vp - vlab).mean(axis=0)
    scene_mm = mae[:4].mean() * 1000.0
    r_true = vlab[:, 5] / vlab[:, 4]
    r_pred = vp[:, 5] / vp[:, 4]
    rel = (r_pred - r_true) / r_true
    hard = r_true >= 1.16
    hard_rel = np.abs(rel[hard]).mean() if hard.any() else float("nan")
    under = (rel[hard] < -0.1).mean() if hard.any() else float("nan")
    return scene_mm, mae[0] * 1000, mae[2] * 1000, mae[4], mae[5], hard_rel, under


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load("data/planner_train.npz")
    im, lab = d["images"], d["labels"]
    vim, vlab = im[int(0.9 * len(im)):], lab[int(0.9 * len(im)):]
    print(f"fixed val: {len(vlab)} scenes; hard-frac in val "
          f"{(vlab[:,5]/vlab[:,4] >= 1.16).mean():.2f}")
    print(f"{'N':>6} {'sceneMAE':>9} {'obj':>6} {'place':>7} {'μ':>6} {'m':>6} "
          f"{'hard m/μ rel':>12} {'under%':>7}")
    rows = [(1200, "ckpts/planner_sweep/N1200/best.pt"),
            (2400, "ckpts/planner_sweep/N2400/best.pt"),
            (4800, "ckpts/planner_sweep/N4800/best.pt"),
            (9600, "ckpts/planner_sweep/N9600/best.pt"),
            (10800, "ckpts/planner/best.pt")]
    for n, p in rows:
        sm, o, pl, mu, m, hrel, under = report(p, vim, vlab, device)
        print(f"{n:>6} {sm:>9.2f} {o:>6.2f} {pl:>7.2f} {mu:>6.3f} {m:>6.3f} "
              f"{hrel * 100:>11.1f}% {under * 100:>6.0f}%")


if __name__ == "__main__":
    main()
