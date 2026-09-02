#!/usr/bin/env python3
"""Control 1 — does the frozen SigLIP feature channel carry the physics info?

Regression probe: pooled frozen SigLIP-B16 features -> margin
(mu*F_max - mass*g0). If a small probe on the *frozen* encoder resolves margin
to ~0.3-0.5 N, the perception half of the SigVLA-tiny structure is not the
bottleneck — the decoder could in principle learn the load-adaptive
contraction from pixels. If the probe cannot, the structure is
information-limited at the encoder.

Uses the same images as VLA training (data/*.npz, metas now carry mu, mass).
Run from the repo root:
  python scripts/probe_margin.py --data data/vla_step.npz --steps 1500
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from soma.physics import F_MAX, G0
from train_vla_tiny import SigVLATiny


class MarginProbe(nn.Module):
    """Mean-pooled frozen patch tokens -> margin."""

    def __init__(self, d=768):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256),
                                 nn.GELU(), nn.Linear(256, 1))

    def forward(self, feats):
        return self.mlp(feats).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/vla_train.npz")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subsample", type=int, default=8000,
                    help="max samples to use (probe speed)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data)
    im, meta = d["images"], d["metas"]
    mu, mass = meta[:, 4], meta[:, 5]
    margin = mu * F_MAX - mass * G0          # N
    mm, ms = margin.mean(), margin.std()
    if len(im) > args.subsample:
        r = np.random.RandomState(args.seed).choice(len(im), args.subsample,
                                                    replace=False)
        im, mu, mass, margin = im[r], mu[r], mass[r], margin[r]
    y = (margin - mm) / ms
    print(f"images {im.shape}  margin N: mean {mm:.2f} std {ms:.2f} "
          f"range [{margin.min():.2f},{margin.max():.2f}]")

    split = int(0.9 * len(im))
    vim, vy = im[split:], y[split:]
    im, y = im[:split], y[:split]

    vision = SigVLATiny(1, 4).vision.to(device)
    for p in vision.parameters():
        p.requires_grad_(False)
    probe = MarginProbe().to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    n = len(im)
    def batch(i):
        sl = slice(i * args.batch, (i + 1) * args.batch)
        return (torch.from_numpy(im[sl]).to(device),
                torch.from_numpy(y[sl]).to(device))

    def feats(x):
        with torch.no_grad():
            t = vision.forward_features(x)   # [B,256,768]
        return t.mean(dim=1)                 # pooled

    steps_per_ep = max(1, n // args.batch)
    for s in range(1, args.steps + 1):
        probe.train()
        img, yy = batch(s % steps_per_ep)
        pred = probe(feats(img))
        loss = nn.functional.mse_loss(pred, yy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0 or s == 1:
            probe.eval()
            with torch.no_grad():
                vp = probe(feats(torch.from_numpy(vim).to(device)))
                vmse = nn.functional.mse_loss(vp, torch.from_numpy(vy).to(device)).item()
                vmae = (vp.cpu().numpy() - vy).__abs__().mean() * ms
            print(f"step {s:5d}  train {loss.item():.5f}  "
                  f"val mse {vmse:.5f}  val MAE {vmae:.2f} N", flush=True)
    print(f"done. probe val MAE {vmae:.2f} N "
          f"({vmae/ms*100:.0f}% of margin std); discriminating band margin "
          f"~0.6-1.1 N (m/mu 1.16-1.26)")


if __name__ == "__main__":
    main()
