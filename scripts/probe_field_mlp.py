#!/usr/bin/env python3
"""E0a informativeness probe: can a small MLP read the teacher's next waypoint
from the local field patch + state + subgoal?

Held-out split is by scene (episode id), so the probe never sees a test scene's
layout.  Reports held-out MAE (cm) and direction cosine.  A model without the
field patch is the informativeness baseline.

Run:
  /home/ubuntu/venvs/lerobot/bin/python scripts/probe_field_mlp.py \
      --data data/multi_lowlevel_2w.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn


def build_xy(d, radius):
    """Inputs: local field patch around EEF + state + subgoal; target delta."""
    f = d["fields"].astype(np.float32)              # [N,4,G,G]
    sc = d["metas"][:, 4].astype(int)               # row
    sj = d["metas"][:, 5].astype(int)               # col
    st = d["states"].astype(np.float32).copy()
    st[:, 3:6] = 0.0
    st = (st - d["st_mean"]) / d["st_std"]
    sub = d["subgoals"].astype(np.float32)
    extra = np.concatenate([st, sub], 1)
    delta = d["actions"][:, 0, :2] - d["states"][:, :2]
    if radius <= 0:
        return extra, delta, d["metas"][:, 0].astype(int)
    r = radius
    pad = np.pad(f, ((0, 0), (0, 0), (r, r), (r, r)), mode="edge")
    patches = np.empty((len(f), 4, 2 * r + 1, 2 * r + 1), np.float32)
    for i in range(len(f)):
        a, b = sc[i], sj[i]
        patches[i] = pad[i, :, a:a + 2 * r + 1, b:b + 2 * r + 1]
    x = np.concatenate([patches.reshape(len(f), -1), extra], 1)
    return x, delta, d["metas"][:, 0].astype(int)


class MLP(nn.Module):
    def __init__(self, d_in, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/multi_lowlevel_2w.npz")
    ap.add_argument("--radius", type=int, default=8,
                    help="local patch radius in cells (0 = no field input)")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    z = np.load(args.data)
    d = {k: z[k] for k in z.files}
    x, delta, scene = build_xy(d, args.radius)
    scenes = np.unique(scene)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(scenes)
    n_ho = max(1, int(len(scenes) * args.holdout_frac))
    ho = set(scenes[:n_ho].tolist())
    te = np.array([s in ho for s in scene])
    tr = ~te
    print(f"data {args.data}: {len(x)} samples, {len(scenes)} scenes, "
          f"holdout {n_ho} scenes / {te.sum()} samples, "
          f"input dim {x.shape[1]} (radius {args.radius})")

    xm, xs = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
    dm, ds = delta[tr].mean(0), np.maximum(delta[tr].std(0), 1e-6)
    xt = torch.from_numpy(((x - xm) / xs).astype(np.float32))
    yt = torch.from_numpy(((delta - dm) / ds).astype(np.float32))
    tr_idx = torch.from_numpy(np.where(tr)[0])
    te_idx = torch.from_numpy(np.where(te)[0])
    model = MLP(x.shape[1]).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    for step in range(1, args.steps + 1):
        bi = tr_idx[torch.randint(len(tr_idx), (args.batch,))]
        pred = model(xt[bi].to(args.device))
        loss = lossf(pred, yt[bi].to(args.device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step} train loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(xt[te_idx].to(args.device)).cpu().numpy() * ds + dm
    true = delta[te_idx.numpy()]
    mae = np.linalg.norm(pred - true, axis=1)
    cos = (pred * true).sum(1) / np.maximum(
        np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1), 1e-9)
    base = np.zeros_like(true) + dm
    mae0 = np.linalg.norm(base - true, axis=1)
    print(f"  held-out MAE {mae.mean()*100:.2f} cm "
          f"(constant-mean baseline {mae0.mean()*100:.2f} cm)")
    print(f"  direction cosine mean {cos.mean():.3f} "
          f"median {np.median(cos):.3f}  >0.5: {np.mean(cos > 0.5)*100:.0f}%")
    print(f"  step-size pred {np.linalg.norm(pred,axis=1).mean()*100:.2f} cm "
          f"true {np.linalg.norm(true,axis=1).mean()*100:.2f} cm")


if __name__ == "__main__":
    main()
