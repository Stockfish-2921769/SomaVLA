#!/usr/bin/env python3
"""Capability probe — MOVING-OBSTACLE TIME-TO-CONTACT FIELD (dynamics).

Scene: k blocks with constant integer velocity, bouncing off the walls.
ext [3,G,G] = [obstacle_now, vx, vy]  (obstacle in channel 0 -> the NCA's
frozen-cell gate uses obstacle_ch=0). Target = the normalized FIRST-ARRIVAL
time of any block at each cell (1.0 = never within the horizon, 64 steps).

This requires propagating a wavefront in TIME, and integrating obstacle motion
— a fixed-depth CNN has no clock, so it should fail; the NCA's iterated rule
should propagate the front K steps deep. Generalization: more blocks than
trained (k=5,7 at G32) and scale (G64).

Run:  python scripts/probe_dynamics_ttc.py --gpu cuda:0 --steps 4000
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.spatial_baselines import build_model
from spatial_plan_data import ttc_scene_targets
from field_eval import eval_field

MODELS = ["nca", "cnn", "transformer", "mlp"]
HORIZON = 64
# (label, G, n_obs)  -- train distribution is G32, n_obs in {1,2,3}
AXES = [
    ("k=1 G32", 32, 1),
    ("k=3 G32", 32, 3),
    ("k=5 G32*", 32, 5),
    ("k=7 G32*", 32, 7),
    ("k=2 G64*", 64, 2),
]


def gen_ttc(n_obs, horizon):
    def gen(g, rng):
        ext, tar, val = ttc_scene_targets(g, rng, n_obs=n_obs, horizon=horizon)
        if val.sum() < 2:
            return None
        return ext, tar, val, None
    return gen


def train(model, args):
    out = os.path.join(args.out, "dynamics")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "scripts/train_spatial_nca.py",
           "--model", model, "--family", "random", "--target", "ttc",
           "--horizon", str(args.horizon), "--obs-lo", "0", "--obs-hi", "2",
           "--steps", str(args.steps), "--val-every", str(args.val_every),
           "--tag", "_ttc", "--out", out, "--device", args.gpu,
           "--seed", str(args.seed)]
    if model == "nca":
        cmd += ["--k-steps", "64", "--k-lo", "48", "--k-hi", "96"]
    print(f"\n=== train {model} (time-to-contact) ===", flush=True)
    subprocess.run(cmd, check=True)


def load(model, out, device):
    path = os.path.join(out, "dynamics", f"spatial_{model}_ttc.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(model, **ck.get("arch", {"k_steps": 64} if model == "nca" else {}))
    m.load_state_dict(ck["model"])
    m = m.to(device)
    m.eval()
    return m, sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="cuda:0")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--out", type=str, default="checkpoints_spatial")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = args.gpu
    if not args.skip_train:
        for model in MODELS:
            train(model, args)

    rows = []
    for model in MODELS:
        m, nparam = load(model, args.out, device)
        rec = {"model": model, "params": int(nparam)}
        for ai, (aname, g, k) in enumerate(AXES):
            rng = np.random.RandomState(args.seed + 101 * (ai + 1))
            r = eval_field(m, model, gen_ttc(k, args.horizon), g, args.n, rng, device)
            rec[aname] = r
        rows.append(rec)
        print(f"  {model:11s} {nparam:7,d}  " +
              "  ".join(f"{a.split()[0]}:{rec[a]['rho']:+.2f}" for a, *_ in AXES),
              flush=True)

    dst = os.path.join(args.out, "dynamics_probe.json")
    with open(dst, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved → {dst}")
    print(f"\n{'model':>11} {'params':>8} | " +
          " | ".join(f"{a:>14}" for a, *_ in AXES))
    for rec in rows:
        cells = [f"{rec[a]['rho']:+.2f} {rec[a]['mse']:.3f}" for a, *_ in AXES]
        print(f"{rec['model']:>11} {rec['params']:>8,} | " + " | ".join(f"{c:>14}" for c in cells))
    print("\nreading: time-to-contact field (rho/MSE). '*' = out of train range. "
          "No goal -> no prog.")


if __name__ == "__main__":
    main()
