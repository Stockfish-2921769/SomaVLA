#!/usr/bin/env python3
"""Axis B — multi-goal capability probe (the "what to pick next" primitive).

Same homogeneous-field regression, but the scene now holds k goals and the
target is the GEODESIC DISTANCE TO THE NEAREST goal (multi-source Dijkstra).
The field's argmin region = the goal to head for, so this is the spatial
version of the MoE router's "what to do next" -- now learned as a field.

TRAIN on k in {1..3}; TEST on k in {1,3,4,6,8}. The out-of-range counts
(k=4,6,8) are the count-generalization probe: does one shared iteration rule
handle MORE targets than it ever saw, without retraining or an extra head?

Runs the four param-matched models. Run:
  python scripts/probe_multigoal.py --gpu cuda:1 --steps 4000
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
from eval_spatial_plan import run_axis

MODELS = ["nca", "cnn", "transformer", "mlp"]
EVAL_COUNTS = [1, 3, 4, 6, 8]


def train(model, args):
    out = os.path.join(args.out, "multigoal")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "scripts/train_spatial_nca.py",
           "--model", model, "--family", "random", "--norm-c", str(args.norm_c),
           "--steps", str(args.steps), "--val-every", str(args.val_every),
           "--n-goals-lo", "1", "--n-goals-hi", "3",
           "--tag", "_mg", "--out", out, "--device", args.gpu,
           "--seed", str(args.seed)]
    if model == "nca":
        cmd += ["--k-steps", "48", "--k-lo", "48", "--k-hi", "96"]
    print(f"\n=== train {model} (multi-goal k=1..3) ===", flush=True)
    subprocess.run(cmd, check=True)


def load(model, out, device):
    path = os.path.join(out, "multigoal", f"spatial_{model}_mg.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    kw = {"k_steps": ck.get("k_steps", 48)} if model == "nca" else {}
    m = build_model(model, **kw).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="cuda:1")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm-c", type=float, default=64.0)
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
        for ng in EVAL_COUNTS:
            rng = np.random.RandomState(args.seed + ng)
            r = run_axis(m, model, 32, args.n, rng, device, "random", (0, 3),
                         norm_c=args.norm_c, n_goals=ng)
            rec[f"k={ng}"] = {"rho": r["rho"], "mse": r["mse"], "prog": r["prog"]}
        rows.append(rec)
        print(f"  {model:11s} {nparam:7,d}  " +
              "  ".join(f"k{ng}:{rec[f'k={ng}']['rho']:+.2f}/"
                        f"{rec[f'k={ng}']['prog']*100:4.0f}%" for ng in EVAL_COUNTS),
              flush=True)

    dst = os.path.join(args.out, "multigoal_probe.json")
    with open(dst, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved → {dst}")

    print(f"\n{'model':>11} {'params':>8} | " +
          " | ".join(f"{('k=' + str(ng)):>20}" for ng in EVAL_COUNTS))
    for rec in rows:
        cells = [f"{rec[f'k={ng}']['rho']:+.2f} {rec[f'k={ng}']['mse']:.3f} "
                 f"{rec[f'k={ng}']['prog']*100:4.0f}%" for ng in EVAL_COUNTS]
        print(f"{rec['model']:>11} {rec['params']:>8,} | " +
              " | ".join(f"{c:>20}" for c in cells))
    print("\nreading: k=1,3 in train distribution; k=4,6,8 are count generalization.")


if __name__ == "__main__":
    main()
