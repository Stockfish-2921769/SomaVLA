#!/usr/bin/env python3
"""Axis A — NCA vs CNN CAPACITY scaling on the spatial plan-field task.

Independent variable: model size (trainable params). Dependent: rho / mse / prog
on the four generalization axes. Two falsifiable questions:

  1. Does the NCA's advantage over the fixed-depth CNN WIDEN with scale?
  2. Can a CNN close the long-range / scale gap by growing DEPTH? Its receptive
     field is 1+2*depth, so a deep-enough CNN (depth~32 -> RF=65) can in
     principle "see" a G=64 wall -- but at linear param cost. If a huge CNN
     still fails, the gap is inductive bias (iteration), not capacity.

Trains every config on mixed G32 (subprocess, GPU pinned) then evaluates.
Run:  python scripts/sweep_spatial_scale.py --gpu cuda:0 --steps 4000
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

AXES = [
    ("in-dist G32", 32, "mixed", (0, 3)),
    ("scale G64", 64, "mixed", (0, 3)),
    ("complex o4-8", 32, "random", (4, 8)),
    ("longrange wall", 32, "wall", None),
]

# name, model, build kwargs  (ladders chosen to span ~1e4 .. ~2e5 params)
CONFIGS = [
    ("nca_xs", "nca", dict(k_steps=48, hidden=32, morphogen_dim=4, msg_dim=16)),
    ("nca_s", "nca", dict(k_steps=48, hidden=48, morphogen_dim=8, msg_dim=24)),
    ("nca_m", "nca", dict(k_steps=48, hidden=96, morphogen_dim=8, msg_dim=40)),
    ("nca_l", "nca", dict(k_steps=48, hidden=192, morphogen_dim=16, msg_dim=64)),
    ("cnn_s", "cnn", dict(depth=2, channels=12)),
    ("cnn_m", "cnn", dict(depth=4, channels=24)),
    ("cnn_l", "cnn", dict(depth=8, channels=24)),
    ("cnn_xl", "cnn", dict(depth=16, channels=24)),
    ("cnn_xxl", "cnn", dict(depth=32, channels=24)),
]


def train(cfg_name, model, kw, args):
    out = os.path.join(args.out, "scale")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "scripts/train_spatial_nca.py",
           "--model", model, "--family", "mixed", "--norm-c", str(args.norm_c),
           "--steps", str(args.steps), "--val-every", str(args.val_every),
           "--tag", f"_{cfg_name}", "--out", out, "--device", args.gpu,
           "--seed", str(args.seed)]
    if model == "nca":
        cmd += ["--k-steps", str(kw["k_steps"]), "--k-lo", "48", "--k-hi", "96",
                "--hidden", str(kw["hidden"]),
                "--morphogen-dim", str(kw["morphogen_dim"]),
                "--msg-dim", str(kw["msg_dim"])]
    elif model == "cnn":
        cmd += ["--cnn-depth", str(kw["depth"]),
                "--cnn-channels", str(kw["channels"])]
    print(f"\n=== train {cfg_name} ({model}) ===", flush=True)
    subprocess.run(cmd, check=True)


def load(cfg_name, model, kw, out, device):
    path = os.path.join(out, "scale", f"spatial_{model}_{cfg_name}.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    kw = {**kw, "k_steps": ck.get("k_steps", 48)} if model == "nca" else kw
    m = build_model(model, **kw).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="cuda:0")
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
        for cfg_name, model, kw in CONFIGS:
            train(cfg_name, model, kw, args)

    rows = []
    for cfg_name, model, kw in CONFIGS:
        m, nparam = load(cfg_name, model, kw, args.out, device)
        rec = {"name": cfg_name, "model": model, "params": int(nparam)}
        for ai, (aname, g, family, obs) in enumerate(AXES):
            rng = np.random.RandomState(args.seed + 101 * (ai + 1))
            r = run_axis(m, model, g, args.n, rng, device, family, obs or (0, 3),
                         norm_c=args.norm_c)
            rec[aname] = {"rho": r["rho"], "mse": r["mse"], "prog": r["prog"]}
        rows.append(rec)
        print(f"  {cfg_name:9s} {model:4s} {nparam:7,d}  " +
              "  ".join(f"{aname.split()[0]}:{rec[aname]['rho']:+.2f}/"
                        f"{rec[aname]['prog']*100:4.0f}%" for aname, *_ in AXES),
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "scale_sweep.json")
    with open(dst, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved → {dst}")

    print(f"\n{'cfg':>9} {'model':>4} {'params':>8} | " +
          " | ".join(f"{a:>22}" for a, *_ in AXES))
    for rec in rows:
        cells = [f"{rec[a]['rho']:+.2f} {rec[a]['mse']:.3f} {rec[a]['prog']*100:4.0f}%"
                 for a, *_ in AXES]
        print(f"{rec['name']:>9} {rec['model']:>4} {rec['params']:>8,} | " +
              " | ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
