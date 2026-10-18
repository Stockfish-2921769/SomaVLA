#!/usr/bin/env python3
"""Capability probe — HARMONIC (Laplace) POTENTIAL FIELD. A physical field.

Scene = the same cluttered grid; target = the steady state of ∇²φ=0 with
Dirichlet φ=0 at the goal and φ=1 at obstacles (electrostatics / steady
diffusion). Computing it IS iterative relaxation — the NCA's native operation,
so a fixed-depth CNN should be structurally unable to solve it at scale.

Trains the four param-matched models on mixed G32 and evaluates the four
generalization axes (density / scale / long-range). Reports rho, MSE and the
roll-out `prog` (greedy descent to the goal, geodesic-referenced).
Run:  python scripts/probe_physics_harmonic.py --gpu cuda:1 --steps 4000
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
from spatial_plan_data import sample_scene, scene_to_harmonic_targets, geodesic_field
from eval_spatial_plan import greedy_descent
from field_eval import eval_field

MODELS = ["nca", "cnn", "transformer", "mlp"]
AXES = [
    ("in-dist G32", 32, "mixed", (0, 3)),
    ("complex o4-8", 32, "random", (4, 8)),
    ("scale G64", 64, "mixed", (0, 3)),
    ("longrange wall", 32, "wall", None),
]


def gen_harmonic(family, obs_range):
    def gen(g, rng):
        obstacle, start, goal = sample_scene(g, rng, family, obs_range or (0, 3))
        ext, tar, val = scene_to_harmonic_targets(obstacle, goal, start)
        if val.sum() < 2:
            return None
        return ext, tar, val, (obstacle, start, goal)
    return gen


def prog_fn(field, meta):
    obstacle, start, goal = meta
    g = obstacle.shape[0]
    dist, _ = geodesic_field(obstacle, goal)
    d0 = dist[start]
    if not np.isfinite(d0) or d0 <= 0:
        return None
    fld = np.where(obstacle, np.inf, field)
    pos = greedy_descent(fld, obstacle, start, goal, 3 * g)
    return (d0 - dist[pos]) / d0


def train(model, args):
    out = os.path.join(args.out, "harmonic")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "scripts/train_spatial_nca.py",
           "--model", model, "--family", "mixed", "--norm-c", str(args.norm_c),
           "--target", "harmonic", "--steps", str(args.steps),
           "--val-every", str(args.val_every), "--tag", "_harm", "--out", out,
           "--device", args.gpu, "--seed", str(args.seed)]
    if model == "nca":
        cmd += ["--k-steps", "48", "--k-lo", "48", "--k-hi", "96"]
    print(f"\n=== train {model} (harmonic) ===", flush=True)
    subprocess.run(cmd, check=True)


def load(model, out, device):
    path = os.path.join(out, "harmonic", f"spatial_{model}_harm.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(model, **ck.get("arch", {"k_steps": 48} if model == "nca" else {}))
    m.load_state_dict(ck["model"])
    m = m.to(device)
    m.eval()
    return m, sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="cuda:1")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--n", type=int, default=96)
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
        for ai, (aname, g, family, obs) in enumerate(AXES):
            rng = np.random.RandomState(args.seed + 101 * (ai + 1))
            r = eval_field(m, model, gen_harmonic(family, obs), g, args.n, rng,
                           device, progress_fn=prog_fn)
            rec[aname] = r
        rows.append(rec)
        print(f"  {model:11s} {nparam:7,d}  " +
              "  ".join(f"{a.split()[0]}:{rec[a]['rho']:+.2f}/"
                        f"{rec[a]['prog']*100:4.0f}%" for a, *_ in AXES), flush=True)

    dst = os.path.join(args.out, "harmonic_probe.json")
    with open(dst, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved → {dst}")
    print(f"\n{'model':>11} {'params':>8} | " +
          " | ".join(f"{a:>22}" for a, *_ in AXES))
    for rec in rows:
        cells = [f"{rec[a]['rho']:+.2f} {rec[a]['mse']:.3f} {rec[a]['prog']*100:4.0f}%"
                 for a, *_ in AXES]
        print(f"{rec['model']:>11} {rec['params']:>8,} | " + " | ".join(f"{c:>22}" for c in cells))
    print("\nreading: harmonic potential field (Laplace). rho = field validity; "
          "prog = descent to goal.")


if __name__ == "__main__":
    main()
