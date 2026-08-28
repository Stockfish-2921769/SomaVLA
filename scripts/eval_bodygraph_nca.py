#!/usr/bin/env python3
"""Gate report for the trained per-skill experts.

Loads checkpoints/bodygraph_{skill}.pt and reports:
  gate (a)  — re-convergence to absolute goal B from off-path reseeds.
  gate (c)  — morphogen emergence: a dim qualifies iff |corr(h,u)|>0.5 AND
              within-step activity > 0.05 (guards the '16k linear re-implem').

Run from the repo root:
    python scripts/eval_bodygraph_nca.py [--ckpt-dir checkpoints]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.bodygraph_nca import BodyGraphNCA, N_CELLS, MORPHOGEN_DIM
from soma.skill_experts import SKILLS
from soma.gates import gate_a_loop, gate_b_loop, gate_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    print("── gate (a) closed-loop skill convergence (drive to goal, plant_f=0.5) ──")
    ok = True
    models = {}
    for skill in SKILLS:
        ckpt_path = os.path.join(args.ckpt_dir, f"bodygraph_{skill}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {skill:10s} MISSING checkpoint {ckpt_path}")
            ok = False
            continue
        model = BodyGraphNCA().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["model"])
        models[skill] = model
    for skill in SKILLS:
        if skill not in models:
            continue
        okn, n = gate_a_loop(models[skill], device, rng, n=20)[SKILLS.index(skill)]
        passed = okn >= 0.9 * n
        ok = ok and passed
        print(f"  {skill:10s} closed-loop drive {okn:2d}/{n:2d}  "
              f"{'PASS' if passed else 'FAIL'}")
    print(f"  gate (a): {'ALL PASS' if ok else 'SOME FAIL'}")

    print("\n── gate (b) closed-loop drift re-convergence (one-time 15mm) ──")
    b_ok = True
    for skill in SKILLS:
        if skill not in models:
            continue
        okn, n = gate_b_loop(models[skill], device, rng, n=20)[SKILLS.index(skill)]
        passed = okn >= 0.9 * n
        b_ok = b_ok and passed
        print(f"  {skill:10s} drive+drift {okn:2d}/{n:2d}  "
              f"{'PASS' if passed else 'FAIL'}")
    print(f"  gate (b): {'ALL PASS' if b_ok else 'SOME FAIL'}")

    print("\n── gate (c) morphogen emergence (approach expert) ──")
    ckpt_path = os.path.join(args.ckpt_dir, "bodygraph_approach.pt")
    if os.path.exists(ckpt_path):
        model = BodyGraphNCA().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["model"])
        n_dim = 0
        for cell, maxc, nd, act in gate_c(model, device, rng):
            n_dim += nd
            print(f"  cell {cell}  max|corr| {maxc:.2f}  qualifying dims: {nd}  peak activity {act:.3f}")
        print(f"  → {n_dim} dim(s) qualify (criterion ≥2)  {'PASS' if n_dim >= 2 else 'FAIL'}")
    else:
        print("  no approach checkpoint — skipped")


if __name__ == "__main__":
    main()
