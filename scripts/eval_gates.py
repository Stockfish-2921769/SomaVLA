#!/usr/bin/env python3
"""Gate framework smoke test: quick-train a SINGLE shared body-graph NCA on
all skills (boundary-condition differentiated) and report gates (a) + (c).

The real per-skill expert training is scripts/train_bodygraph_nca.py.

Run from the repo root:
    python scripts/eval_gates.py --train-steps 600 --batch 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.bodygraph_nca import BodyGraphNCA
from soma.skill_experts import SKILL_REGISTRY, SKILLS
from soma.gates import loss_batch, gate_a, gate_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BodyGraphNCA(k_steps=args.k_steps).to(device)
    print(f"[Soma] BodyGraphNCA params = {model.count_parameters():,} | device={device}")

    print("\n── gate (a) BEFORE training (random init) ──")
    for skill, err, conv in gate_a(model, device, rng):
        print(f"  {skill:10s} term-pos {err:6.1f} mm   converge {conv*100:5.1f}%")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    for step in range(1, args.train_steps + 1):
        skill = SKILLS[rng.randint(len(SKILLS))]
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        T = int(rng.randint(lo, hi + 1))
        opt.zero_grad()
        loss = loss_batch(model, device, rng, skill, T, args.batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"[train] step {step:5d}/{args.train_steps} loss {loss.item():.4f}")

    print("\n── gate (a) AFTER training ──")
    for skill, err, conv in gate_a(model, device, rng):
        print(f"  {skill:10s} term-pos {err:6.1f} mm   converge {conv*100:5.1f}%")

    print("\n── gate (c) morphogen emergence (approach) ──")
    n_dim = 0
    for cell, maxc, nd, act in gate_c(model, device, rng):
        n_dim += nd
        print(f"  cell {cell}  max|corr| {maxc:.2f}  qualifying dims: {nd}  peak activity {act:.3f}")
    print(f"  → {n_dim} dim(s) with |corr(h,u)|>0.5 AND within-step activity>0.05  (criterion: ≥2)")
    print("  PASS" if n_dim >= 2 else "  not yet — expected after real training")


if __name__ == "__main__":
    main()
