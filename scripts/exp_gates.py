#!/usr/bin/env python3
"""Quick config ablation for the shared-model gate (a) smoke test.

Tests how firing schedule × drift-loss mode affect re-convergence. Training a
SINGLE shared expert on all skills (weak setup — per-skill experts should do
better); the point is to find a config where approach re-converges, not just
echoes the drifted path.

Run from repo root:
    python scripts/exp_gates.py --firing fixed0.5 --drift dense --drift-max 0.5
    python scripts/exp_gates.py --firing one     --drift dense --drift-max 0.5
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
from soma.gates import loss_batch, gate_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    # firing: fixed0.5 | fixed1.0 | one (anneal 0.5 -> 1.0 over last 70%)
    ap.add_argument("--firing", choices=["fixed0.5", "fixed1.0", "one"],
                    default="fixed0.5")
    # drift loss: dense (re-converge to clean next pose) | terminal-only
    ap.add_argument("--drift", choices=["dense", "terminal"], default="dense")
    ap.add_argument("--drift-max", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BodyGraphNCA(k_steps=args.k_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()

    anneal_start = int(args.steps * 0.3)
    for step in range(1, args.steps + 1):
        dp = (0.0 if step <= anneal_start
              else min(args.drift_max, args.drift_max * (step - anneal_start)
                       / (args.steps - anneal_start)))
        # firing schedule
        if args.firing == "fixed0.5":
            model.firing = 0.5
        elif args.firing == "fixed1.0":
            model.firing = 1.0
        else:  # "one": anneal to 1.0 over the last 70% (keeps early robustness)
            model.firing = (0.5 if step <= anneal_start
                            else 0.5 + 0.5 * (step - anneal_start) / (args.steps - anneal_start))

        skill = SKILLS[rng.randint(len(SKILLS))]
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        T = int(rng.randint(lo, hi + 1))
        opt.zero_grad()
        loss = loss_batch(model, device, rng, skill, T, args.batch,
                          drift_prob=dp, drift_mode=args.drift)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 400 == 0 or step == 1:
            print(f"[train] step {step:5d}/{args.steps} loss {loss.item():.4f}")

    print(f"\n== config firing={args.firing} drift={args.drift} drift-max={args.drift_max} ==")
    for skill, err, conv in gate_a(model, device, rng):
        print(f"  {skill:10s} term-pos {err:6.1f} mm   converge {conv*100:5.1f}%")


if __name__ == "__main__":
    main()
