#!/usr/bin/env python3
"""Phase 2 — clean/robust training of per-skill body-graph NCA experts.

Trains one BodyGraphNCA per primitive (6 experts) on procedural sim with a
Smart-Bricks recipe: random rollout length, sparse terminal supervision +
tracking (clean mode), and drift re-convergence (terminal-only mode) with a
curriculum that anneals drift_prob 0 → 0.5. Normalized masked loss. Saves
per-skill checkpoints and reports gate (a).

Run from the repo root:
    python scripts/train_bodygraph_nca.py --steps 3000 --batch 32
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
from soma.gates import loss_batch, loss_loop, gate_a
from soma.physics import PHYSICS_CTX_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--drift-max", type=float, default=0.5)
    ap.add_argument("--mode", choices=["track", "rollin"], default="rollin",
                    help="track=on-path reseed (old, echo-prone); "
                         "rollin=closed-loop plant roll-in (drives to goal)")
    ap.add_argument("--skill", type=str, default=None,
                    help="train only this skill (default: all)")
    ap.add_argument("--physics", action="store_true",
                    help="train with the Coulomb physics modality channel + "
                         "slip loss (Phase 6)")
    ap.add_argument("--w-slip", type=float, default=2.0,
                    help="slip-loss weight when --physics is on")
    ap.add_argument("--hard-frac", type=float, default=0.5,
                    help="fraction of physics samples from the m/μ>1 hard tail")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    train_skills = [args.skill] if args.skill else SKILLS
    for skill in train_skills:
        print(f"\n═══ training expert: {skill} ═══")
        model = BodyGraphNCA(k_steps=args.k_steps,
                             physics_ctx_dim=PHYSICS_CTX_DIM if args.physics else 0).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lo, hi = SKILL_REGISTRY[skill]["duration"]

        model.train()
        losses = []
        anneal_start = args.steps * 0.3
        for step in range(1, args.steps + 1):
            # curriculum: clean first, anneal drift in over the last 70%.
            dp = (0.0 if step <= anneal_start
                  else min(args.drift_max, args.drift_max * (step - anneal_start) / (args.steps - anneal_start)))
            T = int(rng.randint(lo, hi + 1))
            opt.zero_grad()
            if args.mode == "rollin":
                loss = loss_loop(model, device, rng, skill, T, args.batch,
                                 drift_prob=dp,
                                 physics=args.physics, w_slip=args.w_slip,
                                 hard_frac=args.hard_frac)
            else:
                loss = loss_batch(model, device, rng, skill, T, args.batch, drift_prob=dp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % 500 == 0 or step == 1:
                print(f"  [{skill}] step {step:5d}/{args.steps} loss {np.mean(losses[-200:]):.4f}")

        ckpt = os.path.join(args.out, f"bodygraph_{skill}.pt")
        torch.save({"skill": skill, "model": model.state_dict(), "args": vars(args)}, ckpt)
        print(f"  saved → {ckpt}")

        for sname, err, conv in gate_a(model, device, rng):
            if sname == skill:
                summary[skill] = (err, conv)
                print(f"  gate(a) {skill}: term-pos {err:6.1f} mm  converge {conv*100:5.1f}%")

    print("\n═══ per-skill gate (a) summary ═══")
    for skill in train_skills:
        err, conv = summary.get(skill, (float("nan"), 0.0))
        print(f"  {skill:10s} term-pos {err:6.1f} mm  converge {conv*100:5.1f}%")


if __name__ == "__main__":
    main()
