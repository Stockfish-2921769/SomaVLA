#!/usr/bin/env python3
"""Gate (d) — state-only MoE router quality report.

Loads checkpoints/moe_router.pt and reports, on HELD-OUT scenes (different
seed than training):
  * routing accuracy (overall + per-skill)          — PASS if >= 95%
  * per-skill goal regression on active DOFs        — PASS if pos < 10 mm
      pos (mm), rot (mrad), openness — masked to the skill's alpha mask
  * alpha-mask consistency (registry lookup)        — PASS by construction
  * duration error (mean |pred − T|, steps)
Plus a mid-skill robustness probe (confusion matrix over mid-skill states —
informational, not a gate).

Run from the repo root:
    python scripts/eval_moe_router.py [--ckpt checkpoints/moe_router.pt]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

from soma.moe_router import StateRouter
from soma.skill_experts import SKILLS, SKILL_REGISTRY
import proc_sim


@torch.no_grad()
def collect(rng, model, device, n_per_skill):
    scenes, states, idxs, goals, masks, ts = [], [], [], [], [], []
    for skill in SKILLS:
        for _ in range(n_per_skill):
            sc, st, idx, g, m, T = proc_sim.router_sample(rng, skill=skill)
            scenes.append(sc)
            states.append(st)
            idxs.append(idx)
            goals.append(g)
            masks.append(m)
            ts.append(T)
    scenes = torch.from_numpy(np.stack(scenes)).to(device)
    states = torch.from_numpy(np.stack(states)).to(device)
    idxs = torch.tensor(idxs, device=device)
    goals = torch.from_numpy(np.stack(goals)).to(device)
    masks = torch.from_numpy(np.stack(masks)).to(device)
    ts = torch.tensor(ts, dtype=torch.float32, device=device)
    return scenes, states, idxs, goals, masks, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--seed", type=int, default=1, help="held-out scene seed")
    ap.add_argument("--n", type=int, default=500, help="samples per skill")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StateRouter().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True)["model"])
    model.eval()
    rng = np.random.RandomState(args.seed)

    scenes, states, idxs, goals, masks, ts = collect(rng, model, device, args.n)
    logits, goals_all, dur_pred = model(scenes, states)
    pred = logits.argmax(-1)
    goal_pred = goals_all[torch.arange(len(pred)), pred]   # hard-routed goal

    print(f"── gate (d) state-only router (held-out seed={args.seed}, {args.n}/skill) ──")

    acc_all = (pred == idxs).float().mean().item()
    print(f"routing accuracy overall: {acc_all*100:.1f}%  "
          f"{'PASS' if acc_all >= 0.95 else 'FAIL'} (threshold 95%)")

    goal_ok = True
    for s, skill in enumerate(SKILLS):
        sel = (idxs == s) & (pred == s)  # correctly-routed samples only
        n = sel.sum().item()
        if n == 0:
            print(f"  {skill:10s} (no correctly-routed samples)")
            goal_ok = False
            continue
        m = masks[sel]                  # true skill's alpha mask
        d = (goal_pred[sel] - goals[sel]).abs()
        # Mask to active DOFs, average per DOF (mean over samples AND active dims).
        pos = (m[:, :3] * d[:, :3]).sum() / m[:, :3].sum().clamp(min=1)
        rot = (m[:, 3:6] * d[:, 3:6]).sum() / m[:, 3:6].sum().clamp(min=1)
        opn = (m[:, 6:] * d[:, 6:]).sum() / m[:, 6:].sum().clamp(min=1)
        pas = float(pos * 1000) < 10.0
        goal_ok = goal_ok and pas
        print(f"  {skill:10s} n={n:4d}  active-pos {pos*1000:6.2f} mm  "
              f"rot {rot*1000:6.2f} mrad  openness {opn:6.3f}  "
              f"{'PASS' if pas else 'FAIL'}")

    # alpha-mask consistency: the routed skill's mask comes from the registry
    # lookup, so it is consistent by construction; its correctness == routing
    # accuracy (a wrong route implies the wrong skill's mask).
    print(f"alpha-mask: registry lookup per routed skill — consistent by "
          f"construction; correctness = routing accuracy {acc_all*100:.1f}%")

    dur_err = (dur_pred - ts).abs().mean().item()
    print(f"duration error: {dur_err:.1f} steps (mean |pred − T|)")

    print("\n── mid-skill robustness probe (informational) ──")
    cm = np.zeros((len(SKILLS), len(SKILLS)), dtype=int)
    for s, skill in enumerate(SKILLS):
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        for _ in range(100):
            T = int(rng.randint(lo, hi + 1))
            states, goal, mask, _ = proc_sim.sample_skill(skill, rng, T=T)
            t = int(T * rng.uniform(0.2, 0.9))
            logits_t, _, _ = model(
                torch.from_numpy(np.stack([np.concatenate([goal[:2], np.array([0.5, 0.4], dtype=np.float32)])])).to(device),
                torch.from_numpy(states[t][None]).to(device))
            cm[s, logits_t.argmax(-1).item()] += 1
    print("rows=true skill, cols=routed skill:")
    print("        " + "".join(f"{k[:4]:>6}" for k in SKILLS))
    for s, skill in enumerate(SKILLS):
        print(f"  {skill[:7]:7s} " + "".join(f"{v:6d}" for v in cm[s]))

    print(f"\ngate (d): {'ALL PASS' if acc_all >= 0.95 and goal_ok else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
