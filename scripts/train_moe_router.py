#!/usr/bin/env python3
"""Phase 4d — train the state-only MoE router (BC on procedural episodes).

The router maps (ground-truth scene, current EEF state) to a skill choice
plus boundary conditions. Loss = CE(routing) + masked-MSE(goal, active DOFs,
sigma-normalized) + L1(duration). Saves checkpoints/moe_router.pt and prints
per-step routing accuracy + goal error.

Run from the repo root:
    python scripts/train_moe_router.py --steps 3000 --batch 64
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
from soma.bodygraph_nca import SIGMA
import proc_sim


def make_batch(rng, batch):
    scenes, states, skill_idxs, goals, masks, ts = [], [], [], [], [], []
    for _ in range(batch):
        sc, st, idx, g, m, T = proc_sim.router_sample(rng)
        scenes.append(sc)
        states.append(st)
        skill_idxs.append(idx)
        goals.append(g)
        masks.append(m)
        ts.append(T)
    return (torch.from_numpy(np.stack(scenes)),
            torch.from_numpy(np.stack(states)),
            torch.tensor(skill_idxs, dtype=torch.long),
            torch.from_numpy(np.stack(goals)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(ts, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-goal", type=float, default=1.0)
    ap.add_argument("--w-dur", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model = StateRouter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sigma = torch.tensor(SIGMA, dtype=torch.float32, device=device)

    print(f"StateRouter params: {model.count_parameters()}")
    for step in range(1, args.steps + 1):
        scenes, states, idxs, goals, masks, ts = make_batch(rng, args.batch)
        scenes, states, idxs = scenes.to(device), states.to(device), idxs.to(device)
        goals, masks, ts = goals.to(device), masks.to(device), ts.to(device)

        opt.zero_grad()
        logits, goals_all, dur_pred = model(scenes, states)
        loss_ce = F.cross_entropy(logits, idxs)
        # Per-skill hard supervision: each goal head learns only its own skill's
        # boundary condition (route = CE; goal = that skill's clean head).
        goal_pred = goals_all[torch.arange(args.batch), idxs]
        d = (goal_pred - goals) / sigma
        loss_goal = ((masks * d) ** 2).mean()
        loss_dur = F.l1_loss(dur_pred, ts)
        loss = loss_ce + args.w_goal * loss_goal + args.w_dur * loss_dur
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == 1:
            acc = (logits.argmax(-1) == idxs).float().mean().item()
            # active-DOF goal error in physical units (mm / rad / openness).
            goal_err = ((masks * (goal_pred - goals).abs()) / masks.clamp(min=1)).mean().item()
            print(f"[step {step:5d}/{args.steps}] loss {loss.item():.4f} "
                  f"(ce {loss_ce.item():.3f} goal {loss_goal.item():.4f} dur {loss_dur.item():.1f}) "
                  f"acc {acc*100:5.1f}%  goal_err {goal_err*1000:6.1f}mm-equiv")

    ckpt = os.path.join(args.out, "moe_router.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"saved → {ckpt}")


if __name__ == "__main__":
    main()
