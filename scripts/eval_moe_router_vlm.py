#!/usr/bin/env python3
"""Phase 4f-4 — gate (d) re-run with the SigLIP VLM front-end.

End-to-end: scene → render → SigLIP perceive → scene_pred → MoE router →
skill + boundary conditions. Reports, on held-out draws:

  * perception error       (scene_pred vs ground truth, mm)      — nuisance
  * routing accuracy       (pred vs true skill)     — PASS if >= 95%
  * goal error             (correctly-routed samples, true mask) — PASS if
                            pos < 10 mm  (now = perception error + regression)
  * duration error         (mean |pred − T|, steps)
  * a side-by-side vs ground-truth-scene baseline (same episodes)

Run from the repo root:
    python scripts/eval_moe_router_vlm.py [--n 500] [--router-ckpt checkpoints/moe_router.pt] [--perc-ckpt checkpoints/perception.pt]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.moe_router import StateRouter
from soma.perceive import SigLIPPerception, normalize_scene, denormalize_scene, preprocess_batch
from soma.skill_experts import SKILLS, SKILL_REGISTRY
import proc_sim
from render_scene import render_scene


def collect(rng, n_per_skill):
    """Per-skill episodes; returns images + ground-truth router inputs."""
    imgs, scenes, states, idxs, goals, masks, ts = [], [], [], [], [], [], []
    for skill in SKILLS:
        for _ in range(n_per_skill):
            obj, place = proc_sim._sample_scene(rng)
            states_s, goal, mask, T = proc_sim.sample_skill(skill, rng, scene=(obj, place))
            imgs.append(render_scene(obj, place, rng=rng))
            scenes.append(np.concatenate([obj, place]))
            states.append(states_s[0].copy())
            idxs.append(SKILLS.index(skill))
            goals.append(goal)
            masks.append(mask)
            ts.append(T)
    scenes = torch.from_numpy(np.stack(scenes))
    states = torch.from_numpy(np.stack(states))
    idxs = torch.tensor(idxs)
    goals = torch.from_numpy(np.stack(goals))
    masks = torch.from_numpy(np.stack(masks))
    ts = torch.tensor(ts, dtype=torch.float32)
    return imgs, scenes, states, idxs, goals, masks, ts


@torch.no_grad()
def route_eval(router, scene, states, idxs, goals, masks, ts, device):
    """Router metrics for a given scene tensor [N,4]."""
    scene, states = scene.to(device), states.to(device)
    logits, goals_all, dur_pred = router(scene, states)
    pred = logits.argmax(-1)
    goal_pred = goals_all[torch.arange(len(pred)), pred]
    acc = (pred.cpu() == idxs).float().mean().item()

    per_skill = []
    for s, skill in enumerate(SKILLS):
        sel = (idxs == s) & (pred.cpu() == s)
        n = sel.sum().item()
        if n == 0:
            per_skill.append((skill, 0, float("nan")))
            continue
        m = masks[sel]
        d = (goal_pred[sel].cpu() - goals[sel]).abs()
        pos = (m[:, :3] * d[:, :3]).sum() / m[:, :3].sum().clamp(min=1)
        per_skill.append((skill, n, float(pos * 1000)))

    dur_err = (dur_pred.cpu() - ts).abs().mean().item()
    return acc, per_skill, dur_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--perc-ckpt", type=str, default="checkpoints/perception.pt")
    ap.add_argument("--seed", type=int, default=1, help="held-out draws")
    ap.add_argument("--n", type=int, default=500, help="samples per skill")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = StateRouter().to(device)
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device, weights_only=True)["model"])
    router.eval()
    perc = SigLIPPerception().to(device)
    perc.load_state_dict(torch.load(args.perc_ckpt, map_location=device, weights_only=True)["model"])
    perc.eval()

    rng = np.random.RandomState(args.seed)
    imgs, scenes, states, idxs, goals, masks, ts = collect(rng, args.n)

    print(f"── gate (d) + SigLIP VLM (held-out seed={args.seed}, {args.n}/skill) ──")

    # Perception error (scene_pred vs true, mm). Chunk the frozen-encoder
    # forward — SigLIP activations for the full 3000-image set don't fit GPU.
    with torch.no_grad():
        scene_pred_norm = torch.cat([perc(preprocess_batch(imgs[i:i+64], device))
                                     for i in range(0, len(imgs), 64)])
    scene_pred = torch.from_numpy(denormalize_scene(scene_pred_norm.cpu().numpy()))
    perc_err = (scene_pred - scenes).abs()
    print(f"perception error: {perc_err.mean().item()*1000:6.1f} mm overall  "
          f"per-dim {[round(float(x),1) for x in perc_err.mean(0).numpy()*1000]} mm")

    acc_vlm, per_skill_vlm, dur_vlm = route_eval(router, scene_pred, states, idxs, goals, masks, ts, device)
    acc_gt, per_skill_gt, dur_gt = route_eval(router, scenes, states, idxs, goals, masks, ts, device)

    print(f"\nrouting accuracy:  VLM {acc_vlm*100:5.1f}%   GT {acc_gt*100:5.1f}%   "
          f"{'PASS' if acc_vlm >= 0.95 else 'FAIL'} (threshold 95%)")

    goal_ok = True
    print(f"\nactive-pos goal error (mm, correctly-routed):")
    print(f"  {'skill':10s} {'VLM n':>6s} {'VLM mm':>8s}  {'GT mm':>8s}")
    for s, skill in enumerate(SKILLS):
        n_v, mm_v = per_skill_vlm[s][1], per_skill_vlm[s][2]
        mm_g = per_skill_gt[s][2]
        pas = np.isfinite(mm_v) and mm_v < 10.0
        goal_ok = goal_ok and pas
        print(f"  {skill:10s} {n_v:6d} {mm_v:8.2f}  {mm_g:8.2f}  {'PASS' if pas else 'FAIL'}")
    print(f"duration error: VLM {dur_vlm:.1f} steps   GT {dur_gt:.1f} steps")

    print(f"\ngate (d) + VLM: {'ALL PASS' if acc_vlm >= 0.95 and goal_ok else 'SOME FAIL'}")
    print(f"  (goal error now = perception localisation + regression noise; "
          f"GT column is the zero-perception-error floor)")


if __name__ == "__main__":
    main()
