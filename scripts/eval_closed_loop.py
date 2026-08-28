#!/usr/bin/env python3
"""Phase 5c — closed-loop pick-and-place evaluation (gates e, f, g).

Runs the event-driven controller + self-contained sim env end-to-end and
reports:
  (e) closed-loop assembly — per-skill completion + episode success rate
  (f) robustness — mid-episode perturbation still completes the task
  (g) inference time — per-step latency (expert relax + controller) and the
      sustained closed-loop control rate (Hz)
Perception: --percept gt (ground-truth scene) or siglip (render → SigLIP →
scene). plant_f is the env's per-step execution fraction; --sweep-f runs a
small ablation over it.

Run from the repo root:
    python scripts/eval_closed_loop.py --n 20 --percept gt
    python scripts/eval_closed_loop.py --n 20 --percept siglip
    python scripts/eval_closed_loop.py --n 10 --robust
    python scripts/eval_closed_loop.py --n 8 --sweep-f
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from soma.skill_experts import SKILLS
from sim_env import SimEnv
import proc_sim


def load_experts(ckpt_dir, device):
    from soma.bodygraph_nca import BodyGraphNCA
    experts = {}
    for skill in SKILLS:
        path = os.path.join(ckpt_dir, f"bodygraph_{skill}.pt")
        m = BodyGraphNCA().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
        experts[skill] = m
    return experts


def run_episode(experts, router, scene, env, rng, device, perturb=None):
    """One episode. Returns (success, step_count, mean_step_ms, ctl)."""
    ctl = BodyGraphController(experts, router, device=device)
    ctl.reset(scene, env.state.copy())
    t, perturbed, lats = 0, False, []
    while t < 800:
        if perturb is not None and not perturbed and ctl.skill == "lift":
            env.perturb(perturb)
            perturbed = True
        t0 = time.perf_counter()
        target, info = ctl.step(env.state)
        lats.append((time.perf_counter() - t0) * 1000.0)
        env.step(target)
        t += 1
        if info["task_done"]:
            break
    return env.success(), t, float(np.mean(lats)), ctl


def report(groups, n, percept, device):
    """groups: list of dicts with success, steps, latencies, route_log, skill."""
    print(f"── gates (e)(f)(g) closed-loop ({percept} scene, {n} eps, {device}) ──")

    suc = sum(g["success"] for g in groups)
    steps = [g["steps"] for g in groups]
    lat = [g["lat"] for g in groups]
    print(f"(e) episode success: {suc}/{n} ({suc/n*100:.0f}%)   "
          f"mean steps {np.mean(steps):.0f}")

    per_skill = {s: {"routed": 0, "done": 0} for s in SKILLS}
    for g in groups:
        final = g.get("final_skill")
        for entry in g["route_log"]:
            s = entry["skill"]
            per_skill[s]["routed"] += 1
            if entry["completed"]:
                per_skill[s]["done"] += 1
        if final is not None:
            per_skill[final]["routed"] += 1
            per_skill[final]["done"] += 1
    print("  per-skill completion (routed→done):")
    for s in SKILLS:
        r, d = per_skill[s]["routed"], per_skill[s]["done"]
        print(f"    {s:10s} {d:3d}/{r:3d}")

    print(f"(g) inference time: mean step {np.mean(lat):.2f} ms  →  "
          f"sustained {1000/np.mean(lat):.0f} Hz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--percept", choices=["gt", "siglip"], default="gt")
    ap.add_argument("--plant-f", type=float, default=0.5)
    ap.add_argument("--robust", action="store_true", help="perturb mid-episode")
    ap.add_argument("--sweep-f", action="store_true")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--perc-ckpt", type=str, default="checkpoints/perception.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    experts = load_experts(args.ckpt_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])
    perc = None
    if args.percept == "siglip":
        from soma.perceive import SigLIPPerception
        from render_scene import render_scene
        perc = SigLIPPerception().to(device).eval()
        perc.load_state_dict(torch.load(args.perc_ckpt, map_location=device,
                                        weights_only=True)["model"])

    f_values = [0.3, 0.5, 1.0] if args.sweep_f else [args.plant_f]
    for f in f_values:
        groups = []
        for _ in range(args.n):
            obj, place = proc_sim._sample_scene(rng)
            env = SimEnv(rng=rng, plant_f=f)
            env.reset((obj, place))
            if args.percept == "gt":
                scene = np.concatenate([obj, place]).astype(np.float32)
            else:
                img = render_scene(obj, place, rng=rng)
                scene = perc.perceive(img).astype(np.float32)
            ok, steps, lat, ctl = run_episode(experts, router, scene, env, rng,
                                              device, perturb=0.01 if args.robust else None)
            groups.append({"success": ok, "steps": steps, "lat": lat,
                           "route_log": ctl.route_log,
                           "final_skill": ctl.skill if ok else None})
        report(groups, args.n, args.percept, device)
        if args.sweep_f:
            print(f"  (plant_f = {f})\n")


if __name__ == "__main__":
    main()
