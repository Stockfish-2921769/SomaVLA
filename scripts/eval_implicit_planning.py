#!/usr/bin/env python3
"""Phase 6 — gate (h): does the homogeneous NCA learn implicit action planning
from the Coulomb physics modality?

Runs the full closed-loop pick-and-place on a grid of feasible (μ, m) scenes
and compares a physics-AWARE expert set (Coulomb modality channel + slip-loss
training) against a BLIND baseline (no physics training). Reports:

  (h1) modality-enabled capability — aware vs blind episode success per cell;
       a physics-aware NCA recovers the hard cells the blind baseline drops.
  (h2) implicit speed plan — transport per-step commanded displacement as a
       function of grip margin μ·F_max − m·g0: the aware expert's step size
       contracts monotonically as the margin shrinks (load-adaptive speed),
       while the blind expert's step size is margin-independent.
  (h3) plan-in-morphogen (probe) — ablating the morphogen warm-start across
       steps tests whether the slow-down plan is carried in the NCA's
       morphogen memory or re-derived from the per-step (μ, m) context. Since
       (μ, m) is constant within an episode, the context is fully informative
       every step, so the (negative) finding is that the plan is a
       context→action policy, not morphogen memory.

Aware experts: {ckpt_dir}/bodygraph_{skill}.pt for the held skills
(grasp/lift/transport/place), else the blind {blind_dir} for approach/release.
Blind baseline: {blind_dir}/bodygraph_{skill}.pt for all six.

Run from the repo root:
    python scripts/eval_implicit_planning.py --n-cells 40 --eps 2
    python scripts/eval_implicit_planning.py --n-cells 30 --eps 3 --no-morph
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from soma.skill_experts import SKILLS
from soma.bodygraph_nca import BodyGraphNCA
from soma.physics import (PHYSICS_CTX_DIM, F_MAX, G0, sample_physics,
                          MASS_RANGE, MU_RANGE, GRASPABLE_FRAC)
from sim_env import SimEnv
import proc_sim

HELD = ("grasp", "lift", "transport", "place")


def load_experts(aware_dir, blind_dir, device):
    """aware: physics-trained held skills + blind approach/release."""
    experts = {}
    for skill in SKILLS:
        if skill in HELD and aware_dir is not None:
            m = BodyGraphNCA(physics_ctx_dim=PHYSICS_CTX_DIM).to(device).eval()
            m.load_state_dict(torch.load(os.path.join(aware_dir, f"bodygraph_{skill}.pt"),
                                         map_location=device, weights_only=True)["model"])
        else:
            m = BodyGraphNCA().to(device).eval()
            m.load_state_dict(torch.load(os.path.join(blind_dir, f"bodygraph_{skill}.pt"),
                                         map_location=device, weights_only=True)["model"])
        experts[skill] = m
    return experts


def run_episode(experts, router, mu, mass, rng, device, morph_ablate=False,
                zero_ctx=False, record=False):
    """One closed-loop episode at cell (mu, mass).

    Returns (success, dropped, transport_steps[], lift_steps[], drop_skill,
    info{per-step}).
    """
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene = np.concatenate([obj, place]).astype(np.float32)
    ctl = BodyGraphController(experts, router, device=device)
    ctl.reset(scene, env.state.copy())
    transport_steps = []
    lift_steps = []
    drop_skill = None
    per_step = [] if record else None
    for t in range(800):
        ctx = None if zero_ctx else env.physics_ctx()
        target, info = ctl.step(env.state, physics_ctx=ctx)
        if morph_ablate:
            ctl.morphogen = None          # no cross-step warm-start memory
        if ctl.skill == "transport":
            transport_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
        elif ctl.skill == "lift":
            lift_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
        if per_step is not None:
            per_step.append((ctl.skill, float(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0),
                             float(env.physics_ctx()[3] * 5.0)))
        env.step(target)
        if env.dropped:
            drop_skill = ctl.skill
            break
        if info["task_done"]:
            break
    return (env.success(), env.dropped, transport_steps, lift_steps,
            drop_skill, per_step)


EVAL_HARD_LO, EVAL_HARD_HI = 1.16, 1.26   # m/μ discriminating band at τ=0.05 (Phase 1 calibration: blind drops 3/3 ≥1.158; aware holds to 1.26)


def cells(n, rng, hard_frac=0.5, band_lo=EVAL_HARD_LO, band_hi=EVAL_HARD_HI):
    """n feasible (mu, mass) cells. hard_frac of them drawn from the m/μ ∈
    [band_lo, band_hi] extreme tail — the band where the blind baseline's
    ~54mm steps genuinely slip (risk ≈ m/μ > 1 sustained long enough to
    accumulate the drop threshold) and the aware expert's step contraction
    must kick in hardest. mu is capped so m = (m/μ)·mu stays in the feasible
    mass range."""
    out = []
    while len(out) < n:
        if rng.random() < hard_frac:
            mu = rng.uniform(0.2, min(0.34, 0.35 / band_hi))
            m = rng.uniform(band_lo, band_hi) * mu
        else:
            mu, m = sample_physics(rng)
        if 0.05 <= m <= 0.35 and m * G0 / mu <= GRASPABLE_FRAC * F_MAX:
            out.append((mu, m))
    return out


def margin(mu, m):
    """Excess grip capacity over the gravity load, N."""
    return mu * F_MAX - m * G0


def report(groups, n_cells, eps):
    """groups: list of dicts with ok, dropped, transport, lift, bins, lift_bins,
    cells, ok_by_cell."""
    print("── gate (h) implicit action planning (Coulomb physics, "
          f"{n_cells} cells × {eps} eps) ──")
    blind, aware = groups[0], groups[1]
    for g in groups:
        ok = sum(g["ok"]); tot = len(g["ok"])
        tr = np.concatenate(g["transport"]) if g["transport"] else np.array([np.nan])
        print(f"\n(h1) {g['name']:24s} success {ok}/{tot} ({ok/tot*100:.0f}%)  "
              f"dropped {sum(g['dropped'])}  "
              f"transport step {np.nanmean(tr):.1f} mm")
        if g["bins"]:
            print("    (h2) margin→transport-step:")
            for lo, hi, n, ms in g["bins"]:
                print(f"      margin {lo:5.1f}–{hi:5.1f} N:  mean step {ms:6.2f} mm  (n={n})")
        if g["lift_bins"]:
            print("    (h2) margin→lift-step:")
            for lo, hi, n, ms in g["lift_bins"]:
                print(f"      margin {lo:5.1f}–{hi:5.1f} N:  mean step {ms:6.2f} mm  (n={n})")
    print("\n  (h1) per-cell success (margin = μ·F_max − m·g0):")
    print(f"    {'mu':>5} {'m':>5} {'margin':>7} | {'blind':>6} {'aware':>6}")
    for c in blind["cells"]:
        bc = blind["ok_by_cell"][blind["cells"].index(c)]
        ac = aware["ok_by_cell"][aware["cells"].index(c)]
        print(f"    {c[0]:5.2f} {c[1]:5.2f} {margin(*c):7.2f} | "
              f"{sum(bc):>3d}/{eps:<2} {sum(ac):>3d}/{eps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=40)
    ap.add_argument("--eps", type=int, default=2, help="episodes per cell")
    ap.add_argument("--hard-frac", type=float, default=0.5,
                    help="fraction of cells from the m/μ>1 hard tail")
    ap.add_argument("--no-morph", action="store_true", help="skip (h3) ablation")
    ap.add_argument("--zero-ctx", action="store_true",
                    help="run aware experts with the modality channel zeroed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    aware = load_experts(args.aware_dir, args.blind_dir, device)
    blind = load_experts(None, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    cs = cells(args.n_cells, rng, hard_frac=args.hard_frac)

    def collect(experts, ablate=False, zctx=False):
        ok_by_cell = []
        dropped_all, transport_all, lift_all, ok_all, margins_ep = [], [], [], [], []
        for mu, m in cs:
            okk, dro, tr, lf, _, _ = [list(x) for x in zip(*[
                run_episode(experts, router, mu, m, rng, device,
                            morph_ablate=ablate, zero_ctx=zctx)
                for _ in range(args.eps)])]
            ok_by_cell.append([int(o) for o in okk])
            ok_all += [int(o) for o in okk]
            dropped_all += [int(d) for d in dro]
            transport_all += [np.array(t) for t in tr]
            lift_all += [np.array(t) for t in lf]
            margins_ep += [margin(mu, m)] * args.eps
        return ok_all, dropped_all, transport_all, lift_all, ok_by_cell, margins_ep

    def bin_steps(step_all, ok_all, margins_ep):
        """per-skill mean-step vs margin, binned by margin quintiles."""
        pairs = [(margins_ep[ci], np.mean(st))
                 for ci, (st, ok) in enumerate(zip(step_all, ok_all))
                 if ok and len(st)]
        if len(pairs) < 5:
            return []
        lo = np.percentile([p[0] for p in pairs], [0, 20, 40, 60, 80, 100])
        out = []
        for i in range(5):
            sel = [p[1] for p in pairs
                   if lo[i] <= p[0] < lo[i + 1] or (i == 4 and p[0] >= lo[i])]
            if sel:
                out.append((round(lo[i], 1), round(lo[i + 1], 1), len(sel),
                            round(float(np.mean(sel)), 2)))
        return out

    groups = []
    runs = [("blind (no physics)", blind, False, False),
            ("aware (physics ctx)", aware, False, False)]
    if args.zero_ctx:
        runs.append(("aware (ctx zeroed)", aware, False, True))
    for name, experts, ablate, zctx in runs:
        ok_all, dropped_all, transport_all, lift_all, ok_by_cell, margins_ep = collect(
            experts, ablate, zctx)
        groups.append({"name": name, "ok": ok_all, "dropped": dropped_all,
                       "transport": transport_all, "lift": lift_all,
                       "bins": bin_steps(transport_all, ok_all, margins_ep),
                       "lift_bins": bin_steps(lift_all, ok_all, margins_ep),
                       "cells": cs, "ok_by_cell": ok_by_cell})
    report(groups, args.n_cells, args.eps)

    if not args.no_morph:
        print("\n  (h3) morphogen warm-start ablation (aware experts):")
        ok_abl = []
        for mu, m in cs:
            for _ in range(args.eps):
                o, _, _, _, _, _ = run_episode(aware, router, mu, m, rng, device,
                                               morph_ablate=True)
                ok_abl.append(int(o))
        ok_ws = groups[1]["ok"]
        print(f"    aware (warm-start): {sum(ok_ws)}/{len(ok_ws)}  "
              f"vs  aware (no warm-start): {sum(ok_abl)}/{len(ok_abl)}")


if __name__ == "__main__":
    main()
