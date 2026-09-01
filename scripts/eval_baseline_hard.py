#!/usr/bin/env python3
"""Baseline eval harness — the hard distribution + long-transport distribution.

This is the evaluation distribution the SmolVLA-0.5B baseline will later be
measured against. The two NCA baselines are scored here first:

  blind  (checkpoints_rollin2)      — no Coulomb modality, no slip-loss training
  aware  (checkpoints_phys_honest)  — Coulomb modality + slip-loss training

Hard distribution (per episode):
  * cells from the m/μ discriminating band (hard-frac, EVAL [1.16, 1.26])
  * one Gaussian draw σ on the scene the CONTROLLER sees (routing + expert
    goals); the env keeps the TRUE obj/place for physics and success — models a
    VLA whose perception path regresses scene coords with ~SigLIP-level error.
  * one ±D mm EEF push between the 1st and 2nd transport commands (mid-carry
    disturbance); the transport expert re-plans from the displaced state.

Adjudication:
  * sim: closed-loop success (obj released within place_tol of TRUE place)
  * MuJoCo: worst-case-yaw replay of each recorded transport segment via the
    pseudo-force FixedPadsReplay (independent rigid-body verdict)

Long-transport distribution (--long-transport Kmax): replay each discriminating
segment continuously Kmax× so MuJoCo's rigid-body slip can accumulate past the
15 mm drop threshold — answers whether the sim slip band is physically real
given a longer transport, and gives a MuJoCo-reproducible outcome claim
(blind slips to a drop, aware's contracted steps do not).

Run from the repo root:
  python scripts/eval_baseline_hard.py --n-cells 12 --eps 3
  python scripts/eval_baseline_hard.py --n-cells 8 --eps 3 --long-transport 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from soma.physics import F_MAX
from sim_env import SimEnv
import proc_sim
from eval_implicit_planning import (EVAL_HARD_LO, EVAL_HARD_HI, cells, margin,
                                    load_experts)

try:
    import mujoco
    from eval_level_a import FixedPadsReplay, YAW90, TAU_STEPS, DROP_SLIP
    HAVE_MJ = True
    MJ_ERR = None
except Exception as e:          # noqa: E722
    HAVE_MJ = False
    MJ_ERR = e

DEFAULT_NOISE_MM = 3.7      # SigLIP-level scene-perception error (mm)
DEFAULT_DISTURB_MM = 10.0   # mid-carry EEF push (mm)


def perceived_scene(scene_true, rng, sigma_m):
    """The controller sees the true scene corrupted by one Gaussian draw."""
    return (np.asarray(scene_true, np.float32)
            + rng.normal(0.0, sigma_m, 4).astype(np.float32))


def run_episode_hard(experts, router, mu, mass, rng, device, sigma_m,
                     disturb_m, record=False):
    """One closed-loop episode on the hard distribution.

    Returns (success, dropped, drop_skill, transport_steps, lift_steps,
             transport_segment or None)."""
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene_true = np.concatenate([obj, place]).astype(np.float32)
    scene_see = perceived_scene(scene_true, rng, sigma_m)
    ctl = BodyGraphController(experts, router, device=device)
    ctl.reset(scene_see, env.state.copy())
    transport_steps, lift_steps, tr_seg = [], [], []
    drop_skill, disturbed = None, False
    for _ in range(800):
        # Disturbance lands before the 2nd transport command: the transport
        # expert must re-plan from the displaced EEF (mid-carry push).
        if ctl.skill == "transport" and not disturbed:
            env.perturb(disturb_m)
            disturbed = True
        target, info = ctl.step(env.state, physics_ctx=env.physics_ctx())
        if ctl.skill == "transport":
            transport_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
            tr_seg.append((env.state.copy(), target.copy()))
        elif ctl.skill == "lift":
            lift_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
        env.step(target)
        if env.dropped and drop_skill is None:
            drop_skill = ctl.skill
            # do NOT break: keep recording the full intended transport so the
            # MuJoCo replay adjudicates the complete commanded motion.
        if info["task_done"]:
            break
    seg = None
    if tr_seg:
        seg = {"states": np.array([s for s, _ in tr_seg], dtype=np.float32),
               "targets": np.array([t for _, t in tr_seg], dtype=np.float32)}
    return (env.success(), env.dropped, drop_skill, transport_steps,
            lift_steps, seg)


def collect_agent(experts, router, cs, rng, device, sigma_m, disturb_m, eps):
    """Per-agent: (ok_by_cell[cell][ep], sim_ok[], drop_skill[], segs[])."""
    ok_by_cell, sim_ok, dskills, segs = [], [], [], []
    for mu, m in cs:
        okc = []
        for _ in range(eps):
            ok, dropped, dskill, _, _, seg = run_episode_hard(
                experts, router, mu, m, rng, device, sigma_m, disturb_m,
                record=True)
            okc.append(int(ok)); sim_ok.append(int(ok))
            dskills.append(dskill); segs.append((mu, m, seg))
        ok_by_cell.append(okc)
    return ok_by_cell, sim_ok, dskills, segs


# ── MuJoCo adjudication ──────────────────────────────────────────────
def adjudicate(seg, mu, mass):
    """Worst-case-yaw replay of one transport segment. (held, slip_mm, minz_mm)."""
    Fn = F_MAX * (1.0 - float(np.mean(seg["states"][:, 6])))
    rep = FixedPadsReplay(mu, mass, Fn)
    st = seg["states"][:, :3] @ YAW90.T
    tg = seg["targets"][:, :3] @ YAW90.T
    return rep.replay(st, tg)


def replay_cycles_drop(rep, states, targets, Kmax):
    """Continuous replay of the yawed segment for Kmax cycles. Returns
    (cycle_to_drop or None, steps_to_drop or None, max_slip_mm @ Kmax)."""
    drop = rep._settle()
    if drop > DROP_SLIP:
        return 0, 0, drop * 1000.0
    st = states[:, :3] @ YAW90.T
    tg = targets[:, :3] @ YAW90.T
    n = len(st)
    max_slip, steps = 0.0, 0
    for k in range(Kmax):
        for i in range(n):
            D = tg[i] - st[i]
            for j in range(TAU_STEPS):
                rep._apply_accel(D, j / TAU_STEPS)
                mujoco.mj_step(rep.model, rep.data)
                rel = rep.data.xpos[rep.obj_bid] - rep.ref
                if float(np.linalg.norm(rel)) > max_slip:
                    max_slip = float(np.linalg.norm(rel))
                steps += 1
            if max_slip > DROP_SLIP:
                return k + 1, steps, max_slip * 1000.0
    return None, steps, max_slip * 1000.0


def long_transport(segs, Kmax):
    """Per band segment: (cycle_to_drop, steps_to_drop, slip@Kmax mm) or None."""
    out = []
    for mu, m, seg in segs:
        if seg is None or m / mu < EVAL_HARD_LO:
            out.append(None)
            continue
        Fn = F_MAX * (1.0 - float(np.mean(seg["states"][:, 6])))
        rep = FixedPadsReplay(mu, m, Fn)
        out.append(replay_cycles_drop(rep, seg["states"], seg["targets"], Kmax))
    return out


# ── reporting ─────────────────────────────────────────────────────────
def _mj_cell(segs, mj, ci, eps):
    """'held/seg' of the replayed transport segments of cell ci."""
    start = ci * eps
    held = seg = 0
    for i in range(start, start + eps):
        if segs[i][2] is not None:
            seg += 1
            held += int(mj[i][0])
    return f"{held}/{seg}" if seg else "  n/a"


def report(cs, eps, sigma_mm, disturb_mm, groups, mj, ltr, Kmax):
    print("── Hard-distribution baseline (NCA) ──")
    print(f"  cells {len(cs)} × {eps} eps/agent, hard-frac 0.5, "
          f"noise σ={sigma_mm:.1f} mm, push ±{disturb_mm:.0f} mm")
    print("  sim success = obj within place_tol of TRUE place; "
          "MuJoCo = worst-case-yaw transport replay\n")
    print(f"  {'agent':<26} {'sim ok':>8} {'sim drops':>10} {'mj tseg':>9} "
          f"{'mj held':>9} {'mj slip':>8}")
    for name, sim_ok, dskills, _, segs in groups:
        mj_n = mj.get(name) if mj else None
        tot = len(sim_ok)
        drops = sum(1 for d in dskills if d is not None)
        idx = [i for i in range(len(segs)) if segs[i][2] is not None]
        held = sum(1 for i in idx if mj_n is not None and mj_n[i][0])
        slip = (np.mean([mj_n[i][1] for i in idx]) if (mj_n is not None and idx)
                else float("nan"))
        print(f"  {name:<26} {sum(sim_ok):>4d}/{tot:<3} {drops:>10} "
              f"{len(idx):>9} {held:>4d}/{len(idx):<3} {slip:>7.2f} mm")
    print("\n  per-cell (sim ok / MuJoCo held of replayed transport segments):")
    print(f"    {'mu':>5} {'m':>5} {'m/μ':>5} {'margin':>7} | "
          f"{'blind ok':>9} {'aware ok':>9} | {'blind mj':>10} {'aware mj':>10}")
    b_segs, a_segs = groups[0][4], groups[1][4]
    mj_nb = mj.get(groups[0][0]) if mj else None
    mj_na = mj.get(groups[1][0]) if mj else None
    for ci, (mu, m) in enumerate(cs):
        b_ok, a_ok = groups[0][3][ci], groups[1][3][ci]
        b_mj = _mj_cell(b_segs, mj_nb, ci, eps) if mj_nb is not None else "  n/a"
        a_mj = _mj_cell(a_segs, mj_na, ci, eps) if mj_na is not None else "  n/a"
        print(f"    {mu:5.2f} {m:5.2f} {m / mu:5.2f} {margin(mu, m):7.2f} | "
              f"{sum(b_ok):>4d}/{eps:<4} {sum(a_ok):>4d}/{eps:<4} | "
              f"{b_mj:>10} {a_mj:>10}")
    if mj is not None:
        for name, _, dskills, _, segs in groups:
            mj_n = mj[name]
            tdrop = [i for i in range(len(segs))
                     if dskills[i] == "transport" and segs[i][2] is not None]
            if tdrop:
                held = sum(1 for i in tdrop if mj_n[i][0])
                print(f"\n  {name}: {len(tdrop)} transport segments sim-judged "
                      f"dropped; MuJoCo reproduces {len(tdrop) - held} of them "
                      f"({held} kept HELD)")
    if ltr is not None:
        print(f"\n── Long-transport distribution (continuous replay ×{Kmax}) ──")
        for name, _, _, _, segs in groups:
            res = ltr[name]
            n_band = sum(1 for r in res if r is not None)
            dropped = sum(1 for r in res if r is not None and r[0] is not None)
            ks = [r[0] for r in res if r is not None and r[0] is not None]
            slips = [r[2] for r in res if r is not None]
            med_slip = np.median(slips) if slips else float("nan")
            kstr = (f"median cycle-to-drop {int(np.median(ks))}" if ks
                    else "no cycle dropped")
            print(f"  {name:<26} {dropped:>3d}/{n_band:<3} dropped by Kmax  "
                  f"{kstr:>24}  median max slip @Kmax {med_slip:6.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys_honest")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--band-lo", type=float, default=EVAL_HARD_LO,
                    help="hard-tail m/μ lower bound")
    ap.add_argument("--band-hi", type=float, default=EVAL_HARD_HI,
                    help="hard-tail m/μ upper bound")
    ap.add_argument("--percept-noise", type=float, default=DEFAULT_NOISE_MM,
                    help="perception noise σ, mm")
    ap.add_argument("--disturb", type=float, default=DEFAULT_DISTURB_MM,
                    help="mid-carry EEF push, mm")
    ap.add_argument("--long-transport", type=int, default=0,
                    help="Kmax cycle replay (0 = off)")
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    if not HAVE_MJ:
        print(f"warning: mujoco unavailable ({MJ_ERR}); sim-only")
    use_mj = HAVE_MJ and not args.no_mujoco

    aware = load_experts(args.aware_dir, args.blind_dir, device)
    blind = load_experts(None, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    cs = cells(args.n_cells, rng, hard_frac=args.hard_frac,
               band_lo=args.band_lo, band_hi=args.band_hi)
    sigma_m, disturb_m = args.percept_noise / 1000.0, args.disturb / 1000.0

    groups = []
    for name, experts in (("blind (no physics)", blind),
                          ("aware (physics ctx)", aware)):
        ok_by_cell, sim_ok, dskills, segs = collect_agent(
            experts, router, cs, rng, device, sigma_m, disturb_m, args.eps)
        groups.append((name, sim_ok, dskills, ok_by_cell, segs))

    mj, ltr = None, None
    if use_mj:
        mj = {name: [adjudicate(seg, mu, m) if seg is not None
                     else (None, None, None) for mu, m, seg in segs]
              for name, _, _, _, segs in groups}
        if args.long_transport > 0:
            ltr = {name: long_transport(segs, args.long_transport)
                   for name, _, _, _, segs in groups}
    report(cs, args.eps, args.percept_noise, args.disturb, groups, mj, ltr,
           args.long_transport or 0)


if __name__ == "__main__":
    main()
