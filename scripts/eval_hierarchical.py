#!/usr/bin/env python3
"""Decision B — hierarchical closed-loop eval: SigLIP planner + aware-NCA actuator.

The aware NCA actuator (experts + MoE router) is the low-level executor that
already holds MuJoCo long-transport 0/15. A tiny SigLIP planner head regresses
the scene+physics plan [obj_xy, place_xy, mu, mass] from a top-down image and is
injected EXACTLY where the NCA baselines receive oracle inputs:

  scene       → ctl.reset([obĵ_xy, placê_xy])            (router skill + goal xy)
  physics_ctx → 9-dim ctx rebuilt from (μ̂, m̂)            (aware step contraction)

The env still adjudicates slip/drop/success with TRUE physics + TRUE place, so a
planner that UNDER-estimates load (μ̂/m̂ → m/μ̂ too low) makes the actuator
under-contract → real drops. This is the honest no-oracle test: does the
hierarchical system reproduce the oracle-aware outcome (hard sim 36/36, MuJoCo
long-transport 0/15) where the end-to-end VLA (decision A) failed (92% sim but
14/14 long-transport drops)?

An ORACLE control row (classic run_episode_hard on the SAME cells) is run
in-script to confirm harness parity before reading the planner row.

Run from the repo root:
  python scripts/eval_hierarchical.py --planner-ckpt ckpts/planner/best.pt \
      --n-cells 12 --eps 3 --long-transport 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from sim_env import SimEnv
import proc_sim
from soma.physics import F_MAX, GRASPED, numpy_slip_metrics
from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from eval_baseline_hard import (HAVE_MJ, perceived_scene, run_episode_hard,
                                long_transport, DEFAULT_NOISE_MM,
                                DEFAULT_DISTURB_MM)
from eval_implicit_planning import cells, margin, load_experts
from planner_data import planner_render
from vla_data import siglip_normalize
from train_planner import SigLIPPlanner
from train_vla_tiny import count_params


ANCHOR_AWARE = "aware-oracle sim 36/36 (100%), long-transport 0/15"
ANCHOR_VLA_A = "VLA-A sim 33/36 (92%), long-transport 14/14 (cycle 10)"


def planner_forward(planner, img_arr, norm, device):
    """Render→normalize→decode→denormalize → [obj_x,obj_y,place_x,place_y,mu,mass]."""
    with torch.no_grad():
        y = planner(torch.from_numpy(img_arr[None]).to(device))[0].cpu().numpy()
    mean = np.asarray(norm["mean"].cpu())
    std = np.asarray(norm["std"].cpu())
    return (y * std + mean).astype(np.float32)


def _ctx_from(mu_hat, mass_hat, s, mode, risk):
    """9-dim physics_ctx from PREDICTED physics (mirror sim_env._make_ctx)."""
    F_n = F_MAX * (1.0 - float(s[6]))
    onehot = np.zeros(5, dtype=np.float32)
    onehot[int(mode)] = 1.0
    return np.concatenate([
        [mu_hat / 0.6, mass_hat / 0.35, F_n / F_MAX, min(float(risk), 5.0) / 5.0],
        onehot]).astype(np.float32)


def run_episode_hier(exps, router, planner, norm, mu, mass, rng, device,
                     sigma_m, disturb_m):
    """One hierarchical episode: planner reads scene+physics from pixels, aware
    NCA actuator executes; env/MuJoCo judge with TRUE physics/place.
    Returns (success, dropped, drop_skill, transport_steps, lift_steps, seg,
             est[6], scene_see[4], obj[2], place[2])."""
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene_true = np.concatenate([obj, place]).astype(np.float32)
    scene_see = perceived_scene(scene_true, rng, sigma_m)
    # Image the planner sees: positions drawn at the (noisy) coords the
    # controller "sees"; size/shade encode the TRUE mu/mass (what the camera
    # captures). Planner regresses all six back out of pixels.
    img = planner_render(scene_see[:2], scene_see[2:], mu, mass)
    arr = siglip_normalize(np.asarray(img, dtype=np.float32) / 255.0)
    est = planner_forward(planner, arr, norm, device)
    obj_hat, place_hat, mu_hat, mass_hat = est[:2], est[2:4], est[4], est[5]
    ctl = BodyGraphController(exps, router, device=device)
    ctl.reset(np.concatenate([obj_hat, place_hat]).astype(np.float32),
              env.state.copy())
    # Seed the controller's ctx from the predicted physics (risk 0, no prior
    # command) — replaces the reset ctx the env built from true physics.
    env.ctx = _ctx_from(mu_hat, mass_hat, env.state, env.contact_mode, 0.0)

    transport_steps, lift_steps, tr_seg = [], [], []
    drop_skill, disturbed = None, False
    for _ in range(800):
        if ctl.skill == "transport" and not disturbed:
            env.perturb(disturb_m)
            disturbed = True
        target, info = ctl.step(env.state, physics_ctx=env.physics_ctx())
        if ctl.skill == "transport":
            transport_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
            tr_seg.append((env.state.copy(), target.copy()))
        elif ctl.skill == "lift":
            lift_steps.append(np.linalg.norm(target[:3] - env.state[:3]) * 1000.0)
        prev = env.state.copy()
        env.step(target)
        # env.step set env.ctx from TRUE physics; rebuild it from the predicted
        # physics + the risk of the just-applied command (same one-step-lag and
        # post-step held semantics env.step uses).
        held = (env.contact_mode == GRASPED)
        _, _, risk = numpy_slip_metrics(prev, target, mu_hat, mass_hat,
                                        held=held)
        env.ctx = _ctx_from(mu_hat, mass_hat, env.state, env.contact_mode, risk)
        if env.dropped and drop_skill is None:
            drop_skill = ctl.skill
        if info["task_done"]:
            break
    seg = None
    if tr_seg:
        seg = {"states": np.array([s for s, _ in tr_seg], dtype=np.float32),
               "targets": np.array([t for _, t in tr_seg], dtype=np.float32)}
    return (env.success(), env.dropped, drop_skill, transport_steps, lift_steps,
            seg, est, scene_see, obj, place)


def collect_oracle(exps, router, cs, eps, rng, device, sigma_m, disturb_m):
    """(mu,m,seg,ok,...) per episode via the classic oracle-ctx loop (control)."""
    out = []
    for mu, m in cs:
        for _ in range(eps):
            ok, dropped, dsk, ts, ls, seg = run_episode_hard(
                exps, router, mu, m, rng, device, sigma_m, disturb_m,
                record=True)
            out.append({"ok": ok, "dropped": dropped, "skill": dsk, "seg": seg,
                        "tr": (float(np.mean(ts)) if ts else float("nan")),
                        "mu": float(mu), "m": float(m)})
    return out


def collect_hier(exps, router, planner, norm, cs, eps, rng, device, sigma_m,
                 disturb_m):
    out = []
    for mu, m in cs:
        for _ in range(eps):
            ok, dropped, dsk, ts, ls, seg, est, see, obj, place = \
                run_episode_hier(exps, router, planner, norm, mu, m, rng,
                                 device, sigma_m, disturb_m)
            out.append({"ok": ok, "dropped": dropped, "skill": dsk, "seg": seg,
                        "tr": (float(np.mean(ts)) if ts else float("nan")),
                        "mu": float(mu), "m": float(m), "est": est,
                        "see": see, "obj": obj, "place": place})
    return out


def lt_segs(rows):
    """long_transport's input: (mu, m, seg) per episode, band-filtered inside."""
    return [(r["mu"], r["m"], r["seg"]) for r in rows]


def report(name, rows, ltr, show_regress=False):
    tot = len(rows)
    ok = sum(r["ok"] for r in rows)
    dropped = sum(r["dropped"] for r in rows)
    skills = {}
    for r in rows:
        if r["dropped"] and r["skill"]:
            skills[r["skill"]] = skills.get(r["skill"], 0) + 1
    sk = ", ".join(f"{k} {v}" for k, v in sorted(skills.items())) or "—"
    nseg = sum(1 for r in rows if r["seg"] is not None)
    print(f"\n── {name} (noise σ=3.7mm, push ±10mm) ──")
    print(f"  sim ok {ok}/{tot} ({ok/tot*100:.0f}%)  drops {dropped} "
          f"(skill: {sk})  timeouts {tot-ok-dropped}")
    if show_regress:
        d = []
        for r in rows:
            est = r["est"]
            rr = est[5] / est[4]
            rt = r["m"] / r["mu"]
            d.append((np.abs(rr - rt) / rt, rr < rt))
        rel = np.mean([x[0] for x in d]) * 100
        under = np.mean([x[1] for x in d]) * 100
        print(f"  planner regression: m/μ rel err {rel:.1f}% (mean), "
              f"under-estimate {under:.0f}% of episodes")
    if ltr is not None:
        n_band = sum(1 for r in ltr if r is not None)
        dropped_lt = sum(1 for r in ltr if r is not None and r[0] is not None)
        ks = [r[0] for r in ltr if r is not None and r[0] is not None]
        slips = [r[2] for r in ltr if r is not None]
        med = np.median(slips) if slips else float("nan")
        print(f"  MuJoCo long-transport ×60: {dropped_lt}/{n_band} dropped  "
              f"median cycle-to-drop {int(np.median(ks)) if ks else 'none'}  "
              f"median max slip @Kmax {med:6.2f} mm  "
              f"({nseg} transport segments)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-ckpt", type=str, default="ckpts/planner/best.pt")
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys_honest")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--percept-noise", type=float, default=DEFAULT_NOISE_MM)
    ap.add_argument("--disturb", type=float, default=DEFAULT_DISTURB_MM)
    ap.add_argument("--long-transport", type=int, default=60)
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    exps = load_experts(args.aware_dir, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])
    nca_train = sum(p.numel() for p in router.parameters()) + sum(
        sum(p.numel() for p in e.parameters()) for e in exps.values())
    print(f"NCA actuator: {nca_train/1e3:.0f}K params (router + 6 experts)")

    ck = torch.load(args.planner_ckpt, map_location=device, weights_only=True)
    planner = SigLIPPlanner().to(device).eval()
    planner.head.load_state_dict(ck["head"])
    norm = ck["norm"]
    _, head_tr = count_params(planner)
    print(f"SigLIP planner: frozen SigLIP-B16 + {head_tr/1e3:.0f}K trainable head")

    if not HAVE_MJ:
        print("warning: mujoco unavailable; sim-only")
    use_mj = HAVE_MJ and not args.no_mujoco

    sigma_m, disturb_m = args.percept_noise / 1000.0, args.disturb / 1000.0
    cs = cells(args.n_cells, rng, hard_frac=args.hard_frac)

    oracle = collect_oracle(exps, router, cs, args.eps, rng, device, sigma_m,
                            disturb_m)
    hier = collect_hier(exps, router, planner, norm, cs, args.eps, rng, device,
                        sigma_m, disturb_m)

    ltr_o = (long_transport(lt_segs(oracle), args.long_transport)
             if use_mj and args.long_transport > 0 else None)
    ltr_h = (long_transport(lt_segs(hier), args.long_transport)
             if use_mj and args.long_transport > 0 else None)

    report("Oracle-aware (control: true physics ctx)", oracle, ltr_o)
    report("Decision-B hier (planner scene+physics from pixels)", hier, ltr_h,
           show_regress=True)

    # Per-cell: oracle sim vs hier sim, and the planner's est error vs TRUE
    # scene/physics (the perception+regression the system carries).
    print("\n  per-cell (oracle sim / hier sim, planner est error vs TRUE):")
    for ci, (mu, m) in enumerate(cs):
        sl = slice(ci * args.eps, (ci + 1) * args.eps)
        os_ = sum(o["ok"] for o in oracle[sl])
        hs = sum(h["ok"] for h in hier[sl])
        eo, ep, pm, pd = [], [], [], []
        for h in hier[sl]:
            est = h["est"]
            eo.append(np.linalg.norm(est[:2] - h["obj"]) * 1000.0)
            ep.append(np.linalg.norm(est[2:4] - h["place"]) * 1000.0)
            pm.append(abs(est[4] - mu))
            pd.append(abs(est[5] - m))
        print(f"    mu {mu:.2f} m {m:.2f} m/μ {m/mu:.2f} "
              f"margin {margin(mu, m):.1f} | oracle {os_}/{args.eps}  "
              f"hier {hs}/{args.eps} | obĵ {np.mean(eo):.1f}mm "
              f"placê {np.mean(ep):.1f}mm  μ̂ err {np.mean(pm):.3f} "
              f"m̂ err {np.mean(pd):.3f}")
    print(f"\n  anchors: {ANCHOR_AWARE}; {ANCHOR_VLA_A}.")


if __name__ == "__main__":
    main()
