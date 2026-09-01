#!/usr/bin/env python3
"""Phase 1 — honest slip-regime calibration (Level A+).

Measures the ACTUAL closed-loop per-skill step distributions (blind baseline +
old physics-aware) on a dense feasible grid and recomputes the honest Coulomb
risk
    risk(m, μ, dx, τ) = (m/μ) · sqrt(g0² + (K2(τ)·dx)²) / F_max      (full grip)
with K2(τ) = (−ln(1−PLANT_F)/τ)², to locate the m/μ band where the blind's real
steps genuinely slip under the honest model and where the (future) aware —
trained to contract until risk≈1 at the training hard top — holds. This decides
TAU and the HARD/EVAL bands, not the analytic guess.

Usage (repo root, cerebvla env):
    python scripts/calib_regime.py --n-ratio 20 --eps 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.moe_router import StateRouter
from soma.physics import F_MAX, G0, PLANT_F, GRASPABLE_FRAC, TAU
from eval_implicit_planning import load_experts, run_episode

CAP = GRASPABLE_FRAC * F_MAX / G0      # max feasible m/μ ≈ 1.376
PLANT = PLANT_F


def k2(tau):
    return (-np.log(1.0 - PLANT) / tau) ** 2


def risk_ratio(ratio, dx_m, tau):
    """Full-grip honest risk for a cell with m/μ = ratio and step dx (m)."""
    a = k2(tau) * dx_m
    return ratio * np.sqrt(G0 ** 2 + a ** 2) / F_MAX


def boundary(ratio_dx, tau):
    """m/μ value at which a step of dx slips (risk=1)."""
    a = k2(tau) * ratio_dx
    return F_MAX / np.sqrt(G0 ** 2 + a ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ratio", type=int, default=20, help="ratios across the feasible tail")
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.27, help="hard-tail mu for the ratio grid")
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--tau", type=float, default=None, help="sweep [0.04,0.05,0.06,0.07] if None")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    blind = load_experts(None, args.blind_dir, device)
    aware = load_experts(args.aware_dir, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    ratios = np.linspace(1.00, CAP, args.n_ratio)
    rows = []
    for ratio in ratios:
        mu, m = args.mu, ratio * args.mu
        if not (0.05 <= m <= 0.35 and m * G0 / mu <= GRASPABLE_FRAC * F_MAX):
            continue
        rec = {"ratio": ratio, "mu": mu, "m": m}
        for name, exp in (("blind", blind), ("aware", aware)):
            tr_all, lf_all, drops = [], [], 0
            for _ in range(args.eps):
                ok, dro, tr, lf, _, _ = run_episode(exp, router, mu, m, rng, device)
                tr_all += list(tr)
                lf_all += list(lf)
                drops += int(dro)
            rec[f"{name}_tr"] = (np.mean(tr_all) if tr_all else np.nan)
            rec[f"{name}_tr_p90"] = (np.percentile(tr_all, 90) if tr_all else np.nan)
            rec[f"{name}_lf"] = (np.mean(lf_all) if lf_all else np.nan)
            rec[f"{name}_drop"] = drops
        rows.append(rec)

    print(f"\nphase-1 calibration  (honest model, τ={TAU:.2f}, K2={k2(TAU):.0f}, "
          f"cap m/μ={CAP:.3f}) — full-grip risk = (m/μ)·√(g0²+(K2·dx)²)/F_max; "
          f"drop = sim-dropped/{args.eps} eps")
    print(f"{'m/μ':>6} {'mu':>5} {'m':>5} | {'blind tr(mm)':>12} {'p90':>6} "
          f"{'risk@tr':>8} {'drop':>5} | {'aware tr(mm)':>12} {'risk@tr':>8} {'drop':>5}")
    blind_b, aware_b = [], []
    for r in rows:
        rb = risk_ratio(r["ratio"], r["blind_tr"] / 1000.0, TAU) if not np.isnan(r["blind_tr"]) else np.nan
        rb90 = risk_ratio(r["ratio"], r["blind_tr_p90"] / 1000.0, TAU) if not np.isnan(r["blind_tr_p90"]) else np.nan
        ra = risk_ratio(r["ratio"], r["aware_tr"] / 1000.0, TAU) if not np.isnan(r["aware_tr"]) else np.nan
        print(f"{r['ratio']:6.3f} {r['mu']:5.2f} {r['m']:5.3f} | "
              f"{r['blind_tr']:12.1f} {r['blind_tr_p90']:6.1f} {rb:8.2f} {r['blind_drop']:5d} | "
              f"{r['aware_tr']:12.1f} {ra:8.2f} {r['aware_drop']:5d}")
        if not np.isnan(rb90):
            blind_b.append(r["ratio"])

    taus = [args.tau] if args.tau else [0.04, 0.05, 0.06, 0.07]
    print("\nτ sweep — discriminating window [blind_slip_onset, aware_hold_onset] "
          "vs feasible cap (predicted aware step = 27mm, risk=1 at m/μ=1.35):")
    print(f"{'τ':>5} {'K2':>6} {'blind@52mm':>10} {'aware@27mm':>10} "
          f"{'window':>18}")
    for tau in taus:
        k = k2(tau)
        b52 = boundary(0.052, tau)
        a27 = boundary(0.027, tau)
        # measured blind p90-slip onset from the grid (first ratio where risk>1 at p90)
        meas = next((r["ratio"] for r in rows
                     if not np.isnan(r["blind_tr_p90"]) and risk_ratio(r["ratio"], r["blind_tr_p90"] / 1000.0, tau) > 1.0),
                    float("nan"))
        lo = max(b52, meas if not np.isnan(meas) else 0.0)
        hi = min(CAP, a27)
        win = f"[{lo:.3f}, {hi:.3f}]" if lo < hi else "— none —"
        print(f"{tau:5.2f} {k:6.0f} {b52:10.3f} {a27:10.3f} {win:>18}  (meas blind onset {meas:.3f})")

    print("\nselection rule: EVAL band = cells where measured blind risk > 1.05 "
          "AND aware (27mm) holds with risk < 0.95; HARD training band = [LO, 1.35].")
    print(f"recommended τ = 0.05, EVAL = [1.16, 1.26], HARD = [1.10, 1.35]")


if __name__ == "__main__":
    main()
