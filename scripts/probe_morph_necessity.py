#!/usr/bin/env python3
"""2x2 necessity probe: does the load the net stores in morphogen get USED?

The capacity probe (probe_morphogen.py) shows the aware expert already writes
m/μ into its morphogen (per-skill R^2 ~0.85). The open question is whether that
stored copy is ever *used* — h3 was tested with ctx ON (ablation NULL because
ctx supplies the load), and ctx-off was tested with morphogen ON. The decisive
cell is ctx-OFF x morphogen-ABLATED: if the net reads load from memory when ctx
is gone, cutting the memory must hurt.

Conditions (aware experts, hard distribution):
  ctx ON , morph ON   -> baseline (ctx supplies load; memory redundant)
  ctx ON , morph OFF  -> h3 (known NULL)
  ctx OFF, morph ON   -> memory could carry load
  ctx OFF, morph OFF  -> NO load source at all -> must fall back to a fixed
                         (conservative) policy; if the task doesn't need
                         adaptivity, this still succeeds -> no pressure.

Run from the repo root:
  python scripts/probe_morph_necessity.py --n-cells 16 --eps 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.moe_router import StateRouter
from eval_implicit_planning import load_experts, cells, run_episode, margin


def collect(experts, router, cs, eps, rng, device, ctx, morph):
    ok, dropped, tr_steps, per_cell = 0, 0, [], []
    for mu, m in cs:
        c_ok = 0
        for _ in range(eps):
            o, dro, ts, ls, dsk, _ = run_episode(
                experts, router, mu, m, rng, device,
                morph_ablate=(not morph), zero_ctx=(not ctx))
            ok += int(o)
            dropped += int(dro)
            c_ok += int(o)
            if ts:
                tr_steps.append(float(np.mean(ts)))
        per_cell.append(c_ok)
    n = len(cs) * eps
    return {"ok": ok, "n": n, "dropped": dropped,
            "tr": float(np.mean(tr_steps)) if tr_steps else float("nan"),
            "per_cell": per_cell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys_honest")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    aware = load_experts(args.aware_dir, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])
    cs = cells(args.n_cells, rng, hard_frac=args.hard_frac)
    hard = [(mu, m) for mu, m in cs if margin(mu, m) < 1.5]
    print(f"{len(cs)} cells × {args.eps} eps  (hard cells margin<1.5N: {len(hard)})")
    print(f"aware experts: {args.aware_dir}\n")

    print(f"  {'config':>22} | {'success':>9} | {'dropped':>7} | {'transport mm':>12}")
    rows = {}
    for ctx, morph, tag in [(True, True, "ctx ON  morph ON"),
                            (True, False, "ctx ON  morph OFF (h3)"),
                            (False, True, "ctx OFF morph ON"),
                            (False, False, "ctx OFF morph OFF")]:
        r = collect(aware, router, cs, args.eps, rng, device, ctx, morph)
        rows[tag] = r
        print(f"  {tag:>22} | {r['ok']:>4}/{r['n']:<4} | {r['dropped']:>7} | {r['tr']:>12.1f}")

    base = rows["ctx ON  morph ON"]["ok"]
    print(f"\n  memory-necessity (vs ctx ON morph ON = {base}/{rows['ctx ON  morph ON']['n']}):")
    for tag in ["ctx ON  morph OFF (h3)", "ctx OFF morph ON", "ctx OFF morph OFF"]:
        d = base - rows[tag]["ok"]
        print(f"    {tag:>22}: Δsuccess {d:+d}")
    print("\nreading: if 'ctx OFF morph OFF' is the WORST and clearly below")
    print("'ctx OFF morph ON', the stored morphogen load is load-bearing once ctx")
    print("is gone. If they tie, the task needs no adaptivity (no pressure).")


if __name__ == "__main__":
    main()
