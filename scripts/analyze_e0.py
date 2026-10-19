#!/usr/bin/env python3
"""Paired analysis of E0 closed-loop results (scene-level McNemar + bootstrap).

Usage:
  python scripts/analyze_e0.py logs/e0/*.json --main gt-plan_s0 zero-plan_s0
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def mcnemar_exact(b, c, one_sided=False):
    """Exact McNemar p for discordant counts b, c (b = A win)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    if one_sided:
        return float(sum(comb(n, i) for i in range(b, n + 1)) / 2.0 ** n)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def boot_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    bs = np.array([d[rng.randint(n, size=n)].mean() for _ in range(n_boot)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(path):
    with open(path) as f:
        j = json.load(f)
    j["name"] = os.path.basename(path).replace(".json", "")
    return j


def compare(A, B, label, one_sided=False):
    a, b = np.array(A["success"]), np.array(B["success"])
    if len(a) != len(b):
        print(f"{label}: length mismatch {len(a)} vs {len(b)}, skip")
        return None
    gain = (a.mean() - b.mean()) * 100
    lo, hi = boot_diff(a, b)
    n10 = int(((a == 1) & (b == 0)).sum())   # A success, B fail
    n01 = int(((a == 0) & (b == 1)).sum())   # A fail, B success
    p = mcnemar_exact(n10, n01, one_sided=one_sided)
    tag = "one-sided" if one_sided else "two-sided"
    print(f"{label}: {a.mean()*100:5.1f}% vs {b.mean()*100:5.1f}%  "
          f"gain {gain:+5.1f} pp  boot95 [{lo*100:+.1f},{hi*100:+.1f}]  "
          f"discordant +{n10}/-{n01}  McNemar({tag}) p={p:.4f}")
    return {"gain_pp": gain, "lo": lo * 100, "hi": hi * 100, "p": p,
            "n10": n10, "n01": n01}


def find_seeds(runs, arm):
    out = {}
    for name, r in runs.items():
        m = re.match(rf"(?:vec_)?{re.escape(arm)}_s(\d+)$", name)
        if m:
            out[int(m.group(1))] = r
    return dict(sorted(out.items()))


def pool_compare(runs, arm_a, arm_b, seed_a=0):
    """Seed-stratified contrast: per-seed McNemar + pooled exact test.

    Scenes are identical across seeds (all runs use the same eval seed), so
    the pooled bootstrap resamples scene index jointly across seeds.
    """
    a_runs, b_runs = find_seeds(runs, arm_a), find_seeds(runs, arm_b)
    seeds = sorted(set(a_runs) & set(b_runs))
    if not seeds:
        print(f"pooled {arm_a} vs {arm_b}: no shared seeds, skip")
        return
    print(f"\npooled {arm_a} vs {arm_b} (seeds {seeds})")
    n10_t = n01_t = 0
    dmat = []
    for s in seeds:
        A, B = a_runs[s], b_runs[s]
        a, b = np.array(A["success"]), np.array(B["success"])
        if len(a) != len(b):
            print(f"  s{s}: length mismatch {len(a)} vs {len(b)}, skip")
            continue
        n10 = int(((a == 1) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        p = mcnemar_exact(n10, n01, one_sided=True)
        tag = " [pre-registered]" if s == seed_a else ""
        print(f"  s{s}{tag}: {a.mean()*100:5.1f}% vs {b.mean()*100:5.1f}%  "
              f"gain {(a.mean()-b.mean())*100:+5.1f} pp  "
              f"discordant +{n10}/-{n01}  one-sided p={p:.4f}")
        n10_t += n10
        n01_t += n01
        dmat.append(a.astype(float) - b.astype(float))
    n10_t, n01_t = int(n10_t), int(n01_t)
    p = mcnemar_exact(n10_t, n01_t, one_sided=True)
    D = np.stack(dmat)
    rng = np.random.RandomState(0)
    n = D.shape[1]
    bs = np.empty(10000)
    for i in range(10000):
        idx = rng.randint(n, size=n)
        bs[i] = D[:, idx].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  pooled: discordant +{n10_t}/-{n01_t}  "
          f"gain {D.mean()*100:+5.1f} pp  boot95 "
          f"[{lo*100:+.1f},{hi*100:+.1f}]  one-sided p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--main", nargs=2, default=None,
                    help="two run names for the pre-registered contrast")
    ap.add_argument("--pooled", nargs=2, default=None, metavar=("A", "B"),
                    help="arm names (e.g. gt-plan zero-plan); pools "
                         "<A>_s0/s1 vs <B>_s0/s1 across shared seeds")
    ap.add_argument("--seed", type=int, default=0,
                    help="pre-registered seed within --pooled (default 0)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if "*" in p else [p])
    runs = {load(p)["name"]: load(p) for p in files}
    print(f"{len(runs)} runs\n")

    for name, r in sorted(runs.items()):
        s = np.array(r["success"])
        dv = [x for x in r.get("first_chunk_dev", []) if x is not None]
        dvt = f"  dev {np.median(dv):.3f}" if dv else ""
        print(f"  {name:28s} {s.mean()*100:5.1f}%  "
              f"placed {np.mean(r['placed']):.2f}  "
              f"wall {np.mean(r.get('wall_hits', [0])):.1f}  "
              f"steps {np.mean(r['steps']):.0f}  "
              f"visits {r['visits_ok']}/{r['visits_total']}{dvt}")

    if args.main:
        A, B = args.main
        print()
        compare(runs[A], runs[B], f"main {A} vs {B}", one_sided=True)

    if args.pooled:
        pool_compare(runs, *args.pooled, seed_a=args.seed)

    names = sorted(runs)
    if len(names) > 1:
        print("\nall pairwise (secondary; Holm across these):")
        results = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = compare(runs[names[i]], runs[names[j]],
                            f"{names[i]} vs {names[j]}")
                if r:
                    results.append(r)
        if results:
            m = len(results)
            prev = 0.0
            for rank, r in enumerate(sorted(results, key=lambda r: r["p"])):
                adj = min(1.0, max(prev, r["p"] * (m - rank)))
                prev = adj
                r["p_holm"] = adj
            print("\nHolm-adjusted:")
            for r in results:
                print(f"  p={r['p']:.4f} -> p_holm={r['p_holm']:.4f}")


if __name__ == "__main__":
    main()
