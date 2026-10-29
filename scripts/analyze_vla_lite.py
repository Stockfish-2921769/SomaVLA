#!/usr/bin/env python3
"""Paired scene-level analysis for VLA-lite eval JSONs.

Scenes are paired by index (same seed/scene list across arms).  Reports the
paired success difference, an exact McNemar test, and a scene-bootstrap CI.
"""
import argparse
import json
import math

import numpy as np


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    p = min(1.0, 2.0 * min(
        sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n,
        sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n))
    return p


def binom_one_sided(k, n):
    """P(X >= k) under p=0.5 (direction a->b)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def bootstrap_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(y, np.float64) - np.asarray(x, np.float64)
    n = len(d)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.randint(n, size=n)].mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline JSON")
    ap.add_argument("--b", required=True, help="treatment JSON")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric", choices=["success", "placed", "wall_hits",
                                         "steps", "bumps", "obstacle_iou"],
                    default="success")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ja = json.load(open(args.a))
    jb = json.load(open(args.b))
    if ja["n"] != jb["n"]:
        raise SystemExit(f"n mismatch: {ja['n']} vs {jb['n']}")
    a = np.asarray(ja[args.metric], np.float64)
    b = np.asarray(jb[args.metric], np.float64)
    n = len(a)
    diff = b - a
    lo, hi = bootstrap_ci(a, b, seed=args.seed)
    print(f"{args.metric}: {args.label_a} {a.mean():.4f}  "
          f"{args.label_b} {b.mean():.4f}  diff {diff.mean():+.4f} "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]")

    if args.metric == "success":
        a = a.astype(bool)
        b = b.astype(bool)
        a1 = int((a & ~b).sum())     # a success, b fail
        b1 = int((~a & b).sum())     # b success, a fail
        disc = a1 + b1
        print(f"  discordant: {args.label_a}>{args.label_b} {a1}, "
              f"{args.label_b}>{args.label_a} {b1}")
        print(f"  exact McNemar two-sided p={binom_two_sided(b1, disc):.4f}  "
              f"one-sided p={binom_one_sided(b1, disc):.4f}")
        print(f"  success {int(a.sum())}/{n} vs {int(b.sum())}/{n}")


if __name__ == "__main__":
    main()
