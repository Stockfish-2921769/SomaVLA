#!/usr/bin/env python3
"""Paired McNemar test between two eval_multi logs (per-episode success)."""
import re
import sys
from math import comb


def succ(path):
    out = {}
    for line in open(path):
        m = re.match(r"\s*ep\s*(\d+) success=(\d)", line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def main():
    base = succ(sys.argv[1])
    for path in sys.argv[2:]:
        arm = succ(path)
        eps = sorted(set(base) & set(arm))
        b10 = sum(1 for e in eps if base[e] == 0 and arm[e] == 1)
        b01 = sum(1 for e in eps if base[e] == 1 and arm[e] == 0)
        n = b10 + b01
        k = min(b10, b01)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        print(f"{path.split('/')[-1]:32s} n={len(eps)} "
              f"base {sum(base[e] for e in eps)}/{len(eps)} "
              f"arm {sum(arm[e] for e in eps)}/{len(eps)} "
              f"wins {b10} losses {b01} McNemar p={p:.2e}")


if __name__ == "__main__":
    main()
