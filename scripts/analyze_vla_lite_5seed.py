#!/usr/bin/env python3
"""Multi-seed paired analysis for VLA-lite eval JSONs.

For each requested arm pair, reports per-seed success differences with exact
McNemar tests, the seed-level mean/se, and a pooled scene-bootstrap CI.  Also
optionally writes a compact JSON summary (success bitstrings + statistics).

Eval files are resolved as ``<logs>/<prefix>_<arm>_s<seed>_pred.json`` where the
prefix is taken from --prefix-map (default matches the 2026-11-09 vis control:
vis was evaluated as ``conf_vis_*``; baselines were re-evaluated as
``recheck_*`` with the same code).
"""
import argparse
import json
import math
import random
from pathlib import Path


def load(logs, prefix, arm, seed):
    path = Path(logs) / f"{prefix}_{arm}_s{seed}_pred.json"
    d = json.load(open(path))
    return [bool(x) for x in d["success"]], d, path


def mcnemar(a, b):
    """Exact two-sided McNemar: returns (a_only, b_only, p)."""
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return x, y, min(p, 1.0)


def pooled_ci(A, B, iters=20000, seed=0):
    """Seed-stratified bootstrap over paired scenes; CI of mean A-B in pp."""
    rng = random.Random(seed)
    diffs = [[float(u) - float(v) for u, v in zip(a, b)] for a, b in zip(A, B)]
    n_per = len(diffs[0])
    means = []
    for _ in range(iters):
        acc = 0.0
        for ds in diffs:
            for _ in range(n_per):
                acc += ds[rng.randrange(n_per)]
        means.append(100.0 * acc / (len(diffs) * n_per))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/vla_lite")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prefix-map", nargs="+",
                    default=["vis=conf", "none=recheck", "cnn=recheck",
                             "nca=recheck"])
    ap.add_argument("--pairs", nargs="+",
                    default=["vis:none", "nca:vis", "cnn:vis", "nca:none",
                             "cnn:none", "nca:cnn"])
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    prefix = dict(p.split("=", 1) for p in args.prefix_map)
    arms = sorted({a for pair in args.pairs for a in pair.split(":")})

    data, meta = {}, {}
    for arm in arms:
        data[arm] = []
        for s in args.seeds:
            succ, d, path = load(args.logs, prefix[arm], arm, s)
            data[arm].append(succ)
            meta.setdefault(arm, []).append(
                dict(seed=s, file=str(path), count=sum(succ),
                     n=len(succ), k_steps=d.get("k_steps"),
                     vis_input=d.get("vis_input"), arm=d.get("arm")))
        counts = [sum(x) for x in data[arm]]
        pct = [100 * c / len(x) for c, x in zip(counts, data[arm])]
        print(f"{arm:6s} counts={counts} mean={sum(pct)/len(pct):.1f} "
              f"range=[{min(pct):.1f},{max(pct):.1f}]")

    out = dict(seeds=args.seeds, arms=meta, comparisons={})
    print("\n== paired comparisons ==")
    for pair in args.pairs:
        hi, lo = pair.split(":")
        A, B = data[hi], data[lo]
        diffs, ps, det = [], [], []
        for i, s in enumerate(args.seeds):
            x, y, p = mcnemar(A[i], B[i])
            d = 100 * (sum(A[i]) - sum(B[i])) / len(A[i])
            diffs.append(d)
            ps.append(p)
            det.append(dict(seed=s, diff_pp=d, mcnemar_p=p,
                            hi_only=x, lo_only=y))
        mean = sum(diffs) / len(diffs)
        sd = (sum((d - mean) ** 2 for d in diffs) /
              (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else 0.0
        se = sd / math.sqrt(len(diffs))
        lo95, hi95 = pooled_ci(A, B, iters=args.bootstrap)
        n_pos = sum(1 for d in diffs if d > 0)
        t = mean / se if se else float("inf")
        print(f"{hi} - {lo}: mean {mean:+.1f}pp {n_pos}/{len(diffs)} pos, "
              f"seed se {se:.1f} (t={t:.2f}), pooled 95% CI "
              f"[{lo95:+.1f}, {hi95:+.1f}]")
        print("   " + " | ".join(
            f"{d:+.1f}(p={p:.3f}, {x}v{y})"
            for d, p, (x, y) in zip(diffs, ps, [(q["hi_only"], q["lo_only"])
                                                for q in det])))
        out["comparisons"][f"{hi}-{lo}"] = dict(
            mean_pp=mean, seed_se=se, t=t, n_pos=n_pos, n_seeds=len(diffs),
            ci95=[lo95, hi95], per_seed=det)

    if args.json_out:
        for arm in arms:
            out["arms"][arm] = dict(
                files=meta[arm],
                success="".join("1" if v else "0" for seed in data[arm]
                                for v in seed),
                per_seed={str(m["seed"]): "".join(
                    "1" if v else "0" for v in data[arm][i])
                    for i, m in enumerate(meta[arm])})
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
