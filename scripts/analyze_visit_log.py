#!/usr/bin/env python3
"""Per-visit analysis of VLA-lite eval JSONs: does the low level rescue
visits whose high-level command was already off?

Every finished visit in ``visit_log`` (written by eval_vla_lite.py) is
bucketed by the command the shared high level issued, compared with the
teacher command at the same state:

  on     same candidate as the teacher, xy error within tolerance
  off    same candidate, xy error beyond tolerance (grasp radius for pick
         visits, place tolerance for carry visits); a low level that sees
         the scene could still complete these by correcting the target
  wrong  command points at a different object/target than the teacher

If a field helps mainly by visual correction, vis/nca should beat the blind
``none`` arm on ``off`` visits by much more than on ``on`` visits; the
``off-on`` row is that difference-in-differences.  Visits are not paired
across arms (closed-loop states diverge after the first divergent action),
so arms are compared through per-seed completion rates, summarized at the
seed level as in analyze_vla_lite_5seed.py.
"""
import argparse
import json
import math
from pathlib import Path

PICK_TOL = 0.025             # multi_task.CONTACT_R
CARRY_TOL = 0.020            # multi_task.PLACE_TOL
BUCKETS = ("on", "off", "wrong")


def load_visits(logs, prefix, arm, seed):
    path = Path(logs) / f"{prefix}_{arm}_s{seed}_pred.json"
    d = json.load(open(path))
    if "visit_log" not in d:
        raise SystemExit(f"{path}: no visit_log (evaluated before per-visit "
                         "logging was added; re-evaluate with this branch)")
    return [v for scene in d["visit_log"] for v in scene], path


def bucket(v, pick_tol, carry_tol):
    if not v["same_choice"]:
        return "wrong"
    tol = carry_tol if v["attached"] else pick_tol
    return "off" if v["xy_err"] > tol else "on"


def summarize(visits, pick_tol, carry_tol):
    fin = [v for v in visits if v.get("finished", True)]
    out = dict(n_visits=len(fin), n_unfinished=len(visits) - len(fin))
    for kind in ("all", "pick", "carry"):
        sel = [v for v in fin
               if kind == "all" or v["attached"] == (kind == "carry")]
        for b in BUCKETS:
            vs = [v for v in sel if bucket(v, pick_tol, carry_tol) == b]
            out[f"{kind}/{b}"] = dict(n=len(vs), ok=sum(v["ok"] for v in vs))
        out[f"{kind}/total"] = dict(n=len(sel), ok=sum(v["ok"] for v in sel))
    return out


def rate(c):
    return 100.0 * c["ok"] / c["n"] if c["n"] else float("nan")


def seed_stats(diffs):
    ds = [d for d in diffs if not math.isnan(d)]
    n = len(ds)
    if n == 0:
        return dict(mean=float("nan"), se=float("nan"), t=float("nan"),
                    n_pos=0, n=0)
    mean = sum(ds) / n
    sd = (sum((d - mean) ** 2 for d in ds) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and not math.isnan(se) else float("nan")
    return dict(mean=mean, se=se, t=t, n_pos=sum(d > 0 for d in ds), n=n)


def _clean(x):
    """NaN -> None so the JSON summary stays standard JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/vla_lite")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prefix-map", nargs="+",
                    default=["vis=conf", "none=recheck", "cnn=recheck",
                             "nca=recheck"])
    ap.add_argument("--arms", nargs="+", default=["none", "vis", "cnn", "nca"])
    ap.add_argument("--pairs", nargs="+",
                    default=["vis:none", "nca:none", "nca:vis"])
    ap.add_argument("--kind", choices=["all", "pick", "carry"], default="all",
                    help="visit type used for the tables")
    ap.add_argument("--pick-tol", type=float, default=PICK_TOL)
    ap.add_argument("--carry-tol", type=float, default=CARRY_TOL)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    prefix = dict(p.split("=", 1) for p in args.prefix_map)
    arms = list(dict.fromkeys(
        args.arms + [a for pair in args.pairs for a in pair.split(":")]))
    k = args.kind

    stats = {}
    print(f"buckets ({k} visits): on/off = teacher's candidate, xy err vs "
          f"tol (pick {args.pick_tol * 100:.1f} cm, carry "
          f"{args.carry_tol * 100:.1f} cm); wrong = other candidate")
    print("\n== completion rate % (n visits) ==")
    for arm in arms:
        stats[arm] = []
        for s in args.seeds:
            visits, path = load_visits(args.logs, prefix[arm], arm, s)
            st = summarize(visits, args.pick_tol, args.carry_tol)
            st.update(seed=s, file=str(path))
            stats[arm].append(st)
        n_tot = sum(st[f"{k}/total"]["n"] for st in stats[arm])
        for b in BUCKETS + ("total",):
            cells = [st[f"{k}/{b}"] for st in stats[arm]]
            n_b = sum(c["n"] for c in cells)
            ok_b = sum(c["ok"] for c in cells)
            per = " ".join(f"{rate(c):5.1f}({c['n']})" for c in cells)
            print(f"{arm:6s} {b:5s} share {100 * n_b / max(n_tot, 1):5.1f}%  "
                  f"pooled {100 * ok_b / max(n_b, 1):5.1f}%  | {per}")
        unfinished = sum(st["n_unfinished"] for st in stats[arm])
        print(f"{arm:6s} unfinished (cut by step budget): {unfinished}")

    print(f"\n== seed-level differences in completion rate, pp ({k}) ==")
    comps = {}
    for pair in args.pairs:
        hi, lo = pair.split(":")
        row = {}
        per_bucket = {}
        for b in BUCKETS + ("total",):
            diffs = [rate(a[f"{k}/{b}"]) - rate(c[f"{k}/{b}"])
                     for a, c in zip(stats[hi], stats[lo])]
            per_bucket[b] = diffs
            row[b] = dict(per_seed=diffs, **seed_stats(diffs))
        # visual-correction signature: gain on off visits beyond gain on on
        did = [o - n for o, n in zip(per_bucket["off"], per_bucket["on"])]
        row["off-on"] = dict(per_seed=did, **seed_stats(did))
        comps[pair] = row
        print(f"{hi} - {lo}:")
        for b, r in row.items():
            seeds = " ".join(f"{d:+.1f}" for d in r["per_seed"])
            print(f"   {b:6s} mean {r['mean']:+6.1f}  {r['n_pos']}/{r['n']} "
                  f"pos  t={r['t']:.2f}  | {seeds}")

    if args.json_out:
        out = dict(seeds=args.seeds, kind=k, pick_tol=args.pick_tol,
                   carry_tol=args.carry_tol, arms=stats, comparisons=comps)
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(_clean(out), open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
