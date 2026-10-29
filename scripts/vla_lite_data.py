#!/usr/bin/env python3
"""Derive VLA-lite datasets from the E0 multi-object data.

The E0 files already contain everything needed except explicit scene ids,
scene-level splits, and 32x32 obstacle/plan labels:

  data/multi_lowlevel_2w.npz   25,004 chunks, 600 episodes
  data/multi_visit_2w.npz      72,000 visit decisions, 12,000 episodes

`multi_lowlevel_2w.npz` stores `metas = [ep, ep, t, visit, sc_i, sc_j,
gc_i, gc_j, active>=0, grasp]`; episode id is the scene id.  The visit file
was generated with exactly 6 decisions per episode (3 objects x grasp/place),
so `scene_id = index // 6`, `visit_id = index % 6`.

Outputs
  data/vla_lite_2w.npz
    images [N,3,256,256] uint8, states [N,7], subgoals [N,4] normalized,
    actions [N,8,7], obstacles32 [N,32,32] uint8, plans32 [N,32,32] float16,
    goal_xy [N,2], scene_id/ep_id/visit_id/t/grasp/split [N], detour [N],
    st_mean/std, act_mean/std, ch_mean/std (train split only)
  data/vla_lite_visit_2w.npz
    images [N,3,256,256] uint8, chunks [N,8,4] normalized,
    states [N,7] normalized, metas [N,12], ch_mean/std, st_mean/std,
    scene_id/ep_id/visit_id/split [N]

Splits: low-level 450/75/75 episodes; visit-level 9,000/1,500/1,500 episodes
(75/12.5/12.5 for both).  External evaluation uses seed 777, disjoint from the
seed-0 data by construction.

Run:
  python scripts/vla_lite_data.py
  python scripts/vla_lite_data.py --limit 500 --out-low /tmp/vl_low.npz \
      --out-visit /tmp/vl_visit.npz
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from multi_task import XL, XH, YL, YH, cell_to_xy
from spatial_plan_data import geodesic_field

G_LOW = 64          # stored in multi_lowlevel_2w.npz
G_VLA = 32          # VLA-lite training grid
H = 8


def xy_to_cell_g(xy, g):
    j = int((xy[0] - XL) / (XH - XL) * g)
    i = int((YH - xy[1]) / (YH - YL) * g)
    return (min(max(i, 0), g - 1), min(max(j, 0), g - 1))


def derive_lowlevel(path, limit_episodes=0):
    d = np.load(path)
    metas_all = d["metas"]
    keep = np.ones(len(metas_all), bool)
    if limit_episodes:
        keep = metas_all[:, 0] < limit_episodes
    idx = np.flatnonzero(keep)
    n = len(idx)
    print(f"low-level: {n} samples / {len(np.unique(metas_all[idx, 0]))} "
          f"episodes from {path}", flush=True)

    fields = d["fields"][idx]
    metas = metas_all[idx]
    images = d["images"][idx]
    states = d["states"][idx].astype(np.float32)
    actions = d["actions"][idx].astype(np.float32)

    # obstacle: conservative 2x2 max-pool (any blocked 64-cell -> blocked 32-cell)
    obstacles32 = fields[:, 0].reshape(n, G_VLA, 2, G_VLA, 2).max(axis=(2, 4))
    obstacles32 = obstacles32.astype(np.uint8)

    # goal: metas stores the 64-grid goal cell; convert through world xy
    gc64 = metas[:, 6:8].astype(np.int64)
    goal_xy = np.stack([cell_to_xy(tuple(c)) for c in gc64]).astype(np.float32)
    plans32 = np.full((n, G_VLA, G_VLA), 1.0, np.float32)
    n_unreach = 0
    for k in range(n):
        gc = xy_to_cell_g(goal_xy[k], G_VLA)
        obs = obstacles32[k].astype(bool)
        obs[gc] = False
        dist, reach = geodesic_field(obs, gc)
        plan = np.where(np.isfinite(dist), dist / (2.0 * G_VLA), 1.0)
        plan[obs] = 0.0
        plans32[k] = plan.astype(np.float32)
        n_unreach += int(not reach.any())
    plans32 = plans32.astype(np.float16)

    scene_id = metas[:, 0].astype(np.int32)
    ep_id = metas[:, 1].astype(np.int32)
    visit_id = metas[:, 3].astype(np.int32)
    t = metas[:, 2].astype(np.int32)
    grasp = metas[:, 9].astype(np.uint8)
    split = np.where(ep_id < 450, 0, np.where(ep_id < 525, 1, 2)).astype(np.uint8)

    # renormalize subgoals/states/actions with train-split statistics only
    old_ch_mean = d["ch_mean"].astype(np.float32)
    old_ch_std = d["ch_std"].astype(np.float32)
    subgoals_raw = d["subgoals"][:n].astype(np.float32) * old_ch_std + old_ch_mean
    tr = split == 0
    ch_mean = subgoals_raw[tr].mean(0)
    ch_std = np.maximum(subgoals_raw[tr].std(0), 1e-3)
    subgoals = (subgoals_raw - ch_mean) / ch_std
    st_mean = states[tr].mean(0)
    st_std = np.maximum(states[tr].std(0), 1e-3)
    act_mean = actions[tr].reshape(-1, 7).mean(0)
    act_std = np.maximum(actions[tr].reshape(-1, 7).std(0), 1e-3)

    out = dict(
        images=images, states=states, subgoals=subgoals.astype(np.float32),
        actions=actions, obstacles32=obstacles32, plans32=plans32,
        goal_xy=goal_xy, scene_id=scene_id, ep_id=ep_id, visit_id=visit_id,
        t=t, grasp=grasp, split=split, detour=d["detour"][:n].astype(np.float32),
        st_mean=st_mean, st_std=st_std, act_mean=act_mean, act_std=act_std,
        ch_mean=ch_mean, ch_std=ch_std,
    )
    stats = {}
    for s, name in [(0, "train"), (1, "val"), (2, "test")]:
        m = split == s
        stats[name] = dict(
            samples=int(m.sum()), scenes=int(len(np.unique(scene_id[m]))),
            chunks=int(m.sum()),
        )
    print(f"  split {stats}", flush=True)
    print(f"  obstacle frac {float(obstacles32.mean()):.3f}  "
          f"plan range [{float(plans32.min()):.3f},{float(plans32.max()):.3f}]  "
          f"unreachable-scenes {n_unreach}", flush=True)
    print(f"  ch_mean {np.round(ch_mean, 3)} ch_std {np.round(ch_std, 3)}",
          flush=True)
    return out, stats


def derive_visit(path, limit_episodes=0):
    d = np.load(path)
    n = len(d["states"])
    if limit_episodes:
        n = min(n, limit_episodes * 6)
    if n % 6 != 0:
        raise ValueError(f"{path}: {n} samples is not a multiple of 6; "
                         "cannot infer scene ids from generation order")
    print(f"visit-level: {n} samples / {n // 6} episodes from {path}",
          flush=True)

    images = d["images"][:n]
    chunks = d["chunks"][:n].astype(np.float32)
    states = d["states"][:n].astype(np.float32)
    metas = d["metas"][:n].astype(np.float32)

    scene_id = (np.arange(n, dtype=np.int64) // 6).astype(np.int32)
    ep_id = scene_id.copy()
    visit_id = (np.arange(n, dtype=np.int64) % 6).astype(np.int32)
    split = np.where(scene_id < 9000, 0,
                     np.where(scene_id < 10500, 1, 2)).astype(np.uint8)

    old_ch_mean = d["ch_mean"].astype(np.float32)
    old_ch_std = d["ch_std"].astype(np.float32)
    chunks_raw = chunks * old_ch_std + old_ch_mean
    old_st_mean = d["st_mean"].astype(np.float32)
    old_st_std = d["st_std"].astype(np.float32)
    states_raw = states * old_st_std + old_st_mean
    tr = split == 0
    ch_mean = chunks_raw[tr].reshape(-1, 4).mean(0)
    ch_std = np.maximum(chunks_raw[tr].reshape(-1, 4).std(0), 1e-3)
    st_mean = states_raw[tr].mean(0)
    st_std = np.maximum(states_raw[tr].std(0), 1e-3)

    out = dict(
        images=images,
        chunks=((chunks_raw - ch_mean) / ch_std).astype(np.float32),
        states=((states_raw - st_mean) / st_std).astype(np.float32),
        metas=metas, ch_mean=ch_mean, ch_std=ch_std,
        st_mean=st_mean, st_std=st_std,
        scene_id=scene_id, ep_id=ep_id, visit_id=visit_id, split=split,
    )
    stats = {}
    for s, name in [(0, "train"), (1, "val"), (2, "test")]:
        m = split == s
        stats[name] = dict(samples=int(m.sum()),
                           scenes=int(len(np.unique(scene_id[m]))))
    print(f"  split {stats}", flush=True)
    print(f"  ch_mean {np.round(ch_mean, 3)} ch_std {np.round(ch_std, 3)}",
          flush=True)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-in", default="data/multi_lowlevel_2w.npz")
    ap.add_argument("--visit-in", default="data/multi_visit_2w.npz")
    ap.add_argument("--out-low", default="data/vla_lite_2w.npz")
    ap.add_argument("--out-visit", default="data/vla_lite_visit_2w.npz")
    ap.add_argument("--meta-out", default="data/vla_lite_meta.json")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="only first N episodes (for a quick pipeline test)")
    ap.add_argument("--skip-low", action="store_true")
    ap.add_argument("--skip-visit", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_low), exist_ok=True)
    meta = {}
    if not args.skip_low:
        out, stats = derive_lowlevel(args.low_in, args.limit_episodes)
        np.savez_compressed(args.out_low, **out)
        print(f"wrote {args.out_low} ({os.path.getsize(args.out_low)/1e6:.1f} MB)",
              flush=True)
        meta["low"] = stats
    if not args.skip_visit:
        out, stats = derive_visit(args.visit_in, args.limit_episodes)
        np.savez_compressed(args.out_visit, **out)
        print(f"wrote {args.out_visit} "
              f"({os.path.getsize(args.out_visit)/1e6:.1f} MB)", flush=True)
        meta["visit"] = stats
    meta["grid"] = G_VLA
    meta["chunk"] = H
    meta["low_source"] = args.low_in
    meta["visit_source"] = args.visit_in
    meta["limit_episodes"] = args.limit_episodes
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.meta_out}")


if __name__ == "__main__":
    main()
