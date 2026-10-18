#!/usr/bin/env python3
"""BC dataset for the visit-level (high-level) multi-object policy.

The policy is queried ONLY at hover states (z=NEUTRAL_Z above the previous
visit's end) and predicts the next visit's terminal subgoal
[gx, gy, CONTACT_Z, open]: open=0 -> grasp the object at (gx,gy),
open=1 -> place the carried object at (gx,gy). A fixed skill executor runs
the whole visit. This removes the skill-phase ambiguity that collapsed the
per-step subgoal policy: at a hover the label is a function of the scene
(which objects remain, their colors) and of whether an object is carried.

  images [N, 3, 256, 256] uint8
  chunks [N, 8, 4]        float32  NORMALIZED visit command, repeated 8x
  states [N, 7]           float32  NORMALIZED hover pose [x,y,z,rot,grip]
  metas  [N, 12]          float32  objs[6] + targets[6]
  + ch_mean/ch_std [4], st_mean/st_std [7]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from multi_task import (sample_scene, decision_samples, render_scene,
                        NEUTRAL_Z, N_OBJ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=3000)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/multi_visit.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--two-walls", action="store_true")
    ap.add_argument("--hover-noise", type=float, default=0.003,
                    help="hover xy jitter (m), matches the plant residual")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    chunk = args.chunk_size
    images, chunks, states, metas = [], [], [], []
    for ep in range(args.n_episodes):
        objs, tgts, perm, walls = sample_scene(
            rng, two_walls=args.two_walls)
        meta = np.concatenate([objs.reshape(-1), tgts.reshape(-1)]).astype(np.float32)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(
                objs, tgts, perm):
            eef = (xy + rng.normal(0, args.hover_noise, 2)).astype(np.float32)
            g = float(np.clip(grip + rng.normal(0, 0.01), 0.0, 1.0))
            if active >= 0:
                objs_now = objs_now.copy()
                objs_now[active] = eef
            st = np.array([eef[0], eef[1], NEUTRAL_Z, 0, 0, 0, g], np.float32)
            img = render_scene(objs_now, tgts, perm, placed, walls, eef)
            images.append(np.asarray(img, np.uint8).transpose(2, 0, 1))
            chunks.append(np.repeat(cmd[None], chunk, 0))
            states.append(st)
            metas.append(meta)
        if (ep + 1) % 500 == 0:
            print(f"  {ep+1}/{args.n_episodes} eps  {len(images)} samples",
                  flush=True)

    images = np.stack(images).astype(np.uint8)
    chunks = np.stack(chunks).astype(np.float32)
    states = np.stack(states).astype(np.float32)
    metas = np.stack(metas).astype(np.float32)

    ch_mean = chunks.reshape(-1, 4).mean(0)
    ch_std = chunks.reshape(-1, 4).std(0)
    ch_std = np.maximum(ch_std, 1e-3)          # z is constant (CONTACT_Z)
    st_mean = states.mean(0)
    st_std = np.maximum(states.std(0), 1e-3)   # rot is constant
    chunks = (chunks - ch_mean) / ch_std
    states = (states - st_mean) / st_std

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, images=images, chunks=chunks, states=states,
                        metas=metas, ch_mean=ch_mean, ch_std=ch_std,
                        st_mean=st_mean, st_std=st_std)
    print(f"wrote {args.out}: images {images.shape} chunks {chunks.shape} "
          f"states {states.shape}")
    print(f"  ch_mean {np.round(ch_mean,3)} ch_std {np.round(ch_std,3)}")
    print(f"  st_mean {np.round(st_mean,3)} st_std {np.round(st_std,3)}")


if __name__ == "__main__":
    main()
