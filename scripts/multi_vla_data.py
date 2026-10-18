#!/usr/bin/env python3
"""BC dataset generator for the multi-object long-horizon task.

Runs the procedural expert teacher (nearest-first order + geodesic-routed
transport, scripts/multi_task.py) and records (image, subgoal-chunk) pairs:

  images [N, 3, 256, 256] uint8    raw RGB 0..255 (4x smaller than float32)
  chunks [N, 8, 4]        float32  NORMALIZED skill subgoals [gx,gy,gz,open]
  states [N, 7]           float32  NORMALIZED EEF pose at chunk start
  metas  [N, 12]          float32  objs[6] + targets[6] (meters, for aux/planes)
  + ch_mean/ch_std [4], st_mean/st_std [7]  — the normalization stats

The target is the *active skill's subgoal*, not the next frame pose. A next-pose
target is R^2=0.999 predictable from the proprio state, so BC rationally ignores
the image and stalls; a subgoal (object/placement point, ~10-30cm away) is only
R^2~0.6 state-predictable, so the scene must be read to resolve it. A fixed
low-level controller drives the EEF to the predicted subgoal.

Memory: 600 episodes ~= 70K chunks ~= 14 GB uint8 images. Kept uint8 end-to-end
so both generation and training stay well under the 62 GB box.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from multi_task import (sample_scene, teacher_rollout, render_scene, N_OBJ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/multi_train.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failed", action="store_true",
                    help="also keep demos whose teacher rollout would fail "
                         "(default: keep all — BC imitates the policy, not "
                         "only its successes)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    chunk = args.chunk_size
    images, chunks, states, metas = [], [], [], []
    n_done = 0
    while n_done < args.n_episodes:
        objs, tgts, perm, walls = sample_scene(rng)
        st, goals, oh, ph = teacher_rollout(objs, tgts, perm, walls, rng)
        T = len(st)
        meta = np.concatenate([objs.reshape(-1), tgts.reshape(-1)]).astype(np.float32)
        for t in range(0, T - chunk, chunk):
            img = render_scene(oh[t], tgts, perm, ph[t], walls, st[t][:2])
            arr8 = np.asarray(img, np.uint8)                   # [256,256,3]
            images.append(arr8.transpose(2, 0, 1))             # CHW
            # skill subgoal at each step of the window (mostly constant per
            # skill; piecewise-constant lets the controller re-read the goal).
            chunks.append(np.stack([goals[t + k] for k in range(chunk)]))
            states.append(st[t].copy())
            metas.append(meta)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{args.n_episodes} eps  {len(images)} chunks", flush=True)

    images = np.stack(images).astype(np.uint8)                 # [N,3,256,256]
    chunks = np.stack(chunks).astype(np.float32)
    states = np.stack(states).astype(np.float32)
    metas = np.stack(metas).astype(np.float32)

    ch_mean = chunks.reshape(-1, 4).mean(0)
    ch_std = chunks.reshape(-1, 4).std(0) + 1e-6
    st_mean = states.mean(0)
    st_std = states.std(0) + 1e-6
    chunks = (chunks - ch_mean) / ch_std
    states = (states - st_mean) / st_std

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, images=images, chunks=chunks, states=states,
                        metas=metas, ch_mean=ch_mean, ch_std=ch_std,
                        st_mean=st_mean, st_std=st_std)
    print(f"wrote {args.out}: images {images.shape} chunks {chunks.shape} "
          f"states {states.shape} from {n_done} eps")
    print(f"  ch_mean {np.round(ch_mean,3)} ch_std {np.round(ch_std,3)}")


if __name__ == "__main__":
    main()
