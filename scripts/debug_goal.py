#!/usr/bin/env python3
"""Instrument one closed-loop subgoal episode.

Prints, at every requery boundary, the EEF state, the predicted subgoal, its
distance to the goal, and a heuristic read of what skill the scene calls for.
Also prints per-step motion for the first steps after each query.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_task import (sample_scene, render_scene, MultiEnv, N_OBJ,
                        CONTACT_Z, GRASP_Z, LIFT_Z)
from train_smolvla import build_policy
from eval_multi import as_np

MULTI_TASK = ("Pick up each object and place it on the matching colored "
              "target, going around the walls.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/goal_base/best.pt")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--requery", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--zero-state", action="store_true")
    ap.add_argument("--per-step", type=int, default=3,
                    help="print this many steps after each query")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, lang_ids, lang_mask = build_policy(device, MULTI_TASK)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ck["state"], strict=False)
    policy.eval()
    norm = ck.get("norm")
    ch_mean = as_np(norm["ch_mean"]).astype(np.float32)
    ch_std = as_np(norm["ch_std"]).astype(np.float32)
    st_mean = as_np(norm["st_mean"]).astype(np.float32)
    st_std = as_np(norm["st_std"]).astype(np.float32)
    print(f"ckpt step {ck.get('step')}  norm ch_mean {np.round(ch_mean,3)}")

    rng = np.random.RandomState(args.seed)
    episodes = [sample_scene(rng) for _ in range(args.episodes)]
    for ep, (objs, tgts, perm, walls) in enumerate(episodes):
        env = MultiEnv(rng, objs, tgts, perm, walls)
        state = env.reset()
        print(f"\n=== ep {ep} === objs {np.round(objs,3).tolist()}")
        print(f"  tgts {np.round(tgts,3).tolist()} perm {perm.tolist()}")
        goal = None
        i = 10**9
        queries = 0
        for n in range(args.max_steps):
            if goal is None or i >= args.requery:
                img = render_scene(env.objs, tgts, perm, env.placed, walls,
                                   state[:2])
                img01 = np.asarray(img, np.float32)[None].transpose(0, 3, 1, 2) / 255.0
                st_n = ((state - st_mean) / st_std).astype(np.float32)
                st_n[st_std <= 1.001e-3] = 0.0
                if args.zero_state:
                    st_n = st_n.copy()
                    st_n[:2] = 0.0
                obs = {
                    "observation.images.camera1": torch.from_numpy(img01).to(device),
                    "observation.state": torch.from_numpy(st_n[None]).to(device),
                    "observation.language.tokens": lang_ids,
                    "observation.language.attention_mask": lang_mask,
                }
                with torch.no_grad():
                    a = policy.predict_action_chunk(obs)
                a = a.detach().cpu().numpy().reshape(-1, 4)
                goal = a[0] * ch_std + ch_mean
                i = 0
                queries += 1
                dg = np.linalg.norm(goal[:2] - state[:2])
                d_obj = np.linalg.norm(env.objs - goal[:2], axis=1)
                d_tgt = np.linalg.norm(tgts[perm] - goal[:2], axis=1)
                print(f" q{n:3d} st=({state[0]:.3f},{state[1]:.3f},{state[2]:.3f},"
                      f"g{state[6]:.2f}) goal=({goal[0]:.3f},{goal[1]:.3f},"
                      f"{goal[2]:.3f},o{goal[3]:.2f}) |dg|={dg:.3f} "
                      f"dObj={np.round(d_obj,3)} dTgt={np.round(d_tgt,3)} "
                      f"att={int(env.attached)} placed={env.n_placed()} "
                      f"wh={env.wall_hits}")
            i += 1
            tgt = goal
            tp = np.concatenate([tgt[:3], state[3:6], [tgt[3]]]).astype(np.float32)
            prev = state.copy()
            state, info = env.step(tp)
            if i <= args.per_step:
                print(f"      t{n:3d} -> ({state[0]:.3f},{state[1]:.3f},"
                      f"{state[2]:.3f},g{state[6]:.2f}) "
                      f"move={np.round(state[:2]-prev[:2],3)} "
                      f"att={int(info['attached'])} placed={env.n_placed()} "
                      f"wh={env.wall_hits}")
            if env.success():
                print(f"  SUCCESS at step {n+1}")
                break
        else:
            print(f"  timeout  placed={env.n_placed()}/3  wall_hits={env.wall_hits} "
                  f"queries={queries}")


if __name__ == "__main__":
    main()
