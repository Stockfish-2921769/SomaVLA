#!/usr/bin/env python3
"""Vectorized E0 closed-loop eval: K episodes stepped in lockstep so the
low-level expert (and high-level queries) are batched.

Semantics match eval_field_expert.py: same scenes, same field/override
options, same exec-stride / visit cap / scripted lift.  Batching changes the
flow-matching noise composition relative to the sequential evaluator, so
results are not bit-identical; scene pairing is exact.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_task import (sample_scene, render_scene, MultiEnv, N_OBJ,
                        NEUTRAL_Z, G)
from multi_lowlevel_data import oracle_cmd, H, STRIDE
from field_common import ACTION_DIM
from eval_field_expert import HighLevel, FieldLowLevel


class Ep:
    def __init__(self, scene_id, objs, tgts, perm, walls, rng, max_steps,
                 visit_cap, exec_steps):
        self.scene_id = scene_id
        self.env = MultiEnv(rng, objs, tgts, perm, walls)
        self.state = self.env.reset()
        self.steps = 0
        self.max_steps = max_steps
        self.visit_cap = visit_cap
        self.exec_steps = exec_steps
        self.done = False
        self.need_high = True
        self.in_visit = False
        self.cmd = None
        self.attached0 = False
        self.placed0 = 0
        self.visit_steps = 0
        self.chunk = None
        self.chunk_i = 0
        self.lift = 0
        self.lift_n = 10
        self.lift_k = 0
        self.lift_xy = None
        self.lift_z0 = 0.0
        self.lift_grip = 0.0
        self.dev = None
        self.visits = 0
        self.visits_ok = 0
        self.query_i = 0

    def visit_done(self):
        if self.cmd[3] < 0.5:
            return bool(self.env.attached)
        return self.env.n_placed() > self.placed0

    def finish_visit(self):
        self.in_visit = False
        self.chunk = None
        self.lift = self.lift_n
        self.lift_k = 0
        self.lift_xy = self.state[:2].copy()
        self.lift_z0 = float(self.state[2])
        self.lift_grip = float(self.state[6])

    def end_visit(self):
        self.visits += 1
        self.visits_ok += int(self.visit_done())
        self.finish_visit()

    def action(self):
        if self.lift > 0:
            xy = self.lift_xy
            z0 = self.lift_z0
            self.lift_k += 1
            zz = z0 + (NEUTRAL_Z - z0) * self.lift_k / self.lift_n
            grip = self.lift_grip
            self.lift -= 1
            return np.array([xy[0], xy[1], zz, 0.0, 0.0, 0.0, grip],
                            np.float32)
        t = self.chunk[self.chunk_i]
        self.chunk_i += 1
        if self.chunk_i >= self.exec_steps:
            self.chunk = None
        return np.array([t[0], t[1], t[2], self.state[3], self.state[4],
                         self.state[5], t[6]], np.float32)


def norm_high(ctrl, state):
    return ctrl.norm_state(state)


def render_obs(env, tgts, perm, walls, state):
    img = render_scene(env.objs, tgts, perm, env.placed, walls, state[:2])
    return np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0


def high_predict(high, eps, scenes):
    ctrl = high.ctrl
    imgs = np.stack([render_obs(e.env, scenes[e.scene_id][1],
                                scenes[e.scene_id][2],
                                scenes[e.scene_id][3], e.state)
                     for e in eps])
    sts = np.stack([norm_high(ctrl, e.state) for e in eps])
    obs = {
        "observation.images.camera1": torch.from_numpy(imgs).to(
            ctrl.device),
        "observation.state": torch.from_numpy(sts).to(ctrl.device),
        "observation.language.tokens": ctrl.lang_ids.expand(len(eps), -1),
        "observation.language.attention_mask":
            ctrl.lang_mask.expand(len(eps), -1),
    }
    with torch.no_grad():
        a = ctrl.policy.predict_action_chunk(obs)
    a = a.detach().cpu().numpy().reshape(len(eps), -1, 4)[:, 0]
    return a * ctrl.ch_std + ctrl.ch_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--high-ckpt", default="ckpts/visit_2w_uf/best.pt")
    ap.add_argument("--field-override",
                    choices=["ckpt", "gt", "zero", "random", "wrong", "none"],
                    default="ckpt")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scene-type", choices=["two-walls", "random"],
                    default="two-walls")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--visit-cap", type=int, default=180)
    ap.add_argument("--exec-steps", type=int, default=STRIDE)
    ap.add_argument("--batch-episodes", type=int, default=16)
    ap.add_argument("--measure-dev", action="store_true")
    ap.add_argument("--det-noise", action="store_true",
                    help="flow-matching noise keyed by (scene, query index), "
                         "so arms see identical noise at aligned decisions")
    ap.add_argument("--teacher-subgoal", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    scenes = [sample_scene(rng, two_walls=(args.scene_type == "two-walls"))
              for _ in range(args.n_episodes)]

    high = None if args.teacher_subgoal else HighLevel(args.high_ckpt, device)
    low = FieldLowLevel(args.ckpt, device, field_override=args.field_override,
                        measure_dev=args.measure_dev,
                        exec_steps=args.exec_steps)
    ov = low.field_override
    print(f"vec eval: {args.n_episodes} eps, batch {args.batch_episodes}, "
          f"exec {args.exec_steps}, field {ov}"
          f"{', det-noise' if args.det_noise else ''}", flush=True)

    K = min(args.batch_episodes, args.n_episodes)
    # deterministic per-scene env rng: scenes already fixed; episode noise from
    # a per-scene stream so arms see the same plant noise schedule.
    ep_rngs = [np.random.RandomState(args.seed * 7919 + i)
               for i in range(args.n_episodes)]
    active = {}
    results = {}
    next_id = 0

    def spawn(i):
        objs, tgts, perm, walls = scenes[i]
        ep = Ep(i, objs, tgts, perm, walls, ep_rngs[i], args.max_steps,
                args.visit_cap, args.exec_steps)
        return ep

    for _ in range(K):
        active[next_id] = spawn(next_id)
        next_id += 1
    torch.manual_seed(args.seed * 100003)

    tick = 0
    while active:
        tick += 1
        eps = list(active.values())
        # high-level queries
        hi = [e for e in eps if e.need_high and not e.done]
        if hi:
            if high is None:
                cmds = [oracle_cmd(e.env, e.state) for e in hi]
            else:
                cmds = high_predict(high, hi, scenes)
            for e, c in zip(hi, cmds):
                e.cmd = c.astype(np.float32)
                e.attached0 = bool(e.env.attached)
                e.placed0 = e.env.n_placed()
                e.visit_steps = 0
                e.need_high = False
                e.in_visit = True
                e.chunk = None
                e.chunk_i = 0
                if e.visit_done():          # no-op intent (e.g. grasp attached)
                    e.end_visit()
        # low-level chunk predictions
        lo = [e for e in eps if e.in_visit and e.chunk is None
              and e.lift == 0 and not e.done]
        if lo:
            # all scenes share tgts/perm/walls shape; use each episode's own
            chunks = low_predict_multi(low, lo, scenes, args.measure_dev,
                                       det_noise=args.det_noise)
            for e, c in zip(lo, chunks):
                e.chunk = c
                e.chunk_i = 0
        # one plant step for every episode
        for e in eps:
            if e.done:
                continue
            a = e.action()
            e.state, _ = e.env.step(a)
            e.steps += 1
            if e.in_visit:
                e.visit_steps += 1
                if e.visit_done() or e.visit_steps >= args.visit_cap:
                    e.end_visit()
            if e.lift == 0 and not e.in_visit:
                e.need_high = True
            if e.env.success() or e.steps >= args.max_steps:
                e.done = True
        # retire + refill
        for sid in [k for k, e in active.items() if e.done]:
            e = active.pop(sid)
            results[sid] = e
            if next_id < args.n_episodes:
                active[next_id] = spawn(next_id)
                next_id += 1
        if tick % 500 == 0:
            print(f"  tick {tick} active {len(active)} done "
                  f"{len(results)}/{args.n_episodes}", flush=True)

    ok = 0
    succ, placed, bumps, wh, steps, devs = [], [], [], [], [], []
    visits_all, visits_ok = 0, 0
    lines = []
    for i in range(args.n_episodes):
        e = results[i]
        s = bool(e.env.success())
        ok += int(s)
        succ.append(s)
        placed.append(e.env.n_placed())
        bumps.append(int(e.env.bumps.sum()))
        wh.append(e.env.wall_hits)
        steps.append(e.steps)
        visits_all += e.visits
        visits_ok += e.visits_ok
        if e.dev is not None:
            devs.append(e.dev)
        line = (f"  ep{i:3d} success={int(s)} placed={e.env.n_placed()}/{N_OBJ} "
                f"bumps={int(e.env.bumps.sum())} wall_hits={e.env.wall_hits} "
                f"steps={e.steps}")
        print(line, flush=True)
        lines.append(line)
    n = len(results)
    summary = [
        f"\n── E0 vec field expert closed loop ({n} eps, seed {args.seed}) ──",
        f"  ckpt {args.ckpt}  field {ov}  high {args.high_ckpt}",
        f"  full success {ok}/{n} ({ok/n*100:.1f}%)   "
        f"mean placed {np.mean(placed):.2f}/{N_OBJ}",
        f"  mean bumps/ep {np.mean(bumps):.2f}  "
        f"wall_hits/ep {np.mean(wh):.1f}  steps/ep {np.mean(steps):.0f}",
    ]
    if devs:
        summary.append(f"  first-chunk L2 vs GT field: median "
                       f"{np.median(devs):.4f}  mean {np.mean(devs):.4f}  "
                       f"(n={len(devs)})")
    print("\n".join(summary))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines + summary) + "\n")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "arm": low.arm, "field_override": ov,
                "high_ckpt": args.high_ckpt,
                "teacher_subgoal": args.teacher_subgoal,
                "exec_steps": args.exec_steps, "seed": args.seed,
                "n": n, "success": succ, "placed": placed, "bumps": bumps,
                "wall_hits": wh, "steps": steps,
                "visits_total": visits_all, "visits_ok": visits_ok,
                "first_chunk_dev": devs, "vectorized": True,
                "det_noise": args.det_noise,
            }, f)


def det_noise_for(eps, low):
    """Deterministic flow noise keyed by (scene, per-episode query index).

    Identical across arms at aligned decisions regardless of batch
    composition, which removes the independent flow-noise variance from
    arm--arm comparisons.
    """
    shape = (low.policy.config.chunk_size, low.policy.config.max_action_dim)
    rows = [np.random.RandomState(
        (e.scene_id * 1000003 + e.query_i * 9176 + 12345) & 0x7FFFFFFF)
        .standard_normal(shape) for e in eps]
    return torch.from_numpy(np.stack(rows).astype(np.float32)).to(low.device)


def low_predict_multi(low, eps, scenes, measure, det_noise=False):
    """low_predict with per-episode walls/targets (scenes differ per row)."""
    imgs = np.stack([render_obs(e.env, scenes[e.scene_id][1],
                                scenes[e.scene_id][2],
                                scenes[e.scene_id][3], e.state)
                     for e in eps])
    sts = np.stack([low._norm_state(e.state, e.cmd)[0] for e in eps])
    obs = {
        "observation.images.camera1": torch.from_numpy(imgs).to(low.device),
        "observation.state": torch.from_numpy(sts).to(low.device),
        "observation.language.tokens": low.lang_ids.expand(len(eps), -1),
        "observation.language.attention_mask":
            low.lang_mask.expand(len(eps), -1),
    }
    fields = [low._field(e.state, e.env, e.cmd, scenes[e.scene_id][3])
              for e in eps]
    field = None if all(f is None for f in fields) else torch.cat(
        [f if f is not None else torch.zeros(
            1, 4, G, G, device=low.device) for f in fields], 0)
    noise = det_noise_for(eps, low) if det_noise else None
    rng_state = torch.get_rng_state()
    a = _predict_batch(low, obs, field, noise)
    dev_rows = [i for i, e in enumerate(eps)
                if measure and e.dev is None and low.arm != "no-field"
                and low.field_override != "gt"]
    if dev_rows:
        gt = torch.cat([low._field(e.state, e.env, e.cmd,
                                   scenes[e.scene_id][3], force="gt")
                        for e in eps], 0)
        if noise is None:
            torch.set_rng_state(rng_state)
        a_gt = _predict_batch(low, obs, gt, noise)
        for i in dev_rows:
            eps[i].dev = float(np.linalg.norm(a_gt[i] - a[i]))
    for e in eps:
        e.query_i += 1
    return a * low.act_std + low.act_mean


def _predict_batch(low, obs, field, noise=None):
    """Batched variant of FieldLowLevel._predict: [B,H,ACTION_DIM]."""
    if field is not None:
        low.wrapper.set_field(field)
    else:
        low.wrapper.clear_field()
    with torch.no_grad():
        a = low.policy.predict_action_chunk(obs, noise=noise)
    n = a.shape[0]
    return a.detach().cpu().numpy().reshape(n, -1, ACTION_DIM)[:, :H]


if __name__ == "__main__":
    main()
