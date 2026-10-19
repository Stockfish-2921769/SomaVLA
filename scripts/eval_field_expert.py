#!/usr/bin/env python3
"""E0 closed-loop eval: frozen visit-level high policy + field-conditioned
low-level action expert.

The visit-level checkpoint predicts one subgoal [gx,gy,gz,open] per visit from
a hover.  The low-level expert executes the visit with receding horizon
(H=8, execute 4), conditioned on image, 11-dim state and the plan field.
Fields are recomputed at every chunk boundary from the current environment.

Field source is either the trained arm (ckpt "arm") or an eval-time override
(--field-override gt/zero/random/wrong/none) for the swap gate.

Run:
  HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/venvs/lerobot/bin/python scripts/eval_field_expert.py \
      --ckpt ckpts/e0/gt-plan_s0/last.pt --n-episodes 200 --seed 777
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_task import (sample_scene, render_scene, MultiEnv, N_OBJ,
                        NEUTRAL_Z, CONTACT_Z, G)
from multi_lowlevel_data import (make_field, random_plan, wrong_goal_plan,
                                 oracle_cmd, H, STRIDE)
from field_common import (build_field_policy, load_field_ckpt, TASK,
                          STATE_DIM, ACTION_DIM)


def as_np(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else np.asarray(x)


class HighLevel:
    """Thin wrapper around the visit-level controller from eval_multi."""

    def __init__(self, ckpt_path, device, task=TASK):
        from train_smolvla import build_policy
        from eval_multi import VisitController
        policy, lang_ids, lang_mask = build_policy(device, task)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["state"], strict=False)
        policy.eval()
        self.ctrl = VisitController(policy, lang_ids, lang_mask, ck.get("norm"),
                                    device, chunk_size=8)
        self.norm = ck.get("norm")
        print(f"high-level {ckpt_path} @step {ck.get('step')}")

    def reset(self):
        self.ctrl.reset()

    def predict(self, state, env, tgts, perm, walls):
        return self.ctrl.predict(state, env, tgts, perm, walls)


class FieldLowLevel:
    """Receding-horizon field-conditioned expert producing plant targets."""

    def __init__(self, ckpt_path, device, field_override="gt",
                 measure_dev=False, exec_steps=STRIDE):
        self.measure_dev = measure_dev
        self.exec_steps = exec_steps
        ck = load_field_ckpt(ckpt_path, device)
        self.arm = ck["arm"]
        if field_override == "ckpt":
            field_override = {"no-field": "none", "gt-plan": "gt",
                              "zero-plan": "zero", "random-plan": "random",
                              "wrong-goal": "wrong"}[self.arm]
        self.field_override = field_override
        self.device = device
        wrapper, lang_ids, lang_mask = build_field_policy(device, self.arm)
        policy = wrapper.policy
        policy.load_state_dict(ck["state"], strict=False)
        if wrapper.field_encoder is not None:
            if ck.get("field") is None:
                raise ValueError("ckpt has no field encoder state")
            wrapper.field_encoder.load_state_dict(ck["field"])
        policy.eval()
        wrapper.eval()
        self.wrapper = wrapper
        self.policy = policy
        self.lang_ids = lang_ids
        self.lang_mask = lang_mask
        self.st_mean = as_np(ck["st_mean"]).astype(np.float32)
        self.st_std = as_np(ck["st_std"]).astype(np.float32)
        self.act_mean = as_np(ck["act_mean"]).astype(np.float32)
        self.act_std = as_np(ck["act_std"]).astype(np.float32)
        self.ch_mean = as_np(ck["ch_mean"]).astype(np.float32)
        self.ch_std = as_np(ck["ch_std"]).astype(np.float32)
        self.rng = np.random.RandomState(0)
        self.reset()
        print(f"low-level arm {self.arm} @step {ck.get('step')} "
              f"override={field_override}")

    def reset(self):
        self.chunk = None
        self.i = 0
        self.dev = None
        self.wrapper.clear_field()

    def _norm_state(self, state, cmd):
        st = (state[:STATE_DIM] - self.st_mean) / self.st_std
        st = st.astype(np.float32)
        st[3:6] = 0.0
        sub = ((cmd - self.ch_mean) / self.ch_std).astype(np.float32)
        return np.concatenate([st, sub])[None]

    def _field(self, state, env, cmd, walls, force=None):
        """Return [1,4,G,G] tensor or None (no-field)."""
        ov = self.field_override if force is None else force
        if self.arm == "no-field" or ov == "none":
            return None
        field, obstacle = make_field(env.objs, cmd[:2], walls,
                                     env.placed, env.active)
        if ov == "zero":
            field[1] = 0.0
        elif ov == "random":
            field[1] = random_plan(self.rng)
        elif ov == "wrong":
            field[1] = wrong_goal_plan(obstacle, state[:2], cmd[:2],
                                       self.rng)
        return torch.from_numpy(field[None]).float().to(self.device)

    def _predict(self, obs, field):
        if field is not None:
            self.wrapper.set_field(field)
        else:
            self.wrapper.clear_field()
        with torch.no_grad():
            a = self.policy.predict_action_chunk(obs)
        return a.detach().cpu().numpy().reshape(-1, ACTION_DIM)[:H]

    def act(self, state, env, cmd, walls, tgts, perm):
        """One plant target [x,y,z,rot(3),grip]."""
        if self.chunk is None or self.i >= self.exec_steps:
            img = render_scene(env.objs, tgts, perm, env.placed, walls,
                               state[:2])
            img01 = np.asarray(img, np.float32)[None].transpose(
                0, 3, 1, 2) / 255.0
            obs = {
                "observation.images.camera1": torch.from_numpy(img01).to(
                    self.device),
                "observation.state": torch.from_numpy(
                    self._norm_state(state, cmd)).to(self.device),
                "observation.language.tokens": self.lang_ids,
                "observation.language.attention_mask": self.lang_mask,
            }
            field = self._field(state, env, cmd, walls)
            rng_state = torch.get_rng_state()
            a = self._predict(obs, field)
            if (self.measure_dev and self.dev is None
                    and self.arm != "no-field" and self.field_override != "gt"):
                gt = self._field(state, env, cmd, walls, force="gt")
                torch.set_rng_state(rng_state)
                a_gt = self._predict(obs, gt)
                self.dev = float(np.linalg.norm(a_gt - a))
            self.chunk = a * self.act_std + self.act_mean
            self.i = 0
        t = self.chunk[self.i]
        self.i += 1
        return np.array([t[0], t[1], t[2], state[3], state[4], state[5], t[6]],
                        np.float32)


def lift_scripted(env, steps=10):
    xy = env.state[:2].copy()
    z0 = float(env.state[2])
    grip = float(env.state[6])
    n = 0
    for k in range(1, steps + 1):
        zz = z0 + (NEUTRAL_Z - z0) * k / steps
        env.step(np.array([xy[0], xy[1], zz, 0.0, 0.0, 0.0, grip], np.float32))
        n += 1
    return n


def run_episode(high, low, objs, tgts, perm, walls, rng, max_steps=1500,
                visit_cap=180, teacher_subgoal=False):
    env = MultiEnv(rng, objs, tgts, perm, walls)
    state = env.reset()
    if high is not None:
        high.reset()
    low.reset()
    n = 0
    visits = 0
    visit_ok = 0
    while n < max_steps and not env.success():
        cmd = (oracle_cmd(env, state) if high is None
               else high.predict(state, env, tgts, perm, walls))
        attached0 = env.attached
        placed0 = env.n_placed()
        done = (lambda: env.attached) if cmd[3] < 0.5 else \
               (lambda: env.n_placed() > placed0)
        vs = 0
        while vs < visit_cap and n < max_steps and not done():
            a = low.act(state, env, cmd, walls, tgts, perm)
            state, _ = env.step(a)
            n += 1
            vs += 1
        visits += 1
        visit_ok += int(done())
        n += lift_scripted(env)
        state = env.state.copy()
    return (env.success(), env.n_placed(), int(env.bumps.sum()),
            env.wall_hits, n, visits, visit_ok, low.dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="low-level field ckpt (last.pt from E0 training)")
    ap.add_argument("--high-ckpt", default="ckpts/visit_2w_uf/best.pt")
    ap.add_argument("--field-override",
                    choices=["ckpt", "gt", "zero", "random", "wrong", "none"],
                    default="ckpt",
                    help="eval-time field source; ckpt = use the trained arm")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scene-type", choices=["two-walls", "random"],
                    default="two-walls")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--visit-cap", type=int, default=180)
    ap.add_argument("--exec-steps", type=int, default=STRIDE,
                    help="actions executed per re-query (training stride is 4)")
    ap.add_argument("--measure-dev", action="store_true",
                    help="record first-chunk normalized L2 deviation vs GT "
                         "field (for the swap gate)")
    ap.add_argument("--teacher-subgoal", action="store_true",
                    help="use oracle nearest-first subgoals instead of the "
                         "high-level policy (E0 subgoal diagnostic)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="", help="text log path")
    ap.add_argument("--json-out", default="",
                    help="per-episode JSON for paired analysis")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    scenes = [sample_scene(rng, two_walls=(args.scene_type == "two-walls"))
              for _ in range(args.n_episodes)]

    high = None if args.teacher_subgoal else HighLevel(args.high_ckpt, device)
    low = FieldLowLevel(args.ckpt, device, field_override=args.field_override,
                        measure_dev=args.measure_dev,
                        exec_steps=args.exec_steps)
    ov = low.field_override

    ok = 0
    placed, bumps, wh, steps = [], [], [], []
    succ, devs = [], []
    visits_all, visits_ok = 0, 0
    lines = []
    for ep, (objs, tgts, perm, walls) in enumerate(scenes):
        # identical flow-matching noise stream across arms/scenes
        torch.manual_seed(args.seed * 100003 + ep)
        s, p, b, w, st, nv, vo, dev = run_episode(
            high, low, objs, tgts, perm, walls, rng,
            max_steps=args.max_steps, visit_cap=args.visit_cap,
            teacher_subgoal=args.teacher_subgoal)
        devs.append(dev)
        ok += int(s)
        succ.append(bool(s))
        placed.append(p); bumps.append(b); wh.append(w); steps.append(st)
        visits_all += nv; visits_ok += vo
        line = (f"  ep{ep:3d} success={int(s)} placed={p}/{N_OBJ} "
                f"bumps={b} wall_hits={w} steps={st} visits={vo}/{nv}")
        print(line, flush=True)
        lines.append(line)
        if ep == 0:
            print(f"    [episode {ep}: first visit of {nv}]", flush=True)

    n = len(scenes)
    summary = [
        f"\n── E0 field expert closed loop ({n} eps, seed {args.seed}) ──",
        f"  ckpt {args.ckpt}  field {ov}  high {args.high_ckpt}",
        f"  full success {ok}/{n} ({ok/n*100:.1f}%)   "
        f"mean placed {np.mean(placed):.2f}/{N_OBJ}",
        f"  visits ok {visits_ok}/{visits_all} "
        f"({visits_ok/max(visits_all,1)*100:.0f}%)  "
        f"mean bumps/ep {np.mean(bumps):.2f}  "
        f"wall_hits/ep {np.mean(wh):.1f}  steps/ep {np.mean(steps):.0f}",
    ]
    dv = [x for x in devs if x is not None]
    if dv:
        summary.append(f"  first-chunk L2 vs GT field: median {np.median(dv):.4f}"
                       f"  mean {np.mean(dv):.4f}  (n={len(dv)})")
    print("\n".join(summary))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines + summary) + "\n")
        print(f"wrote {args.out}")
    if args.json_out:
        import json
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "arm": low.arm,
                "field_override": ov,
                "high_ckpt": args.high_ckpt,
                "teacher_subgoal": args.teacher_subgoal,
                "exec_steps": args.exec_steps,
                "seed": args.seed,
                "n": n,
                "success": succ,
                "placed": placed,
                "bumps": bumps,
                "wall_hits": wh,
                "steps": steps,
                "visits_total": visits_all,
                "visits_ok": visits_ok,
                "first_chunk_dev": devs,
            }, f)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
