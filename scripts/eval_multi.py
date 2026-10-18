#!/usr/bin/env python3
"""Closed-loop eval of a VLA on the multi-object long-horizon task.

Runs the policy (SmolVLA-0.5B fine-tune) as a receding-horizon chunk controller
on a FIXED set of evaluation scenes: render the current scene at each chunk
boundary (objects left, targets, walls, EEF) -> decode an 8-step chunk of
absolute [x,y,z,open] -> execute open-loop -> re-render. Success = all N objects
placed on their color-matched targets within the step budget.

Reported per variant: full-success rate, mean objects placed (/N), mean bumps,
mean wall hits, mean steps.

Run inside the lerobot venv:
  HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 \
  python scripts/eval_multi.py --ckpt ckpts/smolvla_multi/best.pt --n-episodes 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_task import (sample_scene, render_scene, MultiEnv, N_OBJ,
                        CONTACT_Z, GRASP_Z, LIFT_Z, NEUTRAL_Z,
                        CONTACT_R, PLACE_TOL, route_xy, _polyline_traj,
                        xy_to_cell, G)
from multi_plan_data import (obstacle_grid, field_path, field_path_escape,
                             field_path_cost, make_ext, EXT_DIM)
from soma.spatial_baselines import build_model, predict_field


def as_np(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else np.asarray(x)


class MultiChunkController:
    def __init__(self, policy, lang_ids, lang_mask, norm, device, chunk_size=8,
                 zero_state=False):
        self.policy = policy
        self.device = device
        self.lang_ids = lang_ids.to(device)
        self.lang_mask = lang_mask.to(device).bool()
        self.chunk_size = chunk_size
        self.zero_state = zero_state
        if norm:
            self.ch_mean = as_np(norm["ch_mean"]).astype(np.float32)
            self.ch_std = as_np(norm["ch_std"]).astype(np.float32)
            self.st_mean = as_np(norm["st_mean"]).astype(np.float32)
            self.st_std = as_np(norm["st_std"]).astype(np.float32)
        else:
            self.ch_mean = np.zeros(4, np.float32); self.ch_std = np.ones(4, np.float32)
            self.st_mean = np.zeros(7, np.float32); self.st_std = np.ones(7, np.float32)
        # Dims that were constant in training were fed as exactly 0; env noise
        # on them (e.g. rot random walk) would blow up by 1/st_std. The data
        # generator clamps degenerate stds at 1e-3, so use <=1e-3.
        self.const = self.st_std <= 1.001e-3

    def norm_state(self, state):
        st_n = ((state - self.st_mean) / self.st_std).astype(np.float32)
        st_n[self.const] = 0.0
        if self.zero_state:
            st_n = st_n.copy()
            st_n[:2] = 0.0
        return st_n

    def reset(self):
        self.policy.reset()
        self.chunk = None
        self.i = 0

    def step(self, state, env, tgts, perm, walls):
        if self.chunk is None or self.i >= self.chunk_size:
            img = render_scene(env.objs, tgts, perm, env.placed, walls, state[:2])
            img01 = np.asarray(img, np.float32)[None].transpose(0, 3, 1, 2) / 255.0
            st_n = self.norm_state(state)
            obs = {
                "observation.images.camera1": torch.from_numpy(img01).to(self.device),
                "observation.state": torch.from_numpy(st_n[None]).to(self.device),
                "observation.language.tokens": self.lang_ids,
                "observation.language.attention_mask": self.lang_mask,
            }
            with torch.no_grad():
                a = self.policy.predict_action_chunk(obs)
            a = a.detach().cpu().numpy().reshape(-1, 4)[: self.chunk_size]
            self.chunk = a * self.ch_std + self.ch_mean
            self.i = 0
        t = self.chunk[self.i]
        self.i += 1
        return t


class SubgoalController(MultiChunkController):
    """Predict the active skill's subgoal [gx,gy,gz,open] from the image and
    hold it for `requery` control steps. A fixed low-level controller (the
    env's damped plant) drives the EEF to that subgoal; the policy re-observes
    every requery steps and predicts the next one."""

    def __init__(self, *a, requery=8, **kw):
        super().__init__(*a, **kw)
        self.requery = requery

    def reset(self):
        super().reset()
        self.goal = None
        self.i = 10**9

    def step(self, state, env, tgts, perm, walls):
        if self.goal is None or self.i >= self.requery:
            img = render_scene(env.objs, tgts, perm, env.placed, walls, state[:2])
            img01 = np.asarray(img, np.float32)[None].transpose(0, 3, 1, 2) / 255.0
            st_n = self.norm_state(state)
            obs = {
                "observation.images.camera1": torch.from_numpy(img01).to(self.device),
                "observation.state": torch.from_numpy(st_n[None]).to(self.device),
                "observation.language.tokens": self.lang_ids,
                "observation.language.attention_mask": self.lang_mask,
            }
            with torch.no_grad():
                a = self.policy.predict_action_chunk(obs)
            a = a.detach().cpu().numpy().reshape(-1, 4)
            self.goal = a[0] * self.ch_std + self.ch_mean
            self.i = 0
        self.i += 1
        return self.goal


class VisitController(MultiChunkController):
    """Query once per visit, from a hover. Predicts the visit's terminal
    subgoal [gx,gy,CONTACT_Z,open]: open<0.5 -> grasp object at (gx,gy),
    otherwise place the carried object at (gx,gy)."""

    def reset(self):
        super().reset()

    def predict(self, state, env, tgts, perm, walls):
        img = render_scene(env.objs, tgts, perm, env.placed, walls, state[:2])
        img01 = np.asarray(img, np.float32)[None].transpose(0, 3, 1, 2) / 255.0
        st_n = self.norm_state(state)
        obs = {
            "observation.images.camera1": torch.from_numpy(img01).to(self.device),
            "observation.state": torch.from_numpy(st_n[None]).to(self.device),
            "observation.language.tokens": self.lang_ids,
            "observation.language.attention_mask": self.lang_mask,
        }
        with torch.no_grad():
            a = self.policy.predict_action_chunk(obs)
        a = a.detach().cpu().numpy().reshape(-1, 4)
        return a[0] * self.ch_std + self.ch_mean


def _pose(xy, z, grip):
    return np.array([xy[0], xy[1], z, 0.0, 0.0, 0.0, grip], np.float32)


class NCAWaypoints:
    """Predicted plan field -> waypoints. Mirrors the training-time occupancy
    (walls + unplaced non-goal objects, same inflation as Dijkstra)."""

    def __init__(self, ckpt, device, extract="greedy", radius=24):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.name = ck["model_name"]
        self.extract = extract
        self.radius = radius
        self.model = build_model(self.name, **ck.get("arch", {})).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.g = ck.get("g", 64)
        self.k = ck.get("k_steps")
        self.device = device
        print(f"plan field {self.name} G={self.g} step {ck.get('step')} "
              f"valMSE {ck.get('val_mse', float('nan')):.4f} "
              f"extract={self.extract} radius={self.radius}")

    @torch.no_grad()
    def __call__(self, start_xy, goal_xy, env):
        obstacle = obstacle_grid(env.objs, goal_xy, env.walls,
                                 env.placed, env.active)
        sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
        obstacle[sc] = False
        obstacle[gc] = False
        ext = make_ext(obstacle, gc)
        ext_t = torch.from_numpy(
            ext.reshape(EXT_DIM, -1).T[None]).to(self.device)
        field = predict_field(self.model, self.name, ext_t, self.g,
                              k_steps=self.k if self.name == "nca" else None)
        field = field[0].reshape(self.g, self.g).cpu().numpy()
        if self.extract == "cost":
            return field_path_cost(field, start_xy, goal_xy, obstacle)
        if self.extract == "escape":
            return field_path_escape(field, start_xy, goal_xy, obstacle,
                                     radius=self.radius)
        return field_path(field, start_xy, goal_xy, obstacle)


class GTWaypoints:
    """Oracle plan field (normalized geodesic distance) through the same
    descent readout and executor: the pipeline's ceiling without field error."""

    def __init__(self, radius=24):
        self.radius = radius

    def __call__(self, start_xy, goal_xy, env):
        from spatial_plan_data import geodesic_field
        obstacle = obstacle_grid(env.objs, goal_xy, env.walls,
                                 env.placed, env.active)
        sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
        obstacle[sc] = False
        obstacle[gc] = False
        dist, reach = geodesic_field(obstacle, gc)
        if not reach[sc]:
            return None
        field = np.where(np.isfinite(dist), dist / (2.0 * G), 1.0)
        return field_path_escape(field, start_xy, goal_xy, obstacle,
                                 radius=self.radius)


class EuclidWaypoints:
    """Parameter-free control: field = Euclidean distance to the goal cell,
    the monotone potential a wall-blind pointwise model can learn. Tests
    whether any goal-directed field + the same readout suffices."""

    def __init__(self, extract="escape", radius=24):
        self.extract = extract
        self.radius = radius
        print(f"plan field euclid extract={extract} radius={radius}")

    def __call__(self, start_xy, goal_xy, env):
        obstacle = obstacle_grid(env.objs, goal_xy, env.walls,
                                 env.placed, env.active)
        sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
        obstacle[sc] = False
        obstacle[gc] = False
        ext = make_ext(obstacle, gc)
        field = np.hypot(ext[1], ext[2])
        if self.extract == "cost":
            return field_path_cost(field, start_xy, goal_xy, obstacle)
        if self.extract == "escape":
            return field_path_escape(field, start_xy, goal_xy, obstacle,
                                     radius=self.radius)
        return field_path(field, start_xy, goal_xy, obstacle)


class Executor:
    """Fixed low-level skill executor. One visit = grasp or place; every visit
    ends at a hover (NEUTRAL_Z) above where it ended, grip = attachment."""

    def __init__(self, env, state, max_steps, route=False, planner=None):
        self.env = env
        self.state = state
        self.n = 0
        self.max_steps = max_steps
        self.route = route
        self.planner = planner

    def drive_route(self, goal_xy, z, grip, steps):
        """Plan-field waypoints (NCA) or Dijkstra geodesic routing instead of
        a straight line."""
        path = None
        if self.planner is not None:
            path = self.planner(self.env.state[:2], goal_xy, self.env)
        elif self.route:
            # The goal object itself must not be an obstacle (the grasp
            # target is the goal point); nor anything at the start hover.
            others = [self.env.objs[j] for j in range(len(self.env.objs))
                      if not self.env.placed[j] and j != self.env.active
                      and np.linalg.norm(self.env.objs[j] - goal_xy) > 0.05]
            path = route_xy(self.env.state[:2], goal_xy, self.env.walls, others)
        dbg = bool(os.environ.get("EXEC_DEBUG"))
        if dbg:
            print(f"    route to ({goal_xy[0]:.2f},{goal_xy[1]:.2f}) "
                  f"pts={None if path is None else len(path)} "
                  f"wh={self.env.wall_hits}", flush=True)
        if path is None or len(path) < 2:
            self.drive(goal_xy, z, grip, 30)
            if dbg:
                print(f"      straight fallback wh={self.env.wall_hits} "
                      f"err={np.linalg.norm(self.env.state[:2]-goal_xy):.3f}",
                      flush=True)
            return
        wps = _polyline_traj(np.asarray(path, np.float32), steps,
                             self.env.rng)
        for wp in wps:
            self.drive(wp, z, grip, 1)
        if dbg:
            print(f"      done wh={self.env.wall_hits} "
                  f"err={np.linalg.norm(self.env.state[:2]-goal_xy):.3f}",
                  flush=True)

    def drive(self, xy, z, grip, steps):
        p = _pose(xy, z, grip)
        for _ in range(steps):
            if self.n >= self.max_steps:
                return
            self.state, _ = self.env.step(p)
            self.n += 1

    def grasp(self, xy):
        self.drive_route(xy, GRASP_Z, 1.0, 70)
        self.drive(xy, CONTACT_Z, 0.0, 25)
        self.drive(xy, CONTACT_Z, 0.0, 15)
        ok = bool(self.env.attached)
        if ok:
            self.drive(self.state[:2], NEUTRAL_Z, 0.0, 30)
        else:
            self.drive(self.state[:2], CONTACT_Z, 1.0, 12)
            self.drive(self.state[:2], NEUTRAL_Z, 1.0, 30)
        return ok

    def place(self, xy):
        self.drive_route(xy, LIFT_Z, 0.0, 110)
        self.drive(xy, CONTACT_Z, 0.0, 25)
        n0 = self.env.n_placed()
        self.drive(xy, CONTACT_Z, 1.0, 15)
        ok = self.env.n_placed() > n0
        self.drive(self.state[:2], NEUTRAL_Z, 1.0, 30)
        return ok


def oracle_cmd(env, state):
    if env.attached:
        t = env.target_of(env.active)
        return np.array([t[0], t[1], CONTACT_Z, 1.0], np.float32)
    d = np.linalg.norm(env.objs - state[:2], axis=1)
    d[env.placed] = 1e9
    o = env.objs[int(d.argmin())]
    return np.array([o[0], o[1], CONTACT_Z, 0.0], np.float32)


def check_cmd(env, state, cmd):
    """(intent_ok, lands_on_valid_goal, matches_teacher_choice, xy_err)."""
    att = env.attached
    intent_ok = (cmd[3] < 0.5) == (not att)
    xy = cmd[:2]
    if not att:
        d = np.linalg.norm(env.objs - xy, axis=1)
        d = np.where(env.placed, 1e9, d)
        d0 = np.linalg.norm(env.objs - state[:2], axis=1)
        d0 = np.where(env.placed, 1e9, d0)
        teacher = np.linalg.norm(env.objs[int(d0.argmin())] - xy)
        return intent_ok, bool(d.min() < CONTACT_R), bool(teacher < CONTACT_R), \
            float(d.min())
    d = np.linalg.norm(env.targets - xy, axis=1)
    t = env.target_of(env.active)
    return intent_ok, bool(d.min() < PLACE_TOL), \
        bool(np.linalg.norm(t - xy) < PLACE_TOL), float(d.min())


def snap_cmd(env, cmd):
    """Low-level visual-servo stand-in: snap the coarse plan point to the
    nearest unplaced object (grasp) or the nearest target (place). Precision
    comes from the snap; the VLA must still choose WHICH object / WHICH color
    target — a place snapped to the wrong target is not placed."""
    if cmd[3] < 0.5:
        d = np.linalg.norm(env.objs - cmd[:2], axis=1)
        d = np.where(env.placed, 1e9, d)
        o = env.objs[int(d.argmin())]
        return np.array([o[0], o[1], CONTACT_Z, 0.0], np.float32)
    if env.attached:
        d = np.linalg.norm(env.targets - cmd[:2], axis=1)
        t = env.targets[int(d.argmin())]
        return np.array([t[0], t[1], CONTACT_Z, 1.0], np.float32)
    return cmd


def run_episode_visits(vla, policy_fn, objs, tgts, perm, walls, rng, max_steps,
                       stats=None, snap=False, route=False, planner=None):
    env = MultiEnv(rng, objs, tgts, perm, walls)
    state = env.reset()
    if vla is not None:
        vla.reset()
    ex = Executor(env, state, max_steps, route=route, planner=planner)
    while ex.n < max_steps and not env.success():
        cmd = (policy_fn(env, ex.state) if policy_fn
               else vla.predict(ex.state, env, tgts, perm, walls))
        if stats is not None:
            io, val, teach, err = check_cmd(env, ex.state, cmd)
            stats["n"] += 1
            stats["intent"] += int(io)
            stats["valid"] += int(val)
            stats["teacher"] += int(teach)
            stats["xyerr"].append(err)
        if snap:
            cmd = snap_cmd(env, cmd)
        if cmd[3] < 0.5:
            ex.grasp(cmd[:2])
        else:
            ex.place(cmd[:2])
    return (env.success(), env.n_placed(), int(env.bumps.sum()),
            env.wall_hits, ex.n)


def run_episode(vla, objs, tgts, perm, walls, rng, max_steps):
    env = MultiEnv(rng, objs, tgts, perm, walls)
    state = env.reset()
    vla.reset()
    n = 0
    for n in range(max_steps):
        tgt = vla.step(state, env, tgts, perm, walls)          # [x,y,z,open]
        target_pose = np.concatenate([tgt[:3], state[3:6], [tgt[3]]]).astype(np.float32)
        state, _ = env.step(target_pose)
        if env.success():
            return True, env.n_placed(), int(env.bumps.sum()), env.wall_hits, n + 1
    return False, env.n_placed(), int(env.bumps.sum()), env.wall_hits, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpts/smolvla_multi/best.pt")
    ap.add_argument("--norm-ckpt", type=str, default="",
                    help="optional ckpt carrying norm stats if not in --ckpt")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--zero-state", action="store_true",
                    help="feed zero proprio state (must match training)")
    ap.add_argument("--subgoal", action="store_true",
                    help="ckpt predicts skill subgoals (hold each for "
                         "--requery steps) instead of a next-pose chunk")
    ap.add_argument("--requery", type=int, default=8)
    ap.add_argument("--visit", action="store_true",
                    help="query once per visit from a hover; ckpt predicts the "
                         "visit subgoal [gx,gy,CONTACT_Z,open]")
    ap.add_argument("--oracle", action="store_true",
                    help="visit harness with oracle (nearest-first) commands")
    ap.add_argument("--snap", action="store_true",
                    help="snap predicted plan points to the nearest object / "
                         "carried object's target (visual-servo stand-in)")
    ap.add_argument("--two-walls", action="store_true",
                    help="two perpendicular gapped walls (route matters)")
    ap.add_argument("--route", action="store_true",
                    help="Dijkstra geodesic routing in the executor (oracle "
                         "ceiling for the plan field)")
    ap.add_argument("--plan-ckpt", type=str, default="",
                    help="planned-field checkpoint; its waypoints replace the "
                         "straight line (treatment arm)")
    ap.add_argument("--plan-extract", choices=["greedy", "escape", "cost"],
                    default="greedy",
                    help="waypoint readout: strict value descent, descent with "
                         "basin escape, or Dijkstra over predicted cost-to-go")
    ap.add_argument("--plan-radius", type=int, default=24,
                    help="basin-escape BFS radius in cells")
    args = ap.parse_args()

    MULTI_TASK = ("Pick up each object and place it on the matching colored "
                  "target, going around the walls.\n")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vla = None
    policy_fn = None
    if args.oracle:
        policy_fn = oracle_cmd
        print("oracle visit harness")
    else:
        from train_smolvla import build_policy
        policy, lang_ids, lang_mask = build_policy(device, MULTI_TASK)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ck["state"], strict=False)
        policy.eval()
        norm = ck.get("norm")
        if norm is None and args.norm_ckpt:
            norm = torch.load(args.norm_ckpt, map_location="cpu",
                              weights_only=False).get("norm")
        print(f"SmolVLA-0.5B (multi) best @step {ck.get('step')} "
              f"norm={'yes' if norm else 'no'}")
        cls = (VisitController if args.visit else
               SubgoalController if args.subgoal else MultiChunkController)
        vla = cls(policy, lang_ids, lang_mask, norm, device,
                  args.chunk_size, zero_state=args.zero_state,
                  **({"requery": args.requery} if args.subgoal else {}))

    if args.plan_ckpt == "gt":
        planner = GTWaypoints(args.plan_radius)
    elif args.plan_ckpt == "euclid":
        planner = EuclidWaypoints(args.plan_extract, args.plan_radius)
    else:
        planner = (NCAWaypoints(args.plan_ckpt, device, args.plan_extract,
                                args.plan_radius)
                   if args.plan_ckpt else None)
    rng = np.random.RandomState(args.seed)
    scenes = [sample_scene(rng, two_walls=args.two_walls)
              for _ in range(args.n_episodes)]
    ok = 0; placed = []; bumps = []; wh = []; steps = []
    stats = {"n": 0, "intent": 0, "valid": 0, "teacher": 0, "xyerr": []}
    for ep, (objs, tgts, perm, walls) in enumerate(scenes):
        if args.visit or args.oracle:
            s, p, b, w, st = run_episode_visits(
                vla, policy_fn, objs, tgts, perm, walls, rng, args.max_steps,
                stats=stats, snap=args.snap, route=args.route,
                planner=planner)
        else:
            s, p, b, w, st = run_episode(
                vla, objs, tgts, perm, walls, rng, args.max_steps)
        ok += int(s); placed.append(p); bumps.append(b); wh.append(w); steps.append(st)
        print(f"  ep{ep:2d} success={int(s)} placed={p}/3 bumps={b} wall_hits={w} steps={st}",
              flush=True)
    n = len(scenes)
    print(f"\n── multi-object closed loop ({n} eps) ──")
    print(f"  full success {ok}/{n} ({ok/n*100:.0f}%)   "
          f"mean placed {np.mean(placed):.2f}/3")
    print(f"  mean bumps/ep {np.mean(bumps):.2f}  wall_hits/ep {np.mean(wh):.1f}  "
          f"steps/ep {np.mean(steps):.0f}")
    if stats["n"]:
        n = stats["n"]
        print(f"  decisions {n}: intent {stats['intent']/n*100:.0f}%  "
              f"valid-aim {stats['valid']/n*100:.0f}%  "
              f"teacher-choice {stats['teacher']/n*100:.0f}%  "
              f"median xy err {np.median(stats['xyerr'])*100:.1f}cm")


if __name__ == "__main__":
    main()
