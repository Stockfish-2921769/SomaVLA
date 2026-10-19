#!/usr/bin/env python3
"""Low-level dataset for E0/E1: GT-field teacher, absolute target commands.

Every sample is a chunk-aligned observation for SmolVLA's action expert:

  image    [3,256,256] uint8
  state    [7]          raw EEF pose at the chunk start
  subgoal  [4]          normalized visit command [gx,gy,gz,open]
  action   [8,7]        raw absolute plant targets (target command, not delta)
  field    [4,64,64]    [obstacle, GT plan, dx/G, dy/G] float16
  plan_wrong            GT plan for a random free goal at similar distance
  plan_random           smooth low-frequency random field

The teacher is method 1's GT-field controller: oracle (nearest-first) visits,
8-connected Dijkstra geodesic to the visit goal, escape-readout path, pure
pursuit on that path, then the scripted grasp/place descent.  The action label
is the absolute pose command issued to the damped plant:

    target_pose = [waypoint_x, waypoint_y, waypoint_z, rot, grip]
    state_next  = state + 0.5 * (target_pose - state) + noise

Actions are logged per plant step; chunks are H=8 windows at stride 4 that do
not cross visit boundaries.  The last terminal action is repeated to fill the
window tail so gripper-close/open steps are represented.

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/multi_lowlevel_data.py \
      --n-episodes 600 --out data/multi_lowlevel_2w.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import ndimage

from multi_task import (sample_scene, render_scene, MultiEnv, G, N_OBJ,
                        NEUTRAL_Z, GRASP_Z, CONTACT_Z, LIFT_Z,
                        xy_to_cell, cell_to_xy, _pt_wall_dist, route_xy)
from multi_plan_data import obstacle_grid, make_ext, field_path_escape
from spatial_plan_data import geodesic_field


H = 8            # action chunk
STRIDE = 4       # receding-horizon stride


# ── fields ────────────────────────────────────────────────────────────
def gt_plan(obstacle, goal_cell):
    dist, reach = geodesic_field(obstacle, goal_cell)
    return np.where(np.isfinite(dist), dist / (2.0 * G), 1.0).astype(np.float32)


def make_field(objs, goal_xy, walls, placed, active):
    """[4,G,G] = [obstacle, GT plan, dx/G, dy/G]."""
    obstacle = obstacle_grid(objs, goal_xy, walls, placed, active)
    gc = xy_to_cell(goal_xy)
    obstacle[gc] = False
    ext = make_ext(obstacle, gc)
    plan = gt_plan(obstacle, gc)
    return np.stack([ext[0], plan, ext[1], ext[2]]).astype(np.float32), obstacle


def wrong_goal_plan(obstacle, start_xy, goal_xy, rng):
    """GT plan to a random free cell at a similar start distance."""
    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    obstacle = obstacle.copy()
    obstacle[sc] = False
    obstacle[gc] = False
    cand = np.argwhere(~obstacle)
    d0 = float(np.hypot(sc[0] - gc[0], sc[1] - gc[1]))
    if len(cand) == 0:
        return np.zeros((G, G), np.float32)
    d = np.hypot(cand[:, 0] - sc[0], cand[:, 1] - sc[1])
    m = (d >= 0.7 * d0) & (d <= 1.3 * d0) & (d > 2.0)
    pool = cand[m] if m.any() else cand
    for _ in range(30):
        pick = tuple(pool[rng.randint(len(pool))])
        dist, reach = geodesic_field(obstacle, pick)
        if reach[sc]:
            return np.where(np.isfinite(dist), dist / (2.0 * G),
                            1.0).astype(np.float32)
    return np.zeros((G, G), np.float32)


def random_plan(rng, sigma=3.0):
    z = ndimage.gaussian_filter(rng.randn(G, G), sigma=sigma)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return z.astype(np.float32)


# ── path following ────────────────────────────────────────────────────
def _resample(path, ds=0.005):
    path = np.asarray(path, np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return path[[0]], 0.0
    n = max(int(total / ds) + 2, 2)
    d = np.linspace(0.0, total, n)
    out = np.empty((n, 2), np.float32)
    for k in range(2):
        out[:, k] = np.interp(d, cum, path[:, k])
    return out, total


def _project(pts, xy, start_idx):
    d = np.linalg.norm(pts[start_idx:] - xy[None, :], axis=1)
    return start_idx + int(np.argmin(d))


# ── teacher ───────────────────────────────────────────────────────────
class GTFieldTeacher:
    """Executes the method-1 GT-field controller and logs per-step commands."""

    def __init__(self, env, walls, rng, max_steps=1500, lookahead=0.045,
                 tol=0.012):
        self.env = env
        self.walls = walls
        self.rng = rng
        self.max_steps = max_steps
        self.lookahead = lookahead
        self.tol = tol
        self.states, self.actions = [], []
        self.objs_hist, self.placed_hist, self.active_hist = [], [], []
        self.final_state = env.state.copy()

    @property
    def n(self):
        return len(self.actions)

    def _pose(self, xy, z, grip):
        return np.array([xy[0], xy[1], z, 0.0, 0.0, 0.0, grip], np.float32)

    def _act(self, action):
        self.states.append(self.env.state.copy())
        self.actions.append(np.asarray(action, np.float32).copy())
        self.objs_hist.append(self.env.objs.copy())
        self.placed_hist.append(self.env.placed.copy())
        self.active_hist.append(int(self.env.active))
        self.env.step(action)
        self.final_state = self.env.state.copy()

    def _path(self, goal_xy):
        start = self.env.state[:2]
        obstacle = obstacle_grid(self.env.objs, goal_xy, self.walls,
                                 self.env.placed, self.env.active)
        sc, gc = xy_to_cell(start), xy_to_cell(goal_xy)
        obstacle[sc] = False
        obstacle[gc] = False
        dist, reach = geodesic_field(obstacle, gc)
        if not reach[sc]:
            return None
        field = np.where(np.isfinite(dist), dist / (2.0 * G), 1.0)
        return field_path_escape(field, start, goal_xy, obstacle, radius=24)

    def follow(self, goal_xy, z, grip, max_steps=140):
        path = self._path(goal_xy)
        if path is None or len(path) < 2:
            path = np.array([self.env.state[:2], goal_xy], np.float32)
        pts, total = _resample(path)
        step_d = 0.005
        idx = 0
        for _ in range(max_steps):
            if self.n >= self.max_steps:
                break
            if np.linalg.norm(self.env.state[:2] - goal_xy) < self.tol:
                break
            j = min(idx + max(int(self.lookahead / step_d), 1), len(pts) - 1)
            self._act(self._pose(pts[j], z, grip))
            idx = _project(pts, self.env.state[:2], max(idx - 2, 0))
            if idx >= len(pts) - 1 and \
                    np.linalg.norm(self.env.state[:2] - goal_xy) < 2 * self.tol:
                break

    def descend(self, z, grip, steps=8):
        xy = self.env.state[:2].copy()
        z0 = float(self.env.state[2])
        for k in range(1, steps + 1):
            zz = z0 + (z - z0) * k / steps
            self._act(self._pose(xy, zz, grip))

    def close_grip(self, steps=12):
        xy = self.env.state[:2].copy()
        for _ in range(steps):
            self._act(self._pose(xy, CONTACT_Z, 0.0))
            if self.env.attached:
                return True
        return False

    def open_grip(self, steps=12):
        xy = self.env.state[:2].copy()
        n0 = self.env.n_placed()
        for _ in range(steps):
            self._act(self._pose(xy, CONTACT_Z, 1.0))
            if self.env.n_placed() > n0:
                return True
        return False

    def lift(self, steps=10):
        xy = self.env.state[:2].copy()
        z0 = float(self.env.state[2])
        grip = float(self.env.state[6])
        for k in range(1, steps + 1):
            zz = z0 + (NEUTRAL_Z - z0) * k / steps
            self._act(self._pose(xy, zz, grip))


def oracle_cmd(env, state):
    """Same nearest-first decision as eval_multi (teacher, not oracle routing)."""
    if env.attached:
        t = env.target_of(env.active)
        return np.array([t[0], t[1], CONTACT_Z, 1.0], np.float32)
    d = np.linalg.norm(env.objs - state[:2], axis=1)
    d[env.placed] = 1e9
    o = env.objs[int(d.argmin())]
    return np.array([o[0], o[1], CONTACT_Z, 0.0], np.float32)


def _segment_crosses(a, b, walls, points=48, clearance=0.03):
    for t in np.linspace(0.0, 1.0, points):
        p = a + (b - a) * t
        if _pt_wall_dist(p, walls) < clearance:
            return True
    return False


def _crossing_objects(objs, tgts, perm, walls):
    return [i for i in range(len(objs))
            if _segment_crosses(objs[i], tgts[perm[i]], walls)]


def _pair_detour(objs, tgts, perm, walls, i):
    others = [objs[j] for j in range(len(objs)) if j != i]
    path = route_xy(objs[i], tgts[perm[i]], walls, others)
    if path is None or len(path) < 2:
        return 0.0
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    euclid = float(np.linalg.norm(tgts[perm[i]] - objs[i]))
    return length / max(euclid, 1e-6)


def _best_crossing(objs, tgts, perm, walls):
    best = None
    for i in _crossing_objects(objs, tgts, perm, walls):
        r = _pair_detour(objs, tgts, perm, walls, i)
        if best is None or r > best[0]:
            best = (r, i)
    return best


def choose_obj(env, state, pair_ratio, detour_mode, min_ratio):
    if detour_mode and not env.attached:
        cand = [i for i in range(env.n)
                if not env.placed[i] and pair_ratio[i] >= min_ratio]
        if cand:
            return max(cand, key=lambda i: pair_ratio[i])
    d = np.linalg.norm(env.objs - state[:2], axis=1)
    d[env.placed] = 1e9
    return int(d.argmin())


def run_visit(teacher, env, state, pair_ratio, detour_mode, min_ratio):
    """One grasp or place visit; returns (cmd, segment_start, ok)."""
    if env.attached:
        t = env.target_of(env.active)
        cmd = np.array([t[0], t[1], CONTACT_Z, 1.0], np.float32)
    else:
        i = choose_obj(env, state, pair_ratio, detour_mode, min_ratio)
        o = env.objs[i]
        cmd = np.array([o[0], o[1], CONTACT_Z, 0.0], np.float32)
    t0 = teacher.n
    if cmd[3] < 0.5:
        teacher.follow(cmd[:2], GRASP_Z, 1.0)
        teacher.descend(CONTACT_Z, 1.0, 8)
        ok = teacher.close_grip(12)
    else:
        teacher.follow(cmd[:2], LIFT_Z, 0.0)
        teacher.descend(CONTACT_Z, 0.0, 8)
        ok = teacher.open_grip(12)
    return cmd, t0, teacher.n, ok


# ── dataset packing ───────────────────────────────────────────────────
def pack_segment(states, actions, objs_hist, placed_hist, active_hist,
                 cmd, tgts, perm, walls, scene_id, ep_id, visit_id,
                 rng, out):
    """Window starts inside one visit; fields/images rendered only there."""
    T = len(actions)
    if T == 0:
        return
    aug = list(actions) + [actions[-1]] * (H - 1)
    for t in range(0, T, STRIDE):
        st = states[t]
        field, obstacle = make_field(objs_hist[t], cmd[:2], walls,
                                     placed_hist[t], active_hist[t])
        img = render_scene(objs_hist[t], tgts, perm, placed_hist[t], walls,
                           st[:2])
        sc, gc = xy_to_cell(st[:2]), xy_to_cell(cmd[:2])
        plan_start = float(field[1][sc])
        euclid_cells = float(np.hypot(gc[1] - sc[1], gc[0] - sc[0]))
        detour = plan_start * 2.0 * G / max(euclid_cells, 1e-6)
        out["images"].append(np.asarray(img, np.uint8).transpose(2, 0, 1))
        out["states"].append(st.astype(np.float32))
        out["subgoals"].append(cmd.astype(np.float32))
        out["actions"].append(np.stack(aug[t:t + H]).astype(np.float32))
        out["fields"].append(field.astype(np.float16))
        out["plans_wrong"].append(
            wrong_goal_plan(obstacle, st[:2], cmd[:2], rng).astype(np.float16))
        out["plans_random"].append(random_plan(rng).astype(np.float16))
        out["metas"].append(np.array(
            [scene_id, ep_id, t, visit_id, sc[0], sc[1], gc[0], gc[1],
             int(active_hist[t] >= 0), int(cmd[3] >= 0.5)], np.float32))
        out["detour"].append(detour)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--out", type=str, default="data/multi_lowlevel_2w.npz")
    ap.add_argument("--high-ckpt", type=str,
                    default="ckpts/visit_2w_base/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-checks", type=int, default=5)
    ap.add_argument("--detour-frac", type=float, default=0.7,
                    help="fraction of episodes resampled for a long wall detour")
    ap.add_argument("--detour-min-ratio", type=float, default=1.45,
                    help="required routed/euclid length ratio of the forced pair")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="stop after this many chunk samples (0 = no cap)")
    args = ap.parse_args()

    import torch
    ck = torch.load(args.high_ckpt, map_location="cpu", weights_only=False)
    ch_mean = np.asarray(ck["norm"]["ch_mean"], np.float32)
    ch_std = np.asarray(ck["norm"]["ch_std"], np.float32)
    print(f"subgoal norm from {args.high_ckpt}: mean {np.round(ch_mean,3)} "
          f"std {np.round(ch_std,3)}")

    rng = np.random.RandomState(args.seed)
    data_rng = np.random.RandomState(args.seed + 10007)
    out = {k: [] for k in ("images", "states", "subgoals", "actions",
                           "fields", "plans_wrong", "plans_random", "metas",
                           "detour")}
    replay_err = []
    n_fail = 0
    ep = 0
    while ep < args.n_episodes:
        objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
        detour_mode = bool(rng.rand() < args.detour_frac)
        pair_ratio = np.ones(N_OBJ, np.float32)
        if detour_mode:
            best = _best_crossing(objs, tgts, perm, walls)
            for _ in range(120):
                if best is not None and best[0] >= args.detour_min_ratio:
                    break
                objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
                best = _best_crossing(objs, tgts, perm, walls)
            pair_ratio = np.array(
                [_pair_detour(objs, tgts, perm, walls, i)
                 for i in range(N_OBJ)], np.float32)
        rng_state = rng.get_state()
        env = MultiEnv(rng, objs, tgts, perm, walls)
        state = env.reset()
        teacher = GTFieldTeacher(env, walls, rng, max_steps=args.max_steps)
        ok_all = True
        visit_id = 0
        while teacher.n < args.max_steps and not env.success():
            cmd, t0, t1, ok = run_visit(teacher, env, teacher.env.state,
                                        pair_ratio, detour_mode,
                                        args.detour_min_ratio)
            if ok:
                pack_segment(teacher.states[t0:t1], teacher.actions[t0:t1],
                             teacher.objs_hist[t0:t1],
                             teacher.placed_hist[t0:t1],
                             teacher.active_hist[t0:t1], cmd, tgts, perm,
                             walls, ep, ep, visit_id, data_rng, out)
            visit_id += 1
            if not ok:
                ok_all = False
                break
            teacher.lift(10)
            if teacher.n >= args.max_steps:
                break
        if not env.success():
            ok_all = False
            n_fail += 1

        if ep < args.replay_checks:
            rng.set_state(rng_state)
            env2 = MultiEnv(rng, objs, tgts, perm, walls)
            env2.reset()
            acts = teacher.actions
            for a in acts:
                env2.step(a)
            target = teacher.final_state
            ref = env2.state.copy()
            err = float(np.linalg.norm(ref[:3] - target[:3]))
            # wrong semantics: treat target commands as deltas
            rng.set_state(rng_state)
            env3 = MultiEnv(rng, objs, tgts, perm, walls)
            env3.reset()
            prev = env3.state.copy()
            for a in acts:
                env3.step(prev + a)
                prev = env3.state.copy()
            err_delta = float(np.linalg.norm(env3.state[:3] - target[:3]))
            replay_err.append((err, err_delta))
            print(f"  replay ep{ep}: target-command err {err*100:.3f} cm | "
                  f"delta-interpretation err {err_delta*100:.1f} cm",
                  flush=True)

        ep += 1
        ns = len(out["states"])
        if ep % 10 == 0 or ep <= 2:
            print(f"  ep {ep}/{args.n_episodes} samples {ns} "
                  f"steps {teacher.n} success_ep={int(env.success())} "
                  f"placements={env.n_placed()}/{N_OBJ}", flush=True)
        if args.max_samples and ns >= args.max_samples:
            print(f"  reached --max-samples {args.max_samples}", flush=True)
            break

    images = np.stack(out["images"]).astype(np.uint8)
    states = np.stack(out["states"]).astype(np.float32)
    subgoals = np.stack(out["subgoals"]).astype(np.float32)
    actions = np.stack(out["actions"]).astype(np.float32)
    fields = np.stack(out["fields"]).astype(np.float16)
    plans_wrong = np.stack(out["plans_wrong"]).astype(np.float16)
    plans_random = np.stack(out["plans_random"]).astype(np.float16)
    metas = np.stack(out["metas"]).astype(np.float32)
    detour = np.asarray(out["detour"], np.float32)

    subgoals = (subgoals - ch_mean) / ch_std
    st_mean = states.mean(0)
    st_std = np.maximum(states.std(0), 1e-3)
    act_mean = actions.reshape(-1, 7).mean(0)
    act_std = np.maximum(actions.reshape(-1, 7).std(0), 1e-3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, images=images, states=states, subgoals=subgoals,
        actions=actions, fields=fields, plans_wrong=plans_wrong,
        plans_random=plans_random, metas=metas, detour=detour,
        st_mean=st_mean, st_std=st_std, act_mean=act_mean, act_std=act_std,
        ch_mean=ch_mean, ch_std=ch_std,
        lowlevel_args=np.array([args.seed, ep, H, STRIDE], np.float32))

    print(f"wrote {args.out}: images {images.shape} actions {actions.shape} "
          f"fields {fields.shape} from {ep} episodes ({n_fail} failed)")
    print(f"  detour ratio: median {np.median(detour):.2f}  "
          f">1.25 {np.mean(detour > 1.25)*100:.0f}%  "
          f">1.5 {np.mean(detour > 1.5)*100:.0f}%")
    print(f"  st mean {np.round(st_mean,3)} std {np.round(st_std,3)}")
    print(f"  act mean {np.round(act_mean,3)} std {np.round(act_std,3)}")
    if replay_err:
        e = np.asarray(replay_err)
        print(f"  replay max target-command err {e[:,0].max()*100:.3f} cm "
              f"(delta interpretation {e[:,1].min()*100:.1f} cm)")


if __name__ == "__main__":
    main()
