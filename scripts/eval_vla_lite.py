#!/usr/bin/env python3
"""Closed-loop eval for a VLA-lite low-level checkpoint.

Mirrors the E0 vectorized harness (`eval_field_expert_vec.py`): the same
200 seed-777 scenes, per-scene plant RNG, lockstep batched episodes, chunk=8
executed for 4 plant steps, scripted lift, teacher (oracle) high-level.

Field diagnostics (plan MSE / obstacle IoU) are computed online for the
NCA/CNN arms against a 32x32 GT field built from the current occupancy.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_lowlevel_data import oracle_cmd, H, STRIDE
from multi_plan_data import obstacle_grid
from multi_task import (sample_scene, render_scene, MultiEnv, N_OBJ,
                        NEUTRAL_Z, CONTACT_R, XL, XH, YL, YH)
from spatial_plan_data import geodesic_field
from soma.vla_lite.high_level import HighLevelPolicy
from soma.vla_lite.language import LanguageSlot
from soma.vla_lite.policy import FIELD_OVERRIDES, VLALiteLowLevel
from soma.vla_lite.vision import FrozenSigLIP
from vla_lite_ood import sample_scene_multi_wall

G_OBS = 64                   # obstacle_grid native resolution


def xy_to_cell_g(xy, g):
    j = int((xy[0] - XL) / (XH - XL) * g)
    i = int((YH - xy[1]) / (YH - YL) * g)
    return (min(max(i, 0), g - 1), min(max(j, 0), g - 1))


def gt_field_g(env, goal_xy, g):
    """GxG obstacle + solvable plan (G in {32, 64}), same construction as
    vla_lite_data at G=32; plan is d/(2G) so G32/G64 share a value range."""
    obs64 = obstacle_grid(env.objs, goal_xy, env.walls, env.placed, env.active)
    # clear the goal cell at 64 before max-pooling, as the stored fields do
    obs64[xy_to_cell_g(goal_xy, G_OBS)] = False
    f = G_OBS // g
    obs = obs64.reshape(g, f, g, f).max(axis=(1, 3))
    gc = xy_to_cell_g(goal_xy, g)
    obs[gc] = False
    dist, reach = geodesic_field(obs, gc)
    plan = np.where(np.isfinite(dist), dist / (2.0 * g), 1.0)
    plan = plan.astype(np.float32)
    plan[obs] = 0.0
    return plan, obs.astype(np.float32)


def _rate(visits):
    k = sum(v["ok"] for v in visits)
    return f"{k}/{len(visits)} ({k / max(len(visits), 1) * 100:.1f}%)"


def visit_choice(env, xy):
    """Candidate a command points at: nearest unplaced object (pick visit)
    or nearest target (carry visit)."""
    if env.attached:
        d = np.linalg.norm(env.targets - xy, axis=1)
    else:
        d = np.linalg.norm(env.objs - xy, axis=1)
        d[env.placed] = 1e9
    return int(d.argmin())


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
        self.visits = 0
        self.visits_ok = 0
        self.query_i = 0
        self.plan_mse_sum = 0.0
        self.iou_sum = 0.0
        self.diag_n = 0
        self.cur_visit = None
        self.visit_log = []

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
        ok = self.visit_done()
        self.visits += 1
        self.visits_ok += int(ok)
        if self.cur_visit is not None:
            self.cur_visit.update(ok=bool(ok), steps=self.visit_steps,
                                  finished=True)
            self.visit_log.append(self.cur_visit)
            self.cur_visit = None
        self.finish_visit()

    def flush_visit(self):
        """Log a visit cut off by the episode step budget (not counted in
        visits/visits_ok, matching the existing visit totals)."""
        if self.cur_visit is not None:
            self.cur_visit.update(ok=False, steps=self.visit_steps,
                                  finished=False)
            self.visit_log.append(self.cur_visit)
            self.cur_visit = None

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


@torch.no_grad()
def predict_high(hl, lang, vision, eps, scenes, st_mean, st_std,
                 ch_mean, ch_std):
    vdev = next(vision.parameters()).device
    imgs = np.stack([render_scene(e.env.objs, scenes[e.scene_id][1],
                                  scenes[e.scene_id][2], e.env.placed,
                                  scenes[e.scene_id][3], e.state)
                     for e in eps])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(vdev)
    tok = vision(x)
    st = torch.from_numpy(np.stack([e.state for e in eps])).float().to(vdev)
    _, lt = lang(batch_size=len(eps), device=vdev)
    pred = hl(tok, (st - st_mean) / st_std, lt)
    p = pred.float().cpu().numpy()
    xyz = p[:, :3] * ch_std.cpu().numpy() + ch_mean.cpu().numpy()
    op = (p[:, 3] > 0).astype(np.float32)
    return np.concatenate([xyz, op[:, None]], axis=1)


def det_noise_for(eps, chunk, action_dim, device):
    rows = [np.random.RandomState(
        (e.scene_id * 1000003 + e.query_i * 9176 + 12345) & 0x7FFFFFFF)
        .standard_normal((chunk, action_dim)) for e in eps]
    return torch.from_numpy(np.stack(rows).astype(np.float32)).to(device)


@torch.no_grad()
def predict_chunks(pol, vision, eps, scenes, act_mean, act_std,
                   det_noise=True, diag=True, grid=32, k_steps=None,
                   field_override=None, hide_goal=False):
    imgs = np.stack([render_scene(e.env.objs, scenes[e.scene_id][1],
                                  scenes[e.scene_id][2], e.env.placed,
                                  scenes[e.scene_id][3], e.state)
                     for e in eps])
    vdev = next(vision.parameters()).device
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(vdev)
    tok = vision(x)
    dev = tok.device
    st = torch.from_numpy(np.stack([e.state for e in eps])).float().to(dev)
    sg = torch.from_numpy(np.stack([e.cmd for e in eps])).float().to(dev)

    has_plan = pol.field_source in {"nca", "cnn", "zero"}
    want_gt = pol.field_source == "gt" or field_override == "plan-gt" or (
        diag and has_plan)
    gt_plan = gt_obs = None
    if want_gt:
        fields = [gt_field_g(e.env, e.cmd[:2], grid) for e in eps]
        gt_plan = torch.from_numpy(np.stack([f[0] for f in fields])).to(dev)
        gt_obs = torch.from_numpy(np.stack([f[1] for f in fields])).to(dev)

    kw = {}
    if pol.field_source == "gt" or field_override == "plan-gt":
        kw = dict(gt_plan=gt_plan, gt_obstacle=gt_obs)
    noise = None
    if det_noise:
        noise = det_noise_for(eps, pol.decoder.chunk, pol.decoder.action_dim,
                              dev)
    out = pol(vis_tokens=tok, states=st, subgoals=sg, sample_steps=10,
              noise=noise, grid=grid, k_steps=k_steps,
              field_override=field_override, hide_goal=hide_goal, **kw)

    if want_gt and has_plan:
        pred_plan = out["plan"].float().cpu().numpy()
        p_obs = torch.sigmoid(out["obstacle_logits"][:, 0]).cpu().numpy()
        gp = gt_plan.cpu().numpy()
        go = gt_obs.cpu().numpy()
        for i, e in enumerate(eps):
            mask = go[i] < 0.5
            mse = float(((pred_plan[i] - gp[i]) ** 2 * mask).sum() /
                        max(mask.sum(), 1.0))
            pb, gb = p_obs[i] > 0.5, go[i] > 0.5
            iou = float((pb & gb).sum() / max((pb | gb).sum(), 1))
            e.plan_mse_sum += mse
            e.iou_sum += iou
            e.diag_n += 1

    for e in eps:
        e.query_i += 1
    a = out["action"].cpu().numpy()
    return a * act_std + act_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--high-ckpt", default="",
                    help="trained high-level ckpt; empty = teacher subgoal")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scene-type",
                    choices=["two-walls", "random", "multi-wall"],
                    default="two-walls",
                    help="multi-wall = O2 OOD: two-wall base + extra "
                         "segments (--n-extra-walls)")
    ap.add_argument("--n-extra-walls", type=int, nargs=2, default=[2, 6],
                    metavar=("LO", "HI"))
    ap.add_argument("--grid", type=int, default=0,
                    help="eval grid (O1: 64); 0 = training grid")
    ap.add_argument("--k-steps", type=int, default=0,
                    help="NCA inference K; 0 = ckpt K scaled by grid/train")
    ap.add_argument("--no-k-scale", action="store_true",
                    help="keep ckpt K when --grid differs from training")
    ap.add_argument("--field-override", choices=FIELD_OVERRIDES,
                    default=None,
                    help="inference probe: zero = hide field; plan-* = "
                         "replace only the plan channel")
    ap.add_argument("--hide-goal", action="store_true",
                    help="inference probe: decoder does not see subgoal xy "
                         "(field generator still does)")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--visit-cap", type=int, default=180)
    ap.add_argument("--exec-steps", type=int, default=STRIDE)
    ap.add_argument("--batch-episodes", type=int, default=16)
    ap.add_argument("--stochastic-noise", action="store_true")
    ap.add_argument("--subgoal-noise", type=float, default=0.0,
                    help="Gaussian xy noise (m) added to every subgoal; "
                         "keyed by scene and visit ordinal so arms share it")
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    arm = ca.get("arm", "nca")
    train_grid = ca.get("grid", 32)
    grid = args.grid or train_grid
    readout = ca.get("readout", "full")
    # pilot-era CNN ckpts emit the plan scalar only (cnn.out: 1 channel)
    cnn_w = ck["model"].get("cnn.out.weight")
    cnn_channels = cnn_w.shape[0] if cnn_w is not None else 32
    pol = VLALiteLowLevel(field_source=arm, grid=grid, chunk=8, action_dim=7,
                          readout_mode=readout,
                          cnn_layers=ca.get("cnn_layers", 12),
                          cnn_channels=cnn_channels)
    norm = ck["norm"]
    pol.set_norm(norm["st_mean"], norm["st_std"],
                 norm["ch_mean"], norm["ch_std"])
    # pre-random-K ckpts only stored k_steps (0 = NCA default 12)
    k_train = ca.get("k_infer") or ca.get("k_steps") or pol.nca.k_steps
    if args.k_steps:
        k_eval = args.k_steps
    elif args.no_k_scale:
        k_eval = k_train
    else:                    # same physical propagation range at any G
        k_eval = int(round(k_train * grid / train_grid))
    pol.nca.k_steps = k_eval
    pol.load_state_dict(ck["model"])
    pol.eval().to(device)
    vision = FrozenSigLIP(device=device).to(device).eval()
    act_mean = np.asarray(ca["act_mean"], np.float32)
    act_std = np.asarray(ca["act_std"], np.float32)

    hl = lang = None
    h_st_mean = h_st_std = h_ch_mean = h_ch_std = None
    if args.high_ckpt:
        hk = torch.load(args.high_ckpt, map_location="cpu",
                        weights_only=False)
        hl = HighLevelPolicy(use_field=False).to(device)
        lang = LanguageSlot().to(device)
        hl.load_state_dict(hk["model"])
        lang.load_state_dict(hk["language"])
        hl.eval()
        lang.eval()
        h_st_mean = torch.as_tensor(hk["norm"]["st_mean"],
                                    dtype=torch.float32, device=device)
        h_st_std = torch.as_tensor(hk["norm"]["st_std"],
                                   dtype=torch.float32, device=device)
        h_ch_mean = torch.as_tensor(hk["norm"]["ch_mean"][:3],
                                    dtype=torch.float32, device=device)
        h_ch_std = torch.as_tensor(hk["norm"]["ch_std"][:3],
                                   dtype=torch.float32, device=device)

    rng = np.random.RandomState(args.seed)
    if args.scene_type == "multi-wall":
        scenes = [sample_scene_multi_wall(rng, tuple(args.n_extra_walls))
                  for _ in range(args.n_episodes)]
    else:
        scenes = [sample_scene(rng,
                               two_walls=(args.scene_type == "two-walls"))
                  for _ in range(args.n_episodes)]
    ep_rngs = [np.random.RandomState(args.seed * 7919 + i)
               for i in range(args.n_episodes)]
    print(f"vla-lite eval: {args.n_episodes} eps, batch {args.batch_episodes},"
          f" arm {arm}, readout {readout}, grid {grid} (train {train_grid}),"
          f" K {k_eval}, override {args.field_override}, "
          f"scenes {args.scene_type}, exec {args.exec_steps}, "
          f"{'stochastic' if args.stochastic_noise else 'det'}-noise, "
          f"ckpt {args.ckpt}, high "
          f"{'teacher' if hl is None else args.high_ckpt}", flush=True)

    K = min(args.batch_episodes, args.n_episodes)
    active = {}
    results = {}
    next_id = 0

    def spawn(i):
        objs, tgts, perm, walls = scenes[i]
        return Ep(i, objs, tgts, perm, walls, ep_rngs[i], args.max_steps,
                  args.visit_cap, args.exec_steps)

    for _ in range(K):
        active[next_id] = spawn(next_id)
        next_id += 1
    torch.manual_seed(args.seed * 100003)

    tick = 0
    hi_n = 0
    hi_l1 = 0.0
    hi_open_ok = 0
    while active:
        tick += 1
        eps = list(active.values())
        hi = [e for e in eps if e.need_high and not e.done]
        cmds = []
        ocs = [oracle_cmd(e.env, e.state) for e in hi]
        if hi:
            if hl is None:
                cmds = [oc.copy() for oc in ocs]
            else:
                cmds = predict_high(hl, lang, vision, hi, scenes,
                                    h_st_mean, h_st_std, h_ch_mean, h_ch_std)
                for c, oc in zip(cmds, ocs):
                    hi_n += 1
                    hi_l1 += float(np.abs(c[:3] - oc[:3]).mean())
                    hi_open_ok += int((c[3] > 0.5) == (oc[3] > 0.5))
        for e, c, oc in zip(hi, cmds, ocs):
            if args.subgoal_noise > 0:
                r = np.random.RandomState(
                    (e.scene_id * 1000003 + e.visits * 9176 + 54321)
                    & 0x7FFFFFFF)
                c[:2] = c[:2] + r.standard_normal(2) * args.subgoal_noise
            e.cmd = c.astype(np.float32)
            # per-visit record: was the command already wrong at issue time?
            e.cur_visit = dict(
                visit=e.visits, placed0=int(e.env.n_placed()),
                attached=bool(e.env.attached),
                cmd=[float(v) for v in e.cmd],
                oracle=[float(v) for v in oc],
                xy_err=float(np.linalg.norm(e.cmd[:2] - oc[:2])),
                open_ok=bool((e.cmd[3] > 0.5) == (oc[3] > 0.5)),
                same_choice=(visit_choice(e.env, e.cmd[:2]) ==
                             visit_choice(e.env, oc[:2])))
            e.placed0 = e.env.n_placed()
            e.visit_steps = 0
            e.need_high = False
            e.in_visit = True
            e.chunk = None
            e.chunk_i = 0
            if e.visit_done():
                e.end_visit()
        lo = [e for e in eps if e.in_visit and e.chunk is None
              and e.lift == 0 and not e.done]
        if lo:
            chunks = predict_chunks(
                pol, vision, lo, scenes, act_mean, act_std,
                det_noise=not args.stochastic_noise, diag=not args.no_diag,
                grid=grid, field_override=args.field_override,
                hide_goal=args.hide_goal)
            for e, c in zip(lo, chunks):
                e.chunk = c
                e.chunk_i = 0
        for e in eps:
            if e.done:
                continue
            e.state, _ = e.env.step(e.action())
            e.steps += 1
            if e.in_visit:
                e.visit_steps += 1
                if e.visit_done() or e.visit_steps >= args.visit_cap:
                    e.end_visit()
            if e.lift == 0 and not e.in_visit:
                e.need_high = True
            if e.env.success() or e.steps >= args.max_steps:
                e.done = True
        for sid in [k for k, e in active.items() if e.done]:
            e = active.pop(sid)
            results[sid] = e
            if next_id < args.n_episodes:
                active[next_id] = spawn(next_id)
                next_id += 1
        if tick % 500 == 0:
            print(f"  tick {tick} active {len(active)} done "
                  f"{len(results)}/{args.n_episodes}", flush=True)

    succ, placed, bumps, wh, steps, mses, ious = [], [], [], [], [], [], []
    visits_all = visits_ok = 0
    lines = []
    for i in range(args.n_episodes):
        e = results[i]
        e.flush_visit()
        s = bool(e.env.success())
        succ.append(s)
        placed.append(e.env.n_placed())
        bumps.append(int(e.env.bumps.sum()))
        wh.append(e.env.wall_hits)
        steps.append(e.steps)
        visits_all += e.visits
        visits_ok += e.visits_ok
        mse = e.plan_mse_sum / e.diag_n if e.diag_n else float("nan")
        iou = e.iou_sum / e.diag_n if e.diag_n else float("nan")
        mses.append(mse)
        ious.append(iou)
        line = (f"  ep{i:3d} success={int(s)} placed={e.env.n_placed()}/{N_OBJ}"
                f" bumps={int(e.env.bumps.sum())} wall_hits={e.env.wall_hits}"
                f" steps={e.steps} visits={e.visits_ok}/{e.visits}")
        if e.diag_n:
            line += f" plan_mse={mse:.4f} iou={iou:.3f}"
        print(line, flush=True)
        lines.append(line)
    n = len(results)
    ok = int(np.sum(succ))
    summary = [
        f"\n── VLA-lite closed loop ({n} eps, seed {args.seed}, arm {arm}) ──",
        f"  ckpt {args.ckpt}",
        f"  scenes {args.scene_type}  grid {grid} (train {train_grid})  "
        f"K {k_eval}  readout {readout}  override {args.field_override}",
        f"  full success {ok}/{n} ({ok / n * 100:.1f}%)   "
        f"mean placed {np.mean(placed):.2f}/{N_OBJ}",
        f"  mean bumps/ep {np.mean(bumps):.2f}  "
        f"wall_hits/ep {np.mean(wh):.1f}  steps/ep {np.mean(steps):.0f}",
        f"  visits ok {visits_ok}/{visits_all} "
        f"({visits_ok / max(visits_all, 1) * 100:.1f}%)",
    ]
    if np.isfinite(mses).any():
        summary.append(
            f"  plan MSE {np.nanmean(mses):.4f}  obstacle IoU "
            f"{np.nanmean(ious):.3f}")
    if hi_n:
        summary.append(
            f"  high-level vs teacher: xyz L1 {hi_l1 / hi_n:.4f} (raw)  "
            f"open acc {hi_open_ok / hi_n:.3f}  (n={hi_n})")
    vlog = [v for i in range(n) for v in results[i].visit_log
            if v["finished"]]
    if vlog and (hl is not None or args.subgoal_noise > 0):
        # does the low level rescue visits whose issued goal was already off?
        far = [v for v in vlog if v["xy_err"] > CONTACT_R]
        near = [v for v in vlog if v["xy_err"] <= CONTACT_R]
        wrong = [v for v in vlog if not v["same_choice"]]
        summary.append(
            f"  visits with goal xy err > {CONTACT_R * 100:.1f}cm: "
            f"{len(far)}/{len(vlog)} ({len(far) / len(vlog) * 100:.1f}%)  "
            f"ok {_rate(far)}  | err <= : ok {_rate(near)}")
        summary.append(
            f"  wrong-candidate commands {len(wrong)}/{len(vlog)}  "
            f"ok {_rate(wrong)}")
    print("\n".join(summary))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines + summary) + "\n")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "arm": arm,
                "exec_steps": args.exec_steps, "seed": args.seed,
                "scene_type": args.scene_type,
                "n_extra_walls": (args.n_extra_walls
                                  if args.scene_type == "multi-wall"
                                  else None),
                "grid": grid, "train_grid": train_grid, "k_steps": k_eval,
                "readout": readout, "field_override": args.field_override,
                "hide_goal": args.hide_goal,
                "visit_log": [results[i].visit_log for i in range(n)],
                "n": n, "success": succ, "placed": placed, "bumps": bumps,
                "wall_hits": wh, "steps": steps,
                "visits_total": visits_all, "visits_ok": visits_ok,
                "plan_mse": mses, "obstacle_iou": ious,
                "det_noise": not args.stochastic_noise,
                "subgoal_noise": args.subgoal_noise,
                "high_ckpt": args.high_ckpt or "teacher",
                "high_queries": hi_n,
                "high_xyz_l1": hi_l1 / hi_n if hi_n else None,
                "high_open_acc": hi_open_ok / hi_n if hi_n else None,
            }, f)


if __name__ == "__main__":
    main()
