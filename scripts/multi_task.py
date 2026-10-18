#!/usr/bin/env python3
"""Multi-object long-horizon pick-and-place with wall obstacles — the task for
the planner A/B.

Long-horizon: each object needs the full skill chain (approach -> grasp -> lift
-> transport -> place -> release) and an episode must place N=3 objects => ~3x
the horizon of the single-object task. Planning content the VLA must supply:
  (1) ORDER — which object to pick next (any order is legal),
  (2) ROUTE — how to carry it to its color-matched target around the walls and
              past the other objects (hitting either loses the wall race /
              knocks the neighbor).

The teacher ROUTES (classical 8-connected Dijkstra over walls+objects — the same
geodesic the spatial NCA is trained to imitate), so demonstrations are expert
(successful). The naive baseline is the BC VLA alone, which must recover that
routing implicitly; the treatment feeds it an explicit NCA subgoal.

Reuses proc_sim's skill-target semantics and SimEnv's damped plant. Physics is
set EASY/uniform (no Coulomb drops) so the bottleneck is planning, not grip.
Workspace / scene are self-contained here; the single-object stack is untouched.
"""
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw

# ── workspace + task constants ────────────────────────────────────────
XL, XH, YL, YH = 0.22, 0.74, 0.22, 0.60      # workspace bounds (m)
S = 256                                       # image side (VLA input)
N_OBJ = 3
G = 64                                        # routing grid resolution

CONTACT_Z = 0.52
GRASP_Z = 0.58
LIFT_Z = 0.66
NEUTRAL_Z = 0.76

CONTACT_R = 0.025                             # grasp radius (EEF -> object)
PLACE_TOL = 0.02                              # placement tolerance (m)
BUMP_R = 0.075                                # carried-object -> other-object knock radius
MIN_SEP_OBJ = 0.105
MIN_SEP_TGT = 0.100
WALL_T = 0.022                                # wall thickness (m)
OBJ_INFLATE = 0.090                           # routing clearance around objects (> BUMP_R)
WALL_INFLATE = 0.025                          # routing clearance around walls (m)
WALL_CLEAR = 0.055                            # min scene-point distance to a wall (reachability)

COLORS = [(205, 65, 65), (60, 120, 215), (70, 170, 90), (210, 160, 40),
          (150, 90, 200), (40, 175, 175)]

DIRS = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356)]

SKILL_TO = {
    "approach":  ("obj",   GRASP_Z,   1.0),
    "grasp":     ("obj",   CONTACT_Z, 0.0),
    "lift":      ("obj",   LIFT_Z,    0.0),
    "transport": ("place", LIFT_Z,    0.0),
    "place":     ("place", CONTACT_Z, 0.0),
    "release":   ("place", CONTACT_Z, 1.0),
}
SKILL_T = {"approach": 90, "grasp": 35, "lift": 35, "transport": 95,
           "place": 35, "release": 20}
SKILLS = ["approach", "grasp", "lift", "transport", "place", "release"]


# ── grid <-> xy helpers ───────────────────────────────────────────────
def xy_to_cell(p):
    j = int((p[0] - XL) / (XH - XL) * G)
    i = int((YH - p[1]) / (YH - YL) * G)
    return (min(max(i, 0), G - 1), min(max(j, 0), G - 1))


def cell_to_xy(ij):
    i, j = ij
    return np.array([XL + (j + 0.5) / G * (XH - XL),
                     YH - (i + 0.5) / G * (YH - YL)], np.float32)


def _mask_rect(mask, x0, x1, y0, y1):
    j0 = int((x0 - XL) / (XH - XL) * G); j1 = int(np.ceil((x1 - XL) / (XH - XL) * G))
    i0 = int((YH - y1) / (YH - YL) * G); i1 = int(np.ceil((YH - y0) / (YH - YL) * G))
    mask[max(i0, 0):min(i1, G), max(j0, 0):min(j1, G)] = True


def in_walls(p, walls):
    for (x0, x1, y0, y1) in walls:
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return True
    return False


# ── geodesic routing (same solver the NCA imitates) ───────────────────
def route_xy(start_xy, goal_xy, walls, other_objs):
    """Waypoint polyline from start to goal around walls + objects (8-connected
    Dijkstra). Returns np[N,2] or None if the goal is unreachable."""
    mask = np.zeros((G, G), bool)
    for (x0, x1, y0, y1) in walls:
        _mask_rect(mask, x0 - WALL_INFLATE, x1 + WALL_INFLATE,
                   y0 - WALL_INFLATE, y1 + WALL_INFLATE)
    for o in other_objs:
        _mask_rect(mask, o[0] - OBJ_INFLATE, o[0] + OBJ_INFLATE,
                   o[1] - OBJ_INFLATE, o[1] + OBJ_INFLATE)
    gc, sc = xy_to_cell(goal_xy), xy_to_cell(start_xy)
    mask[gc] = False; mask[sc] = False
    dist, reach = _geodesic(mask, gc)
    if not reach[sc]:
        return None
    cells, cur = [sc], sc
    for _ in range(8 * G):
        if cur == gc:
            break
        best, bd = None, dist[cur]
        for di, dj, _c in DIRS:
            ni, nj = cur[0] + di, cur[1] + dj
            if 0 <= ni < G and 0 <= nj < G and reach[ni, nj] and dist[ni, nj] < bd - 1e-9:
                bd = dist[ni, nj]; best = (ni, nj)
        if best is None:
            return None
        cur = best
        cells.append(cur)
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[::4]]
    pts += [np.asarray(goal_xy, np.float32)]
    return np.array(pts, np.float32)


def _geodesic(obstacle, goal):
    g = obstacle.shape[0]
    dist = np.full((g, g), np.inf)
    pq = []
    if not obstacle[goal]:
        dist[goal] = 0.0
        heapq.heappush(pq, (0.0, goal[0], goal[1]))
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, c in DIRS:
            ni, nj = i + di, j + dj
            if 0 <= ni < g and 0 <= nj < g and not obstacle[ni, nj]:
                nd = d + c
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
    return dist, np.isfinite(dist)


# ── scene sampling ────────────────────────────────────────────────────
def _pt_wall_dist(p, walls):
    d = 1e9
    for (x0, x1, y0, y1) in walls:
        dx = max(x0 - p[0], 0.0, p[0] - x1)
        dy = max(y0 - p[1], 0.0, p[1] - y1)
        d = min(d, float(np.hypot(dx, dy)))
    return d


def _min_sep_pts(rng, n, existing, sep, walls, tries=800):
    """Points with pairwise separation >= sep when possible, always clear of
    walls. If the scene is crowded, keeps the best-separated wall-clear point."""
    pts = []
    for _ in range(n):
        best, best_d = None, -1.0
        for _ in range(tries):
            q = rng.uniform([XL + 0.03, YL + 0.03],
                            [XH - 0.03, YH - 0.03]).astype(np.float32)
            if _pt_wall_dist(q, walls) < WALL_CLEAR:
                continue
            d = min([np.linalg.norm(q - np.asarray(r))
                     for r in list(existing) + pts] or [9.0])
            if d >= sep:
                best, best_d = q, d
                break
            if d > best_d:
                best, best_d = q, d
        pts.append(best)
    return pts


CENTER = np.array([0.5 * (XL + XH), 0.5 * (YL + YH)], np.float32)


def _scene_reachable(objs, tgts, walls):
    """Every object / target reachable from the start hover with all
    unplaced objects inflated as obstacles. Rejects layouts where a gap is
    plugged by an object or the center hover is inside a wall."""
    for k in range(len(objs)):
        others = [o for j, o in enumerate(objs) if j != k]
        if route_xy(CENTER, objs[k], walls, others) is None:
            return False
    for t in tgts:
        if route_xy(CENTER, t, walls, list(objs)) is None:
            return False
    return True


def sample_scene(rng, n=N_OBJ, two_walls=False):
    """Returns (objs[n,2], targets[n,2], perm[n], walls list-of-rects).

    One full-span wall (random orientation) with a single gap splits the
    workspace. With two_walls, a perpendicular wall is added, so most
    transports must thread two offset gaps — straight-line low-level control
    grinds and stalls, and the object order starts to matter. Layouts where
    the two walls cut the start hover off from the scene are resampled.
    """
    for _attempt in range(300):
        if rng.rand() < 0.5:            # vertical wall (blocks x)
            xw = rng.uniform(0.42, 0.54)
            gw = rng.uniform(0.14, 0.21)
            gap = float(np.clip(CENTER[1] + rng.uniform(-0.02, 0.02),
                                YL + 0.12, YH - 0.12))
            walls = [(xw - WALL_T / 2, xw + WALL_T / 2, YL - 0.02, gap - gw / 2),
                     (xw - WALL_T / 2, xw + WALL_T / 2, gap + gw / 2, YH + 0.02)]
            if two_walls:
                yw = rng.uniform(0.32, 0.48)
                g2w = rng.uniform(0.14, 0.21)
                g2 = float(np.clip(CENTER[0] + rng.uniform(-0.02, 0.02),
                                   XL + 0.12, XH - 0.12))
                walls += [(XL - 0.02, g2 - g2w / 2, yw - WALL_T / 2, yw + WALL_T / 2),
                          (g2 + g2w / 2, XH + 0.02, yw - WALL_T / 2, yw + WALL_T / 2)]
        else:                           # horizontal wall (blocks y)
            yw = rng.uniform(0.33, 0.47)
            gw = rng.uniform(0.14, 0.21)
            gap = float(np.clip(CENTER[0] + rng.uniform(-0.02, 0.02),
                                XL + 0.12, XH - 0.12))
            walls = [(XL - 0.02, gap - gw / 2, yw - WALL_T / 2, yw + WALL_T / 2),
                     (gap + gw / 2, XH + 0.02, yw - WALL_T / 2, yw + WALL_T / 2)]
            if two_walls:
                xw = rng.uniform(0.40, 0.56)
                g2w = rng.uniform(0.14, 0.21)
                g2 = float(np.clip(CENTER[1] + rng.uniform(-0.02, 0.02),
                                   YL + 0.12, YH - 0.12))
                walls += [(xw - WALL_T / 2, xw + WALL_T / 2, YL - 0.02, g2 - g2w / 2),
                          (xw - WALL_T / 2, xw + WALL_T / 2, g2 + g2w / 2, YH + 0.02)]
        if _pt_wall_dist(CENTER, walls) < WALL_CLEAR:
            continue
        objs = _min_sep_pts(rng, n, [], MIN_SEP_OBJ, walls)
        tgts = _min_sep_pts(rng, n, objs, MIN_SEP_TGT, walls)
        if any(p is None for p in objs + tgts):
            continue
        if not _scene_reachable(objs, tgts, walls):
            continue
        perm = rng.permutation(n)
        return (np.array(objs, np.float32), np.array(tgts, np.float32), perm,
                walls)
    raise RuntimeError("sample_scene: no reachable layout found")


# ── environment ───────────────────────────────────────────────────────
class MultiEnv:
    """Damped plant + multi-object grasp/place + wall barrier + object knock.

    state[7] = [pos(3), rot(3), openness(1)]. Walls are full-height barriers:
    an EEF move whose xy lands inside a wall does not advance in xy. An unplaced
    object coming within BUMP_R of the carried object is knocked (displaced).
    Deterministic attach physics (no Coulomb drops) — the failure signal is
    planning (wall stall / bump / misplacement / timeout).
    """

    def __init__(self, rng, objs, targets, perm, walls, plant_f=0.5,
                 pos_noise=0.001, rot_noise=0.01, open_noise=0.01):
        self.rng = rng
        self.objs0 = np.asarray(objs, np.float32).copy()
        self.targets = np.asarray(targets, np.float32)
        self.perm = np.asarray(perm)
        self.walls = list(walls)
        self.n = len(objs)
        self.plant_f = plant_f
        self.noise = np.array([pos_noise] * 3 + [rot_noise] * 3 + [open_noise], np.float32)
        self.reset()

    def reset(self):
        self.objs = self.objs0.copy()
        self.placed = np.zeros(self.n, bool)
        self.bumps = np.zeros(self.n, int)
        self.wall_hits = 0
        self.attached = False
        self.active = -1
        cx = 0.5 * (XL + XH); cy = 0.5 * (YL + YH)
        self.state = np.array([cx, cy, NEUTRAL_Z, 0, 0, 0, 1.0], np.float32)
        return self.state.copy()

    def target_of(self, i):
        return self.targets[self.perm[i]]

    def _knock(self, j):
        if self.placed[j] or j == self.active:
            return
        ang = self.rng.uniform(0, 2 * np.pi)
        mag = self.rng.uniform(0.04, 0.09)
        p = self.objs[j] + mag * np.array([np.cos(ang), np.sin(ang)], np.float32)
        p = np.clip(p, [XL + 0.03, YL + 0.03], [XH - 0.03, YH - 0.03])
        if not in_walls(p, self.walls):
            self.objs[j] = p
        self.bumps[j] += 1

    def step(self, target_pose):
        target = np.asarray(target_pose, np.float32)
        prev_xy = self.state[:2].copy()
        new = self.state + self.plant_f * (target - self.state)
        new = new + self.noise * self.rng.randn(7).astype(np.float32)
        new[6] = float(np.clip(new[6], 0.0, 1.0))
        if in_walls(new[:2], self.walls):          # full-height barrier: slide, don't pass
            self.wall_hits += 1
            tx, ty = new[:2]
            px, py = prev_xy
            if not in_walls((tx, py), self.walls):
                px = tx
            if not in_walls((px, ty), self.walls):
                py = ty
            new[:2] = (px, py)
        self.state = new
        s = self.state
        info = {"attached": self.attached, "active": self.active,
                "placed_now": -1, "bumped": []}

        if not self.attached and s[6] < 0.5 and s[2] < GRASP_Z + 0.03:
            d = np.linalg.norm(self.objs - s[:2], axis=1)
            d[self.placed] = 1e9
            i = int(d.argmin())
            if d[i] < CONTACT_R:
                self.attached = True
                self.active = i

        if self.attached:
            self.objs[self.active] = s[:2]
            d = np.linalg.norm(self.objs - s[:2], axis=1)
            for j in range(self.n):
                if j != self.active and not self.placed[j] and d[j] < BUMP_R:
                    self._knock(j)
                    info["bumped"].append(j)
            if s[6] >= 0.5:
                i = self.active
                if np.linalg.norm(self.objs[i] - self.target_of(i)) < PLACE_TOL:
                    self.placed[i] = True
                    info["placed_now"] = i
                self.attached = False
                self.active = -1

        info["attached"] = self.attached
        info["active"] = self.active
        return s.copy(), info

    def n_placed(self):
        return int(self.placed.sum())

    def success(self):
        return bool(self.placed.all())


# ── render (VLA input) ────────────────────────────────────────────────
def _px(p):
    return (int((p[0] - XL) / (XH - XL) * (S - 1)),
            int((YH - p[1]) / (YH - YL) * (S - 1)))


def render_scene(objs, targets, perm, placed, walls, eef, s=S):
    img = Image.new("RGB", (s, s), (238, 235, 228))
    d = ImageDraw.Draw(img)
    for g in range(0, s, 32):
        d.line([(g, 0), (g, s)], fill=(224, 220, 212))
        d.line([(0, g), (s, g)], fill=(224, 220, 212))
    for (x0, x1, y0, y1) in walls:               # walls (gray)
        a = _px((x0, y1)); b = _px((x1, y0))
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]),
                     max(a[0], b[0]), max(a[1], b[1])], fill=(120, 120, 128))
    for i in range(len(objs)):                   # targets (rings, object color)
        col = COLORS[i % len(COLORS)]
        tx, ty = _px(targets[perm[i]])
        d.ellipse([tx - 11, ty - 11, tx + 11, ty + 11], outline=col, width=3)
        if placed[i]:
            d.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], fill=col)
    for i in range(len(objs)):                   # objects (skip placed)
        if placed[i]:
            continue
        col = COLORS[i % len(COLORS)]
        ox, oy = _px(objs[i])
        d.rectangle([ox - 7, oy - 7, ox + 7, oy + 7], fill=col,
                    outline=(70, 70, 80), width=2)
    ex, ey = _px(eef)
    d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(25, 25, 30),
              outline=(255, 255, 255), width=1)
    return img


def siglip_normalize(arr):
    return (arr.transpose(2, 0, 1) - 0.5) / 0.5


# ── teacher: nearest-first order + geodesic-routed transport ──────────
def nearest_order(objs, start_xy):
    rem = list(range(len(objs)))
    order, cur = [], np.asarray(start_xy, np.float32)
    while rem:
        d = [np.linalg.norm(objs[i] - cur) for i in rem]
        i = rem[int(np.argmin(d))]
        order.append(i)
        cur = objs[i]
        rem.remove(i)
    return order


def _smooth(x):
    return x * x * (3.0 - 2.0 * x)


def _polyline_traj(path, t, rng):
    """Interpolate t poses along a polyline (xy only); returns [t,2] xy."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 1e-9 else 1.0
    f = _smooth(np.linspace(0, 1, t))
    out = np.zeros((t, 2), np.float32)
    for k, fk in enumerate(f):
        d = fk * total
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        r = (d - cum[i]) / seg[i] if seg[i] > 1e-9 else 0.0
        out[k] = path[i] + (path[i + 1] - path[i]) * r
    out += rng.normal(0, 0.002, out.shape).astype(np.float32)
    return out


def teacher_rollout(objs, targets, perm, walls, rng, order=None):
    """Procedural expert demo: nearest-first order, geodesic-routed transport.

    Returns (states[T,7], goals[T,4], obj_hist[T,n,2], placed_hist[T,n] bool).
    goals[t] is the active skill's subgoal [gx, gy, gz, open] at step t."""
    objs = objs.copy()
    placed = np.zeros(len(objs), bool)
    start_xy = np.array([0.5 * (XL + XH), 0.5 * (YL + YH)], np.float32)
    if order is None:
        order = nearest_order(objs, start_xy)

    state = np.array([start_xy[0], start_xy[1], NEUTRAL_Z, 0, 0, 0, 1.0], np.float32)
    states, goals, obj_hist, pl_hist = [], [], [], []
    for i in order:
        obj, place = objs[i], targets[perm[i]]
        for skill in SKILLS:
            which, z, op = SKILL_TO[skill]
            base = obj if which == "obj" else place
            t = SKILL_T[skill]
            if skill == "transport":
                others = [objs[j] for j in range(len(objs))
                          if j != i and not placed[j]]
                path = route_xy(state[:2], place, walls, others)
                if path is None or len(path) < 2:
                    path = np.array([state[:2], place], np.float32)
                xy = _polyline_traj(np.asarray(path, np.float32), t, rng)
            else:
                B = np.array([base[0], base[1], z], np.float32)
                prof = _smooth(np.linspace(0, 1, t))[:, None]
                xy = state[:2][None, :] + (B[None, :2] - state[:2][None, :]) * prof
                xy += rng.normal(0, 0.002, xy.shape).astype(np.float32)
            zz = np.linspace(state[2], z, t)[:, None]
            rot = np.tile(state[3:6], (t, 1))
            openp = np.linspace(state[6], op, t)[:, None]
            carrying = skill in ("lift", "transport", "place", "release")
            goal = np.array([base[0], base[1], z, op], np.float32)
            for k in range(t):
                st = np.concatenate([xy[k], zz[k], rot[k], openp[k]]).astype(np.float32)
                states.append(st)
                goals.append(goal)
                o = objs.copy()
                if carrying:
                    o[i] = xy[k]
                obj_hist.append(o)
                pl_hist.append(placed.copy())
            state = np.concatenate([np.array([base[0], base[1], z], np.float32),
                                    state[3:6], [op]]).astype(np.float32)
            objs[i] = np.array([base[0], base[1]], np.float32)
        placed[i] = True
    return (np.asarray(states, np.float32), np.asarray(goals, np.float32),
            np.asarray(obj_hist, np.float32), np.asarray(pl_hist, bool))


def decision_samples(objs, targets, perm, start_xy=None, order=None):
    """Teacher visit plan as high-level decision samples at hover states.

    One decision per visit (grasp a specific object / place at a specific
    target), taken from a hover at NEUTRAL_Z above the previous visit's end.
    This is the state the policy is queried from, and it is unambiguous: the
    label depends on the scene (which objects remain, colors) and on whether
    something is carried — not on the hidden skill phase. Labels are the visit
    terminal subgoal [gx, gy, CONTACT_Z, open] with open=0 (grasp object) /
    1 (place at target).

    Returns a list of (eef_xy[2], grip, objs_now[n,2], placed[n], active, cmd[4]).
    """
    objs = np.asarray(objs, np.float32).copy()
    targets = np.asarray(targets, np.float32)
    perm = np.asarray(perm)
    n = len(objs)
    if start_xy is None:
        start_xy = np.array([0.5 * (XL + XH), 0.5 * (YL + YH)], np.float32)
    cur = np.asarray(start_xy, np.float32).copy()
    objs_now = objs.copy()
    placed = np.zeros(n, bool)
    grip = 1.0
    out = []
    # nearest-first recomputed from the CURRENT hover, so the rule is
    # reproducible from the decision-time observation (EEF dot + objects).
    todo = list(range(n)) if order is None else list(order)
    while todo:
        d = [np.linalg.norm(objs_now[j] - cur) for j in todo]
        i = todo[int(np.argmin(d))]
        todo.remove(i)
        o = objs_now[i]
        out.append((cur.copy(), grip, objs_now.copy(), placed.copy(), -1,
                    np.array([o[0], o[1], CONTACT_Z, 0.0], np.float32)))
        cur = o.copy()                 # carried, hovering above the object
        grip = 0.0
        tgt = targets[perm[i]]
        out.append((cur.copy(), grip, objs_now.copy(), placed.copy(), i,
                    np.array([tgt[0], tgt[1], CONTACT_Z, 1.0], np.float32)))
        objs_now[i] = tgt.copy()
        placed[i] = True
        cur = tgt.copy()               # released, hovering above the target
        grip = 1.0
    return out


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    objs, tgts, perm, walls = sample_scene(rng)
    print("walls", [tuple(round(v, 3) for v in w) for w in walls])
    print("objs", np.round(objs, 3).tolist(), "perm", perm.tolist())
    st, goals, oh, ph = teacher_rollout(objs, tgts, perm, walls, rng)
    print("teacher rollout T =", len(st))
