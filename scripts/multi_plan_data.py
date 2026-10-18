#!/usr/bin/env python3
"""Plan-field samples for the multi-object visit task.

One sample = one visit leg. Inputs are the same occupancy grid the routed
executor plans on — walls and unplaced non-goal objects, inflated exactly as
in route_xy — plus start/goal cells. Target is the normalized 8-connected
Dijkstra distance-to-goal field (supervision = the classical solver).

ext[3,G,G] = [obstacle, dx/G, dy/G] where (dx,dy) are the cell-to-goal
offsets in cells. The goal is a command, and the offsets encode it exactly;
one-hot goal/start channels were deliberately dropped because they leaked
into the field as local bumps (the goal became a local maximum, the start a
local minimum), trapping greedy waypoint descent at both ends.
"""
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from multi_task import (sample_scene, decision_samples, G, WALL_INFLATE,
                        OBJ_INFLATE, xy_to_cell, cell_to_xy, _mask_rect, DIRS)
from spatial_plan_data import geodesic_field


def obstacle_grid(objs, goal_xy, walls, placed, active):
    """Occupancy the plan field routes on: inflated walls + inflated unplaced
    objects that are neither the carried one nor the goal object."""
    obstacle = np.zeros((G, G), bool)
    for (x0, x1, y0, y1) in walls:
        _mask_rect(obstacle, x0 - WALL_INFLATE, x1 + WALL_INFLATE,
                   y0 - WALL_INFLATE, y1 + WALL_INFLATE)
    for j in range(len(objs)):
        if placed[j] or j == active:
            continue
        if np.linalg.norm(objs[j] - goal_xy) < 0.05:
            continue
        _mask_rect(obstacle, objs[j, 0] - OBJ_INFLATE, objs[j, 0] + OBJ_INFLATE,
                   objs[j, 1] - OBJ_INFLATE, objs[j, 1] + OBJ_INFLATE)
    return obstacle


EXT_DIM = 3
OBSTACLE_CH = 0


def make_ext(obstacle, gc):
    ext = np.zeros((EXT_DIM, G, G), np.float32)
    ext[0] = obstacle
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    ext[1] = (gc[1] - jj) / float(G)
    ext[2] = (gc[0] - ii) / float(G)
    return ext


def leg_sample(objs, start_xy, goal_xy, walls, placed, active, norm_c=2 * G):
    obstacle = obstacle_grid(objs, goal_xy, walls, placed, active)
    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    obstacle[sc] = False
    obstacle[gc] = False
    dist, reach = geodesic_field(obstacle, gc)
    if not reach[sc]:
        return None
    finite = np.isfinite(dist)
    field = np.where(finite, dist / (norm_c + 1e-9), 1.0).astype(np.float32)
    ext = make_ext(obstacle, gc)
    valid = reach & ~obstacle
    return ext, field, valid


def _sink_goal(field, gc, obstacle):
    """The goal is a known boundary condition (distance 0): make it strictly
    lower than every free neighbour so descent can always enter it. A flat
    field stays flat away from the goal, so an uninformative field still
    fails; this only removes the readout's failure to enter the goal."""
    field = field.copy()
    vals = [field[gc[0] + di, gc[1] + dj] for di, dj, _c in DIRS
            if 0 <= gc[0] + di < G and 0 <= gc[1] + dj < G
            and not obstacle[gc[0] + di, gc[1] + dj]]
    if vals:
        field[gc] = min(field[gc], min(vals)) - 1e-6
    return field


def field_path(field, start_xy, goal_xy, obstacle, max_steps=400, stride=2):
    """Greedy-descent waypoints on a predicted plan field. None if the descent
    does not reach the goal (caller falls back to a straight line)."""
    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    obstacle = obstacle.copy()
    obstacle[sc] = False
    obstacle[gc] = False
    field = _sink_goal(field, gc, obstacle)
    cur, cells = sc, [sc]
    for _ in range(max_steps):
        if cur == gc:
            break
        best, bv = None, field[cur]
        for di, dj, _c in DIRS:
            ni, nj = cur[0] + di, cur[1] + dj
            if (0 <= ni < G and 0 <= nj < G and not obstacle[ni, nj]
                    and field[ni, nj] < bv - 1e-6):
                bv, best = field[ni, nj], (ni, nj)
        if best is None:
            return None
        cur = best
        cells.append(cur)
    if cur != gc:
        return None
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[stride::stride]]
    pts.append(np.asarray(goal_xy, np.float32))
    return np.array(pts, np.float32)


def field_path_escape(field, start_xy, goal_xy, obstacle, radius=10,
                      max_steps=400, stride=2):
    """Strict descent; when trapped in a shallow basin, BFS to the nearest
    strictly lower cell and resume descent. Driven only by the predicted
    field, so an uninformative field still fails."""
    import collections

    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    obstacle = obstacle.copy()
    obstacle[sc] = False
    obstacle[gc] = False
    field = _sink_goal(field, gc, obstacle)
    cur, cells = sc, [sc]
    for _ in range(max_steps):
        if cur == gc:
            break
        best, bv = None, field[cur]
        for di, dj, _c in DIRS:
            v = (cur[0] + di, cur[1] + dj)
            if (0 <= v[0] < G and 0 <= v[1] < G and not obstacle[v]
                    and field[v] < bv - 1e-6):
                bv, best = field[v], v
        if best is not None:
            cur = best
            cells.append(cur)
            continue
        q = collections.deque([(cur, [cur])])
        seen = {cur}
        found = None
        target = field[cur] - 1e-6
        while q and found is None:
            u, p = q.popleft()
            if len(p) > radius:
                continue
            for di, dj, _c in DIRS:
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < G and 0 <= v[1] < G) or obstacle[v] \
                        or v in seen:
                    continue
                seen.add(v)
                if field[v] < target:
                    found = p + [v]
                    break
                q.append((v, p + [v]))
        if found is None:
            return None
        cells.extend(found[1:])
        cur = found[-1]
    if cur != gc:
        return None
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[stride::stride]]
    pts.append(np.asarray(goal_xy, np.float32))
    return np.array(pts, np.float32)


def field_path_cost(field, start_xy, goal_xy, obstacle, step_cost=0.02,
                    stride=2):
    """Least-cost path over free cells with entry cost = predicted value +
    step_cost. Unlike strict descent this cannot stall in a shallow local
    minimum; a flat field degenerates to ordinary shortest-path routing."""
    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    obstacle = obstacle.copy()
    obstacle[sc] = False
    obstacle[gc] = False
    dist = np.full((G, G), np.inf)
    prev = {}
    dist[sc] = 0.0
    pq = [(0.0, sc)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        if u == gc:
            break
        for di, dj, c in DIRS:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < G and 0 <= v[1] < G) or obstacle[v]:
                continue
            nd = d + max(float(field[v]), 0.0) + step_cost * c
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[gc]):
        return None
    cells = [gc]
    while cells[-1] != sc:
        cells.append(prev[cells[-1]])
    cells.reverse()
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[stride::stride]]
    pts.append(np.asarray(goal_xy, np.float32))
    return np.array(pts, np.float32)


def make_batch(batch, rng, device="cpu", two_walls=True, norm_c=2 * G):
    exts, tars, vals = [], [], []
    tries = 0
    while len(exts) < batch and tries < batch * 20:
        tries += 1
        objs, tgts, perm, walls = sample_scene(rng, two_walls=two_walls)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(
                objs, tgts, perm):
            s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active,
                           norm_c=norm_c)
            if s is None or s[2].sum() < 2:
                continue
            exts.append(s[0].reshape(EXT_DIM, -1).T)
            tars.append(s[1].reshape(-1))
            vals.append(s[2].reshape(-1))
            if len(exts) >= batch:
                break
    ext = torch.tensor(np.stack(exts), dtype=torch.float32, device=device)
    tar = torch.tensor(np.stack(tars), dtype=torch.float32, device=device)
    val = torch.tensor(np.stack(vals), dtype=torch.bool, device=device)
    return ext, tar, val


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    ext, tar, val = make_batch(4, rng)
    print(f"ext {tuple(ext.shape)} target [{tar.min():.2f},{tar.max():.2f}] "
          f"valid_frac {val.float().mean():.3f}")
