#!/usr/bin/env python3
"""Compare path extractors on predicted plan fields."""
import heapq
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G, xy_to_cell, cell_to_xy, DIRS
from multi_plan_data import leg_sample, field_path, EXT_DIM
from soma.spatial_baselines import build_model, predict_field


def smooth(field, obstacle, iters=2):
    f = field.astype(np.float64).copy()
    free = ~obstacle
    for _ in range(iters):
        acc = np.zeros_like(f)
        cnt = np.zeros_like(f)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                src = (slice(max(0, di), G + min(0, di)),
                       slice(max(0, dj), G + min(0, dj)))
                dst = (slice(max(0, -di), G + min(0, -di)),
                       slice(max(0, -dj), G + min(0, -dj)))
                acc[dst] += f[src]
                cnt[dst] += 1.0
        f = np.where(free, acc / cnt, f)
    return f


def dijkstra_field(field, start, goal, obstacle, eps=0.02):
    """Least-cost path over free cells, cost = field value of entered cell."""
    dist = np.full((G, G), np.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        if u == goal:
            break
        for di, dj, c in DIRS:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < G and 0 <= v[1] < G) or obstacle[v]:
                continue
            nd = d + max(field[v], 0.0) + eps * c
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[goal]):
        return None
    cells = [goal]
    while cells[-1] != start:
        cells.append(prev[cells[-1]])
    return cells[::-1]


def cells_to_pts(cells, start_xy, goal_xy, stride=4):
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[stride::stride]]
    pts.append(np.asarray(goal_xy, np.float32))
    return np.array(pts, np.float32)


def plen(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def main():
    ckpts = sys.argv[1:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for ckpt in ckpts:
        d = torch.load(ckpt, map_location=device, weights_only=False)
        m = build_model(d["model_name"], **d["arch"]).to(device)
        m.load_state_dict(d["model"])
        m.eval()
        models.append((ckpt.split("/")[-1], d["model_name"], m))

    rng = np.random.RandomState(777)
    legs = []
    while len(legs) < 48:
        objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(objs, tgts, perm):
            s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active)
            if s is None or s[2].sum() < 2:
                continue
            legs.append(s + (xy, cmd[:2]))
            if len(legs) >= 48:
                break

    # ground truth reference lengths
    gt_ref = []
    for ext, field, valid, xy, gy in legs:
        ob = ext[0].astype(bool)
        p = field_path(field, xy, gy, ob)
        gt_ref.append(plen(p))

    for fname, name, model in models:
        res = {k: [0, []] for k in ("greedy", "smooth", "dijk")}
        for li, (ext, field, valid, xy, gy) in enumerate(legs):
            ob = ext[0].astype(bool)
            x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = predict_field(model, name, x, G)[0].cpu().numpy().reshape(G, G)
            sc, gc = xy_to_cell(xy), xy_to_cell(gy)
            outs = {}
            outs["greedy"] = field_path(pred, xy, gy, ob)
            fs = smooth(pred, ob)
            outs["smooth"] = field_path(fs, xy, gy, ob)
            cells = dijkstra_field(pred, sc, gc, ob)
            outs["dijk"] = (cells_to_pts(cells, xy, gy) if cells else None)
            for k, p in outs.items():
                if p is not None:
                    res[k][0] += 1
                    res[k][1].append(plen(p) / max(gt_ref[li], 1e-9))
        line = f"{fname} {name}:"
        for k in res:
            n = res[k][0]
            mean = np.mean(res[k][1]) if res[k][1] else float("nan")
            line += f"  {k} {n}/48 len {mean:.3f}"
        print(line)


if __name__ == "__main__":
    main()
