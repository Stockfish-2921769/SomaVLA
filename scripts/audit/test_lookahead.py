#!/usr/bin/env python3
"""Strict descent vs short-horizon lookahead descent on predicted fields."""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G, xy_to_cell, cell_to_xy, DIRS
from multi_plan_data import leg_sample, field_path, EXT_DIM
from soma.spatial_baselines import build_model, predict_field


def lookahead_path(field, start_xy, goal_xy, obstacle, depth=3, beam=8,
                   max_steps=400, stride=4):
    sc, gc = xy_to_cell(start_xy), xy_to_cell(goal_xy)
    ob = obstacle.copy()
    ob[sc] = False
    ob[gc] = False
    cur, cells, seen = sc, [sc], {sc}
    for _ in range(max_steps):
        if cur == gc:
            break
        frontier = [([cur], field[cur])]
        for _d in range(depth):
            nxt = []
            for path, _v in frontier:
                u = path[-1]
                for di, dj, _c in DIRS:
                    v = (u[0] + di, u[1] + dj)
                    if (0 <= v[0] < G and 0 <= v[1] < G and not ob[v]
                            and v not in seen and v not in path):
                        nxt.append((path + [v], field[v]))
            if not nxt:
                break
            nxt.sort(key=lambda x: x[1])
            frontier = nxt[:beam]
        if not frontier:
            return None
        path = frontier[0][0]
        for v in path[1:]:
            seen.add(v)
            cells.append(v)
            cur = v
            if cur == gc:
                break
    if cur != gc:
        return None
    pts = [np.asarray(start_xy, np.float32)]
    pts += [cell_to_xy(c) for c in cells[stride::stride]]
    pts.append(np.asarray(goal_xy, np.float32))
    return np.array(pts, np.float32)


def plen(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def main():
    ckpt = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(ckpt, map_location=device, weights_only=False)
    m = build_model(d["model_name"], **d["arch"]).to(device)
    m.load_state_dict(d["model"]); m.eval()
    name = d["model_name"]
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
    gt_len = []
    for ext, field, valid, xy, gy in legs:
        gt_len.append(plen(field_path(field, xy, gy, ext[0].astype(bool))))
    for depth in (1, 2, 3, 4):
        ok, rat = 0, []
        for li, (ext, field, valid, xy, gy) in enumerate(legs):
            ob = ext[0].astype(bool)
            x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = predict_field(m, name, x, G)[0].cpu().numpy().reshape(G, G)
            p = lookahead_path(pred, xy, gy, ob, depth=depth)
            if p is not None:
                ok += 1
                rat.append(plen(p) / gt_len[li])
        print(f"{name} depth {depth}: ok {ok}/48  len/gt {np.mean(rat):.3f}")


if __name__ == "__main__":
    main()
