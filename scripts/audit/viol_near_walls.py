#!/usr/bin/env python3
"""Are local ordering violations enriched near obstacle cells?"""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from multi_task import sample_scene, decision_samples, G, xy_to_cell, DIRS
from multi_plan_data import leg_sample, EXT_DIM
from soma.spatial_baselines import build_model, predict_field

ck = sys.argv[1]
dev = "cpu"
d = torch.load(ck, map_location=dev, weights_only=False)
m = build_model(d["model_name"], **d["arch"]).to(dev)
m.load_state_dict(d["model"]); m.eval()

rng = np.random.RandomState(777)
legs = []
while len(legs) < 48:
    objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
    for xy, grip, objs_now, placed, active, cmd in decision_samples(
            objs, tgts, perm):
        s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active)
        if s is None or s[2].sum() < 2:
            continue
        legs.append(s + (xy, cmd[:2]))
        if len(legs) >= 48:
            break

all_d, viol_d, step_d = [], [], []
for ext, field, valid, xy, gy in legs:
    ob = ext[0].astype(bool)
    x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32)
    with torch.no_grad():
        pred = predict_field(m, d["model_name"], x, G)[0].numpy().reshape(G, G)
    dist = distance_transform_edt(~ob)          # cells to nearest obstacle
    cur, gc = xy_to_cell(xy), xy_to_cell(gy)
    seen = set()
    while cur != gc and cur not in seen:
        seen.add(cur)
        best, bv = None, field[cur]
        for di, dj, _c in DIRS:
            nb = (cur[0] + di, cur[1] + dj)
            if (0 <= nb[0] < G and 0 <= nb[1] < G and not ob[nb]
                    and field[nb] < bv - 1e-9):
                bv, best = field[nb], nb
        if best is None:
            break
        dv = dist[best]
        all_d.append(dv)
        step_d.append((cur, best, dv, pred[best] - pred[cur]))
        if pred[best] - pred[cur] >= -1e-6:
            viol_d.append(dv)
        cur = best

all_d, viol_d = np.array(all_d), np.array(viol_d)
print(f"{ck.split('/')[-1]}: steps {len(all_d)}  violations {len(viol_d)} "
      f"({len(viol_d)/len(all_d):.3f})")
for name, a in (("all steps", all_d), ("violations", viol_d)):
    if len(a):
        print(f"  {name:11s} dist-to-wall  median {np.median(a):.1f}  "
              f"<=1: {np.mean(a <= 1):.2f}  <=2: {np.mean(a <= 2):.2f}  "
              f"<=4: {np.mean(a <= 4):.2f}")
