#!/usr/bin/env python3
"""Diagnose greedy-descent failures: local gradient sign along the GT path."""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G, xy_to_cell, DIRS
from multi_plan_data import leg_sample, field_path, EXT_DIM
from soma.spatial_baselines import build_model, predict_field


def main():
    ckpt = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(d["model_name"], **d["arch"]).to(device)
    model.load_state_dict(d["model"])
    model.eval()
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

    nfail = 0
    signed = []
    ratios = []
    stuck_cells = []
    for ext, field, valid, xy, gy in legs:
        ob = ext[0].astype(bool)
        x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = predict_field(model, name, x, G)[0].cpu().numpy().reshape(G, G)
        sc, gc = xy_to_cell(xy), xy_to_cell(gy)
        # walk the GT-optimal path; at each cell measure the predicted change
        cur = sc
        seen = set()
        ok = True
        while cur != gc and cur not in seen:
            seen.add(cur)
            best, bv = None, field[cur]
            for di, dj, _c in DIRS:
                nb = (cur[0] + di, cur[1] + dj)
                if (0 <= nb[0] < G and 0 <= nb[1] < G and not ob[nb]
                        and field[nb] < bv - 1e-9):
                    bv, best = field[nb], nb
            if best is None:
                ok = False
                break
            dp = pred[best] - pred[cur]
            dg = field[best] - field[cur]
            signed.append(dp)
            ratios.append(abs(dp) / max(abs(dg), 1e-9))
            cur = best
        if not ok:
            nfail += 1
            stuck_cells.append(cur)
        if field_path(pred, xy, gy, ob) is None:
            pass
    signed = np.array(signed)
    print(f"{name}: gt-path cells {len(signed)}  descent fails {nfail}/{len(legs)}")
    print(f"  pred delta along GT path: mean {signed.mean():+.5f}  "
          f"frac<0 {np.mean(signed < 0):.3f}  frac<0 within 1 cell-grad {np.mean(signed < -1.0/128):.3f}")
    ratios = np.array(ratios)
    print(f"  |dpred|/|dgt| median {np.median(ratios):.3f}  p90 {np.percentile(ratios,90):.3f}")
    # where do strict-descent runs stick?
    for ext, field, valid, xy, gy in legs:
        ob = ext[0].astype(bool)
        x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = predict_field(model, name, x, G)[0].cpu().numpy().reshape(G, G)
        p = field_path(pred, xy, gy, ob)
        if p is None:
            # replicate stuck cell
            sc, gc = xy_to_cell(xy), xy_to_cell(gy)
            cur = sc
            for _ in range(400):
                if cur == gc:
                    break
                best, bv = None, pred[cur]
                for di, dj, _c in DIRS:
                    nb = (cur[0] + di, cur[1] + dj)
                    if (0 <= nb[0] < G and 0 <= nb[1] < G and not ob[nb]
                            and pred[nb] < bv - 1e-6):
                        bv, best = pred[nb], nb
                if best is None:
                    break
                cur = best
            print(f"  stuck at {cur}  gt_dist {field[cur]*128:6.1f} cells  "
                  f"pred {pred[cur]:.4f}  gt/pred at start {field[sc]:.4f}/{pred[sc]:.4f}")


if __name__ == "__main__":
    main()
