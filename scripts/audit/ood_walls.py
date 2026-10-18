#!/usr/bin/env python3
"""OOD wall test: two perpendicular gapped walls at random positions, gaps not
anchored at the start. Models trained on center-anchored walls are evaluated on
their ability to produce a usable plan field on these layouts."""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import (G, XL, XH, YL, YH, WALL_INFLATE, OBJ_INFLATE,
                        xy_to_cell, cell_to_xy, _mask_rect, CENTER)
from multi_plan_data import make_ext, field_path, field_path_escape, EXT_DIM
from spatial_plan_data import geodesic_field
from soma.spatial_baselines import build_model, predict_field


def random_walls(rng):
    walls = []
    x = rng.uniform(0.32, 0.64); gy = rng.uniform(0.30, 0.52); hg = 0.055
    walls += [(x - 0.011, x + 0.011, YL, gy - hg),
              (x - 0.011, x + 0.011, gy + hg, YH)]
    y = rng.uniform(0.30, 0.52); gx = rng.uniform(0.32, 0.64); hg = 0.055
    walls += [(XL, gx - hg, y - 0.011, y + 0.011),
              (gx + hg, XH, y - 0.011, y + 0.011)]
    return walls


def obstacle_of(walls):
    ob = np.zeros((G, G), bool)
    for (x0, x1, y0, y1) in walls:
        _mask_rect(ob, x0 - WALL_INFLATE, x1 + WALL_INFLATE,
                   y0 - WALL_INFLATE, y1 + WALL_INFLATE)
    return ob


def make_leg(rng):
    for _ in range(200):
        walls = random_walls(rng)
        ob = obstacle_of(walls)
        sc = xy_to_cell(CENTER)
        free = np.argwhere(~ob)
        gc = tuple(free[rng.randint(len(free))])
        t = ob.copy(); t[sc] = False; t[gc] = False
        dist, reach = geodesic_field(t, gc)
        if not reach[sc] or gc == sc:
            continue
        field = np.where(np.isfinite(dist), dist / 128.0, 1.0).astype(np.float32)
        ext = make_ext(t, gc)
        return ext, field, t, CENTER.copy(), cell_to_xy(gc)
    return None


def plen(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def main():
    rng = np.random.RandomState(4242)
    legs = []
    while len(legs) < 40:
        leg = make_leg(rng)
        if leg is not None:
            legs.append(leg)
    gtlen = []
    keep = []
    for ext, field, ob, sc, gc in legs:
        p = field_path_escape(field, sc, gc, ob, radius=24)
        if p is None:
            continue
        keep.append((ext, field, ob, sc, gc))
        gtlen.append(plen(p))
    legs = keep
    print(f"OOD layouts: {len(legs)}  (gaps at random positions)")
    for ck in sys.argv[1:]:
        d = torch.load(ck, map_location="cuda", weights_only=False)
        m = build_model(d["model_name"], **d["arch"]).cuda()
        m.load_state_dict(d["model"]); m.eval()
        g = e = 0; rat = []
        for li, (ext, field, ob, sc, gc) in enumerate(legs):
            x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None],
                             dtype=torch.float32, device="cuda")
            with torch.no_grad():
                pred = predict_field(m, d["model_name"], x, G)[0].cpu().numpy().reshape(G, G)
            pg = field_path(pred, sc, gc, ob)
            pe = field_path_escape(pred, sc, gc, ob, radius=24)
            g += int(pg is not None)
            e += int(pe is not None)
            if pe is not None:
                rat.append(plen(pe) / gtlen[li])
        print(f"{ck.split('/')[-1]:24s} greedy {g}/40  escape {e}/40  "
              f"len/gt {np.mean(rat) if rat else float('nan'):.3f}")


if __name__ == "__main__":
    main()
