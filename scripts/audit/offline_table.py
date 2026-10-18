#!/usr/bin/env python3
"""Uniform offline table for plan-field checkpoints on one fixed set of legs."""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G, xy_to_cell, DIRS
from multi_plan_data import (leg_sample, field_path, field_path_escape,
                             EXT_DIM)
from soma.spatial_baselines import build_model, predict_field


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"{'model':22s} {'params':>8s} {'valMSE':>7s} {'strict':>7s} "
          f"{'escape':>7s} {'sign<0':>7s} {'|dp|/|dg|':>9s}")
    for ck in sys.argv[1:]:
        d = torch.load(ck, map_location=dev, weights_only=False)
        name = d["model_name"]
        m = build_model(name, **d["arch"]).to(dev)
        m.load_state_dict(d["model"]); m.eval()
        nparam = sum(p.numel() for p in m.parameters())
        okg = oke = 0
        sign, ratio = [], []
        for ext, field, valid, xy, gy in legs:
            ob = ext[0].astype(bool)
            x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None],
                             dtype=torch.float32, device=dev)
            with torch.no_grad():
                pred = predict_field(m, name, x, G)[0].cpu().numpy().reshape(G, G)
            okg += int(field_path(pred, xy, gy, ob) is not None)
            oke += int(field_path_escape(pred, xy, gy, ob, radius=24) is not None)
            # local ordering along the GT-descent path
            cur = xy_to_cell(xy); gc = xy_to_cell(gy)
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
                dp, dg = pred[best] - pred[cur], field[best] - field[cur]
                sign.append(dp < -1e-6)
                ratio.append(abs(dp) / max(abs(dg), 1e-9))
                cur = best
        print(f"{ck.split('/')[-1]:22s} {nparam:8d} {d.get('val_mse', float('nan')):7.4f} "
              f"{okg:3d}/48 {oke:3d}/48 {np.mean(sign):7.3f} "
              f"{np.median(ratio):9.3f}")


if __name__ == "__main__":
    main()
