#!/usr/bin/env python3
"""End-to-end descent test for a trained plan-field checkpoint on fresh scenes."""
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G
from multi_plan_data import leg_sample, field_path, EXT_DIM
from soma.spatial_baselines import build_model, predict_field


def load(ckpt, device):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(d["model_name"], **d["arch"]).to(device)
    model.load_state_dict(d["model"])
    model.eval()
    return model, d["model_name"]


def main():
    ckpt = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, name = load(ckpt, device)
    rng = np.random.RandomState(777)

    legs = []
    while len(legs) < 48:
        objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(objs, tgts, perm):
            s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active)
            if s is None or s[2].sum() < 2:
                continue
            ext, field, valid = s
            legs.append((ext, field, xy, cmd[:2]))
            if len(legs) >= 48:
                break

    ok = ok_gt = 0
    len_err = []
    t0 = time.time()
    for ext, field, xy, goal_xy in legs:
        # obstacle channel is ext[1] before flattening; reconstruct from ext
        obstacle = ext[0].astype(bool)
        x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = predict_field(model, name, x, G)[0].cpu().numpy().reshape(G, G)
        p_pred = field_path(pred, xy, goal_xy, obstacle)
        p_gt = field_path(field, xy, goal_xy, obstacle)
        ok += int(p_pred is not None)
        ok_gt += int(p_gt is not None)
        if p_pred is not None and p_gt is not None:
            ln = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
            len_err.append(ln(p_pred) / max(ln(p_gt), 1e-9))
    dt = (time.time() - t0) / len(legs)
    print(f"{os.path.basename(ckpt)} {name}: descent ok {ok}/{len(legs)} "
          f"(gt {ok_gt}/{len(legs)})  len/gt "
          f"{np.mean(len_err):.3f}+-{np.std(len_err):.3f}  {dt*1e3:.0f} ms/leg")


if __name__ == "__main__":
    main()
