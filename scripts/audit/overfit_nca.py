#!/usr/bin/env python3
"""Can the NCA architecture represent a geodesic field near-perfectly?
Train on 8 fixed legs (memorization) and evaluate ordering on the same 8
plus 8 held-out legs from the same distribution."""
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import numpy as np
import torch

from multi_task import sample_scene, decision_samples, G, xy_to_cell, DIRS
from multi_plan_data import leg_sample, field_path, field_path_escape, EXT_DIM
from soma.spatial_baselines import build_model, predict_field
from soma.spatial_nca import grid_neighbors


def violations(pred, tar, ob, sc, gc):
    """Fraction of GT-descent steps where pred does not decrease."""
    cur = sc
    seen, sign = set(), []
    while cur != gc and cur not in seen:
        seen.add(cur)
        best, bv = None, tar[cur]
        for di, dj, _c in DIRS:
            nb = (cur[0] + di, cur[1] + dj)
            if (0 <= nb[0] < G and 0 <= nb[1] < G and not ob[nb]
                    and tar[nb] < bv - 1e-9):
                bv, best = tar[nb], nb
        if best is None:
            break
        sign.append(pred[best] - pred[cur] < -1e-6)
        cur = best
    return 1.0 - float(np.mean(sign)) if sign else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--morph", type=int, default=12)
    ap.add_argument("--msg", type=int, default=48)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--wall-weight", type=float, default=0.0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    rng = np.random.RandomState(777)
    legs = []
    while len(legs) < 16:
        objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(
                objs, tgts, perm):
            s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active)
            if s is None or s[2].sum() < 2:
                continue
            legs.append(s + (xy, cmd[:2]))
            if len(legs) >= 16:
                break
    train, test = legs[:8], legs[8:]

    def pack(ls):
        ext = torch.tensor(np.stack([e.reshape(EXT_DIM, -1).T for e, f, v, x, y in ls]),
                           dtype=torch.float32, device=dev)
        tar = torch.tensor(np.stack([f.reshape(-1) for e, f, v, x, y in ls]),
                           dtype=torch.float32, device=dev)
        val = torch.tensor(np.stack([v.reshape(-1) for e, f, v, x, y in ls]),
                           dtype=torch.bool, device=dev)
        return ext, tar, val

    kv = {"k_steps": 64, "hidden": args.hidden,
          "morphogen_dim": args.morph, "msg_dim": args.msg,
          "ext_dim": EXT_DIM, "obstacle_ch": 0,
          "mask_obstacle_msgs": args.mask}
    model = build_model("nca", **kv).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())}")
    nb_idx, nb_valid = grid_neighbors(G, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    xt, tt, vt = pack(train)
    xh, th, vh = pack(test)

    def weight_of(x):
        """1 + lambda on cells within 2 (Chebyshev) of an obstacle."""
        ob = x[..., 0].reshape(-1, 1, G, G)
        near = torch.nn.functional.max_pool2d(ob, 5, 1, 2)
        return 1.0 + args.wall_weight * near.reshape(x.shape[0], -1)

    def report(step):
        out = []
        for tag, (x, t, v) in (("train", (xt, tt, vt)), ("held", (xh, th, vh))):
            with torch.no_grad():
                p = model(x, nb_idx, nb_valid)
                mse = (((p - t) ** 2) * v).sum() / v.sum()
                pn = p.cpu().numpy().reshape(-1, G, G)
                tn = t.cpu().numpy().reshape(-1, G, G)
            vios, strict, esc = [], 0, 0
            for i, (ext, field, valid, xy, gy) in enumerate(train if tag == "train" else test):
                ob = ext[0].astype(bool)
                pg = pn[i].reshape(G, G)
                vios.append(violations(pg, tn[i], ob, xy_to_cell(xy),
                                       xy_to_cell(gy)))
                strict += int(field_path(pg, xy, gy, ob) is not None)
                esc += int(field_path_escape(pg, xy, gy, ob, radius=24) is not None)
            out.append(f"{tag} mse {mse.item():.4f} viol {np.nanmean(vios):.3f} "
                       f"strict {strict}/8 esc {esc}/8")
        print(f"step {step:4d}  " + "  |  ".join(out), flush=True)

    report(0)
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        p = model(xt, nb_idx, nb_valid)
        w = weight_of(xt) * vt
        loss = (((p - tt) ** 2) * w).sum() / w.sum()
        loss.backward(); opt.step()
        if step % 400 == 0:
            report(step)


if __name__ == "__main__":
    main()
