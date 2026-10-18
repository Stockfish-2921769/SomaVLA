#!/usr/bin/env python3
"""Render a scene, GT/NCA plan fields, ordering diagnostic, and a short GIF."""
import os
import sys

sys.path.insert(0, "/home/ubuntu/Soma")
sys.path.insert(0, "/home/ubuntu/Soma/scripts")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from multi_task import (sample_scene, decision_samples, G, XL, XH, YL, YH,
                        xy_to_cell, cell_to_xy, DIRS, render_scene)
from multi_plan_data import leg_sample, field_path_escape, EXT_DIM
from soma.spatial_baselines import build_model, predict_field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.environ.get("VIZ_OUT", os.path.join(ROOT, "assets", "plan_field"))
os.makedirs(OUT, exist_ok=True)


def plen(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(777)
    ck = os.environ.get("PLAN_CKPT", "/tmp/plan_nca_long3r.pt")
    d = torch.load(ck, map_location=dev, weights_only=False)
    model = build_model(d["model_name"], **d["arch"]).to(dev)
    model.load_state_dict(d["model"]); model.eval()

    legs = []
    while len(legs) < 48:
        objs, tgts, perm, walls = sample_scene(rng, two_walls=True)
        for xy, grip, objs_now, placed, active, cmd in decision_samples(
                objs, tgts, perm):
            s = leg_sample(objs_now, xy, cmd[:2], walls, placed, active)
            if s is None or s[2].sum() < 2:
                continue
            legs.append(s + (xy, cmd[:2], objs_now, placed, tgts, perm, walls))
            if len(legs) >= 48:
                break

    best = None
    for leg in legs:
        ext, field, valid, xy, gy, objs_now, placed, tgts, perm, walls = leg
        ob = ext[0].astype(bool)
        p_gt = field_path_escape(field, xy, gy, ob, radius=24)
        if p_gt is None:
            continue
        detour = plen(p_gt) / max(np.linalg.norm(gy - xy), 1e-9)
        if best is None or detour > best[0]:
            best = (detour, leg, p_gt, ob)
    detour, leg, p_gt, ob = best
    ext, field, valid, xy, gy, objs_now, placed, tgts, perm, walls = leg

    x = torch.tensor(ext.reshape(EXT_DIM, -1).T[None], dtype=torch.float32,
                     device=dev)
    with torch.no_grad():
        pred = predict_field(model, d["model_name"], x, G)[0].cpu().numpy()
    pred = pred.reshape(G, G)
    p_nca = field_path_escape(pred, xy, gy, ob, radius=24)
    print(f"selected leg detour {detour:.2f}x  gt {p_gt.shape[0]} pts  "
          f"nca {'ok' if p_nca is not None else 'FAIL'}")

    img = render_scene(objs_now, tgts, perm, placed, walls, xy)
    img.save(f"{OUT}/scene.png")

    extent = [XL, XH, YL, YH]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(ob, cmap="Greys", origin="upper", extent=extent, vmin=0, vmax=1)
    axs[0].set_title("obstacle grid (inflated)")
    for i in range(len(objs_now)):
        if placed[i]:
            continue
        axs[0].plot(*objs_now[i], "o", ms=6, label=f"obj{i}")
    axs[0].plot(*gy, "k*", ms=14)
    axs[0].plot(*xy, "bs", ms=7)
    axs[0].legend(fontsize=6, loc="upper right")

    for ax, f, title, p in (
            (axs[1], field, "GT geodesic field", p_gt),
            (axs[2], pred, "NCA predicted field", p_nca)):
        m = np.ma.masked_where(ob, f)
        im = ax.imshow(m, cmap="viridis", origin="upper", extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.plot(*xy, "bs", ms=7, label="start")
        ax.plot(*gy, "k*", ms=14, label="goal")
        if p is not None:
            ax.plot(p[:, 0], p[:, 1], "r-", lw=2, label="escape path")
        ax.legend(fontsize=7, loc="upper right")
    for ax in axs:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fields.png", dpi=140)
    plt.close(fig)

    # ordering diagnostic along the GT-descent path
    sc, gc = xy_to_cell(xy), xy_to_cell(gy)
    cur, vals, gvals, viol = sc, [pred[sc]], [field[sc]], []
    seen = set()
    while cur != gc and cur not in seen:
        seen.add(cur)
        best_nb, bv = None, field[cur]
        for di, dj, _c in DIRS:
            nb = (cur[0] + di, cur[1] + dj)
            if (0 <= nb[0] < G and 0 <= nb[1] < G and not ob[nb]
                    and field[nb] < bv - 1e-9):
                bv, best_nb = field[nb], nb
        if best_nb is None:
            break
        vals.append(pred[best_nb]); gvals.append(field[best_nb])
        viol.append(pred[best_nb] - pred[cur] >= -1e-6)
        cur = best_nb
    vals, gvals, viol = np.array(vals), np.array(gvals), np.array(viol)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(gvals * 128, "k-", label="GT distance (cells)")
    ax1.set_xlabel("step along GT path"); ax1.set_ylabel("GT distance")
    ax2 = ax1.twinx()
    ax2.plot(vals, color="tab:blue", label="NCA predicted value")
    vi = np.where(viol)[0] + 1          # vals[0] is the start cell
    ax2.plot(vi, vals[vi], "rx", ms=9, label="sign violation")
    ax2.set_ylabel("NCA field value")
    l1, n1 = ax1.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper right")
    ax1.set_title(f"GT-path ordering: {viol.sum()}/{len(viol)} violation steps")
    fig.tight_layout(); fig.savefig(f"{OUT}/ordering.png", dpi=140)
    plt.close(fig)
    print(f"ordering violations {viol.sum()}/{len(viol)}")

    # GIF: damped plant following the NCA escape path
    if p_nca is not None:
        frames = []
        eef = np.array(xy, np.float32)
        pts = p_nca
        for k in range(len(pts) - 1):
            for _ in range(6):
                eef = eef + 0.5 * (pts[k + 1] - eef)
                im = render_scene(objs_now, tgts, perm, placed, walls, eef)
                frames.append(im.resize((256, 256), Image.BILINEAR))
        for _ in range(20):
            frames.append(frames[-1])
        frames[0].save(f"{OUT}/executor.gif", save_all=True,
                       append_images=frames[1:], duration=60, loop=0)
        print(f"gif frames {len(frames)}")


if __name__ == "__main__":
    main()
