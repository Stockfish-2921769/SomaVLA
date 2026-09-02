#!/usr/bin/env python3
"""Decision B — training data for the SigLIP scene+physics planner.

Renders the abstract workspace as a 256×256 top-down RGB image containing ONLY
the static scene the low-level actuator needs: the object (its SIZE encodes
mass, its SHADE encodes friction mu) and the green place target. No EEF dot —
the planner is a static scene estimator, and the actuator gets EEF pose from
truthful proprioception. The VLA encoder renders the same object encoding, so
the frozen SigLIP features that read margin (probe_margin.py: 0.32N) carry the
physics here too.

Labels = [obj_x, obj_y, place_x, place_y, mu, mass] — exactly the 6-vector the
hierarchical controller feeds in where the NCA baselines receive oracle inputs:
  obj/place → router scene [4] (skill choice + goal xy)
  mu/mass  → the physics modality ctx [P] the aware experts contract on.

Scene distribution matches eval_baseline_hard.cells(): hard-frac 0.5 over the
EVAL m/μ discriminating band [1.16, 1.26], rest broad feasible.

Run from the repo root:
  python scripts/planner_data.py --n 12000 --out data/planner_train.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
from PIL import Image, ImageDraw

import proc_sim
from vla_data import sample_cell, siglip_normalize, XL, XH, YL, YH


def planner_render(obj, place, mu, mass, s=256):
    """Top-down RGB: green place target + object rectangle at obj (no EEF dot).
    obj/place = (x, y) meters; object size ∝ mass, shade ∝ mu (darker = higher
    friction). Returns PIL RGB."""
    def P(p):
        x, y = p
        px = (x - XL) / (XH - XL) * (s - 1)
        py = (YH - y) / (YH - YL) * (s - 1)
        return (int(px), int(py))
    img = Image.new("RGB", (s, s), (235, 232, 225))
    d = ImageDraw.Draw(img)
    for g in range(0, s, 32):
        d.line([(g, 0), (g, s)], fill=(220, 216, 208))
        d.line([(0, g), (s, g)], fill=(220, 216, 208))
    px, py = P(place)
    d.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(40, 160, 80), width=3)
    d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(40, 160, 80))
    side = int(8 + mass / 0.35 * 20)       # 12..28 px, mass 0.05..0.35
    ox, oy = P(obj)
    mu_t = min(max((mu - 0.2) / 0.4, 0.0), 1.0)
    r = int(210 - 70 * mu_t); g = int(60 + 25 * mu_t); b = int(50 + 30 * mu_t)
    d.rectangle([ox - side // 2, oy - side // 2, ox + side // 2, oy + side // 2],
                fill=(r, g, b), outline=(120, 30, 30), width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--band-lo", type=float, default=1.16)
    ap.add_argument("--band-hi", type=float, default=1.26)
    ap.add_argument("--out", type=str, default="data/planner_train.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    images, labels = [], []
    for _ in range(args.n):
        obj, place = proc_sim._sample_scene(rng)
        mu, mass = sample_cell(rng, args.hard_frac, args.band_lo, args.band_hi)
        img = planner_render(obj, place, mu, mass)
        arr = siglip_normalize(np.asarray(img, dtype=np.float32) / 255.0)
        images.append(arr)
        labels.append(np.concatenate([obj, place, [mu, mass]]).astype(np.float32))
    images = np.stack(images).astype(np.float32)
    labels = np.stack(labels).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, images=images, labels=labels)
    print(f"wrote {args.out}: images {images.shape} labels {labels.shape}")
    rat = labels[:, 5] / labels[:, 4]
    print(f"  m/μ range [{rat.min():.2f}, {rat.max():.2f}] (feasible cap "
          f"{0.9 * 15 / 9.81:.2f}); hard band frac "
          f"{(rat >= args.band_lo).mean():.2f}")


if __name__ == "__main__":
    main()
