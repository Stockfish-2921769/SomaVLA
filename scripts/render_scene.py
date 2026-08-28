#!/usr/bin/env python3
"""Phase 4f-1 — procedural top-down scene renderer for the SigLIP perception stage.

Maps a pick-and-place scene (obj_xy, place_xy in meters) to a fixed-size RGB
image the SigLIP vision encoder can consume. Ground-truth geometry:

  obj   x in [0.36, 0.48],  y in [0.40, 0.52]
  place x in [0.46, 0.58],  y in [0.34, 0.46]

The render is deterministic in (obj, place); small visual jitter (table shade,
object scale, marker radius, low-pass noise) is added so the perception task
is a robust regression, not an exact-template lookup.

Run from the repo root:
    python scripts/render_scene.py            # dump a 4-panel preview to /tmp
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

# Workspace coverage (meters) — padded beyond the scene distribution so edge
# scenes never clip. x=horizontal, y=vertical.
W_LO, W_HI = 0.32, 0.62
H_LO, H_HI = 0.30, 0.56

OBJ_COLOR = (200, 60, 50)      # red-ish block
PLACE_COLOR = (50, 120, 220)   # blue-ish marker
TABLE_COLOR = (172, 162, 146)  # warm table top


def _px(x, size):
    """Meter x → pixel column."""
    return (x - W_LO) / (W_HI - W_LO) * size


def _py(y, size):
    """Meter y → pixel row (image y flipped vs world y)."""
    return (H_HI - y) / (H_HI - H_LO) * size


def render_scene(obj, place, size=256, rng=None):
    """Render one scene → np.uint8 RGB [size, size, 3].

    obj, place: (x, y) meters. Rendered at 2× then downsampled (box, anti-
    alias) so the 256×256 SigLIP input has smooth edges.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    ss = size * 2  # supersample factor

    # Table background with a subtle per-scene shade variation.
    base = np.array(TABLE_COLOR, dtype=np.float32)
    shade = base + rng.uniform(-8, 8, 3)
    img = Image.new("RGB", (ss, ss), tuple(int(max(0, min(255, c))) for c in shade))
    d = ImageDraw.Draw(img)

    # Object block (red square) centered on obj.
    ox, oy = _px(obj[0], ss), _py(obj[1], ss)
    o_half = rng.uniform(0.012, 0.018)          # meters
    o_half_px = o_half / (W_HI - W_LO) * ss
    d.rectangle([ox - o_half_px, oy - o_half_px, ox + o_half_px, oy + o_half_px],
                fill=OBJ_COLOR)

    # Place marker (blue ring + crosshair) centered on place.
    px_, py_ = _px(place[0], ss), _py(place[1], ss)
    r_in = rng.uniform(0.010, 0.014) / (W_HI - W_LO) * ss
    r_out = r_in + 2.2
    d.ellipse([px_ - r_out, py_ - r_out, px_ + r_out, py_ + r_out],
              outline=PLACE_COLOR, width=3)
    d.ellipse([px_ - r_in, py_ - r_in, px_ + r_in, py_ + r_in],
              outline=PLACE_COLOR, width=2)
    arm = r_in + 5.0
    d.line([px_ - arm, py_, px_ + arm, py_], fill=PLACE_COLOR, width=2)
    d.line([px_, py_ - arm, px_, py_ + arm], fill=PLACE_COLOR, width=2)

    # Workspace border so the frame is visible to the encoder.
    d.rectangle([2, 2, ss - 3, ss - 3], outline=(90, 84, 76), width=4)

    img = img.resize((size, size), Image.BOX)
    arr = np.asarray(img, dtype=np.float32)

    # Low-amplitude sensor noise (helps the regression learn structure).
    arr = arr + rng.normal(0, 1.5, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/render_scene_preview.png")
    args = ap.parse_args()
    rng = np.random.RandomState(7)
    panels = [
        ([0.42, 0.46], [0.52, 0.40]),
        ([0.38, 0.50], [0.55, 0.42]),
        ([0.46, 0.42], [0.49, 0.44]),
        ([0.36, 0.51], [0.58, 0.35]),
    ]
    tiles = [render_scene(o, p, size=256, rng=rng) for o, p in panels]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    Image.fromarray(grid).save(args.out)
    print(f"preview → {args.out}")
