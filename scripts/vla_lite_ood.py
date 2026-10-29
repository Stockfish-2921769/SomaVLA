"""O2 obstacle-complexity OOD scenes for VLA-lite (task book 4.4).

Training only ever sees `sample_scene(two_walls=True)`: two full-span walls,
each with one gap (4 wall rectangles). The O2 generator keeps that base and
adds `n_extra` free-standing wall segments, so scenes have 4+n_extra
rectangles and routes must thread more, unseen blockers. Physics, rendering,
teacher and success criterion are unchanged.
"""
import numpy as np

from multi_task import (sample_scene, _min_sep_pts, _pt_wall_dist,
                        _scene_reachable, CENTER, MIN_SEP_OBJ, MIN_SEP_TGT,
                        N_OBJ, WALL_CLEAR, WALL_T, XL, XH, YL, YH)

SEG_LEN = (0.08, 0.16)       # extra segment length (m)


def _extra_segment(rng):
    length = rng.uniform(*SEG_LEN)
    cx = rng.uniform(XL + 0.05, XH - 0.05)
    cy = rng.uniform(YL + 0.05, YH - 0.05)
    if rng.rand() < 0.5:     # vertical
        return (cx - WALL_T / 2, cx + WALL_T / 2,
                cy - length / 2, cy + length / 2)
    return (cx - length / 2, cx + length / 2,
            cy - WALL_T / 2, cy + WALL_T / 2)


def sample_scene_multi_wall(rng, n_extra_range=(2, 6), n=N_OBJ, tries=300):
    """Returns (objs, targets, perm, walls) like `sample_scene`, with
    n_extra ~ U{lo..hi} extra wall segments on top of the two-wall base."""
    lo, hi = n_extra_range
    for _attempt in range(tries):
        _, _, _, walls = sample_scene(rng, n=n, two_walls=True)
        walls = list(walls)
        n_extra = rng.randint(lo, hi + 1)
        for _ in range(n_extra):
            for _seg_try in range(50):
                seg = _extra_segment(rng)
                if _pt_wall_dist(CENTER, walls + [seg]) >= WALL_CLEAR:
                    walls.append(seg)
                    break
        objs = _min_sep_pts(rng, n, [], MIN_SEP_OBJ, walls)
        tgts = _min_sep_pts(rng, n, objs, MIN_SEP_TGT, walls)
        if any(p is None for p in objs + tgts):
            continue
        if not _scene_reachable(objs, tgts, walls):
            continue
        perm = rng.permutation(n)
        return (np.array(objs, np.float32), np.array(tgts, np.float32), perm,
                walls)
    raise RuntimeError("sample_scene_multi_wall: no reachable layout found")
