#!/usr/bin/env python3
"""Scene + ground-truth generator for the Tier-1 spatial plan-field task.

Task: cluttered reach planning on a G x G grid. Random obstacle blocks, one
goal cell, one start cell. Ground truth = the geodesic (Dijkstra, 8-connected)
distance field to the goal, normalized to [0, 1]. Supervision is the classical
solver — pure regression, no RL.

Scene families:
  * "random"  — k random rectangles + scattered blocks (density via k).
  * "u_wall"  — a single-gap wall between start and goal so the EUCLIDEAN
                distance is tiny but the GEODESIC distance is ~O(G): the
                long-range-propagation probe (a fixed-depth CNN is blind to it).

Also exposes `make_batch`, which packs a batch of scenes into the ext tensor
[B, N, 3] = [goal, obstacle, start] plus target field and validity mask.
"""
import heapq
import os
import sys

import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIRS = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356)]


def geodesic_field(obstacle, goal):
    """Dijkstra (8-connected, diagonal cost sqrt2) from `goal` (a single cell or
    an iterable of goal cells — multi-source for the multi-goal probe). Returns
    (dist [G,G] float, reachable [G,G] bool). Blocked cells are inf/unreachable."""
    g = obstacle.shape[0]
    goals = [goal] if (isinstance(goal, tuple) and len(goal) == 2
                       and np.isscalar(goal[0])) else list(goal)
    dist = np.full((g, g), np.inf)
    pq = []
    for gi, gj in goals:
        if not obstacle[gi, gj] and dist[gi, gj] > 0.0:
            dist[gi, gj] = 0.0
            heapq.heappush(pq, (0.0, gi, gj))
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, c in DIRS:
            ni, nj = i + di, j + dj
            if 0 <= ni < g and 0 <= nj < g and not obstacle[ni, nj]:
                nd = d + c
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
    return dist, np.isfinite(dist)


def _random_scene(g, rng, n_obs):
    obstacle = np.zeros((g, g), dtype=bool)
    kinds = rng.randint(0, 3, size=max(n_obs, 1))
    for k in kinds[:n_obs]:
        if k == 0:                                   # small square
            h = w = rng.randint(1, 3)
        elif k == 1:                                 # thin wall segment
            h, w = (rng.randint(3, max(4, g // 3)), 1) if rng.rand() < 0.5 \
                else (1, rng.randint(3, max(4, g // 3)))
        else:                                        # block
            h, w = rng.randint(2, 4), rng.randint(2, 4)
        i = rng.randint(0, max(1, g - h))
        j = rng.randint(0, max(1, g - w))
        obstacle[i:i + h, j:j + w] = True
    return obstacle


def _free_cells(obstacle):
    free = ~obstacle
    idx = np.argwhere(free)
    return idx


def _random_start_goal(obstacle, rng, min_sep=0.35):
    g = obstacle.shape[0]
    free = _free_cells(obstacle)
    if len(free) < 2:
        return None
    for _ in range(50):
        a = free[rng.randint(len(free))]
        b = free[rng.randint(len(free))]
        if np.hypot(*(a - b)) >= min_sep * g:
            return tuple(a), tuple(b)
    a, b = free[0], free[-1]
    return tuple(a), tuple(b)


def wall_scene(g, rng):
    """A single-gap wall (random orientation, gap at either end) with start and
    goal straddling it at the FAR end from the gap, so the euclidean distance is
    ~g//2 but the geodesic distance is ~2g: the long-range-propagation probe."""
    obstacle = np.zeros((g, g), dtype=bool)
    mid = g // 2
    gap_lo = rng.rand() < 0.5                       # gap at the low or high end
    vertical = rng.rand() < 0.5
    if vertical:
        obstacle[:, mid] = True
        obstacle[0 if gap_lo else g - 1, mid] = False
        far = g - 1 if gap_lo else 0
        start, goal = (far, mid - 2), (far, mid + 2)
    else:
        obstacle[mid, :] = True
        obstacle[mid, 0 if gap_lo else g - 1] = False
        far = g - 1 if gap_lo else 0
        start, goal = (mid - 2, far), (mid + 2, far)
    for _ in range(rng.randint(0, 3)):              # a few extra blocks to vary
        i, j = rng.randint(0, g), rng.randint(0, g)
        h, w = rng.randint(1, 3), rng.randint(1, 3)
        obstacle[i:i + h, j:j + w] = True
    obstacle[start] = False
    obstacle[goal] = False
    return obstacle, start, goal


u_wall_scene = wall_scene


def _multi_goal_scene(g, rng, n_obs, n_goals, min_sep=0.25):
    """Random cluttered scene with `n_goals` well-separated free goal cells and
    one start. Returns (obstacle, start, [goal cells])."""
    obstacle = _random_scene(g, rng, n_obs)
    for _ in range(50):
        free = _free_cells(obstacle)
        if len(free) < n_goals + 1:
            obstacle = _random_scene(g, rng, n_obs)
            continue
        picks = free[rng.choice(len(free), min(len(free), n_goals + 1),
                                replace=False)]
        ok = True
        for a in range(len(picks)):
            for b in range(a + 1, len(picks)):
                if np.hypot(*(picks[a] - picks[b])) < min_sep * g:
                    ok = False
        if ok:
            start = tuple(picks[0])
            goals = [tuple(p) for p in picks[1:]]
            return obstacle, start, goals
    free = _free_cells(obstacle)
    idx = rng.choice(len(free), n_goals + 1, replace=False)
    return obstacle, tuple(free[idx[0]]), [tuple(free[i]) for i in idx[1:]]


def sample_scene(g, rng, family="random", obs_range=(0, 3), n_goals=1):
    if n_goals > 1:                                 # multi-goal (random only)
        n_obs = rng.randint(obs_range[0], obs_range[1] + 1)
        return _multi_goal_scene(g, rng, n_obs, n_goals)
    if family in ("u_wall", "wall"):
        return wall_scene(g, rng)
    if family == "mixed":
        if rng.rand() < 0.5:
            return wall_scene(g, rng)
        family = "random"
    n_obs = rng.randint(obs_range[0], obs_range[1] + 1)
    obstacle = _random_scene(g, rng, n_obs)
    sg = _random_start_goal(obstacle, rng)
    if sg is None:
        return np.zeros((g, g), dtype=bool), (0, 0), (g - 1, g - 1)
    start, goal = sg
    return obstacle, start, goal


def scene_to_targets(obstacle, start, goal, norm_c=None):
    """Returns (ext [3,G,G] float, target [G,G] float, valid [G,G] bool).

    norm_c: if given, target = geodesic distance / norm_c with a FIXED constant
    (scale-consistent across grid sizes — the value at a cell depends only on
    its distance to the goal, not on the scene's max distance). If None, divide
    by the per-scene max finite distance (the classic [0,1] normalization;
    note this makes the target depend on a global quantity a net cannot see,
    which is why it does not transfer across G)."""
    g = obstacle.shape[0]
    dist, reach = geodesic_field(obstacle, goal)
    finite = np.isfinite(dist)
    denom = norm_c if norm_c is not None else (dist[finite].max() if finite.any() else 1.0)
    norm = np.where(finite, dist / (denom + 1e-9), 1.0)
    ext = np.zeros((3, g, g), dtype=np.float32)
    goals = [goal] if (isinstance(goal, tuple) and len(goal) == 2
                       and np.isscalar(goal[0])) else list(goal)
    for gi, gj in goals:
        ext[0][gi, gj] = 1.0
    ext[1][obstacle] = 1.0
    ext[2][start] = 1.0
    valid = reach & (~obstacle)
    return ext, norm.astype(np.float32), valid


# ---------------------------------------------------------------------------
# Capability probe 2: harmonic / diffusion potential field (a physical field)
# ---------------------------------------------------------------------------

def harmonic_field(obstacle, goal, iters=None):
    """Steady state of ∇²φ=0 on free cells with Dirichlet φ=0 at `goal` and
    φ=1 at obstacles. Goal is the unique minimum, so descent reaches it. This is
    a *physical* field: the electrostatics / steady diffusion potential. The
    steady state is the fixed point of iterative relaxation (the NCA's native
    computation); we obtain it to machine precision via a direct sparse solve of
    the discrete Laplace system (far faster than Jacobi, identical result)."""
    g = obstacle.shape[0]
    val = np.where(obstacle, 1.0, 0.0).astype(np.float64)
    goals = [goal] if (isinstance(goal, tuple) and len(goal) == 2
                       and np.isscalar(goal[0])) else list(goal)
    fix = obstacle.copy()
    for gi, gj in goals:
        fix[gi, gj] = True
        val[gi, gj] = 0.0
    free = np.argwhere(~fix)
    n = len(free)
    if n == 0:
        return val
    idx = -np.ones((g, g), dtype=np.int64)
    idx[~fix] = np.arange(n)
    rows, cols, data, b = [], [], [], np.zeros(n)
    for r, (i, j) in enumerate(free):
        deg = 0
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < g and 0 <= nj < g:
                deg += 1
                if idx[ni, nj] >= 0:
                    rows.append(r); cols.append(idx[ni, nj]); data.append(-1.0)
                else:
                    b[r] += val[ni, nj]
        rows.append(r); cols.append(r); data.append(float(deg))
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    x = sparse.linalg.spsolve(A, b)
    phi = val.copy()
    phi[~fix] = x
    return phi


def scene_to_harmonic_targets(obstacle, goal, start):
    g = obstacle.shape[0]
    phi = harmonic_field(obstacle, goal)
    ext = np.zeros((3, g, g), dtype=np.float32)
    ext[0][goal] = 1.0
    ext[1][obstacle] = 1.0
    ext[2][start] = 1.0
    return ext, phi.astype(np.float32), (~obstacle)


# ---------------------------------------------------------------------------
# Capability probe 3: moving-obstacle time-to-contact field (dynamics)
# ---------------------------------------------------------------------------

def _rasterize(blocks, g):
    occ = np.zeros((g, g), dtype=bool)
    for i, j, h, w, _, _ in blocks:
        occ[i:i + h, j:j + w] = True
    return occ


def _step_blocks(blocks, g):
    for b in blocks:
        i, j, h, w, vi, vj = b
        i, j = i + vi, j + vj
        if i < 0 or i + h > g:                      # bounce off the wall
            vi = -vi
            i = min(max(i, 0), g - h)
        if j < 0 or j + w > g:
            vj = -vj
            j = min(max(j, 0), g - w)
        b[:] = [i, j, h, w, vi, vj]


def ttc_scene_targets(g, rng, n_obs=2, horizon=None):
    """`n_obs` moving blocks (integer velocity, wall-bouncing). ext [3,G,G] =
    [obstacle_now, vx, vy]; target = normalized first-arrival time of any block
    at each cell (1.0 = never within the horizon). Valid = free at t=0."""
    horizon = horizon or 2 * g
    blocks = []
    for _ in range(n_obs):
        h, w = rng.randint(2, 4), rng.randint(2, 4)
        i, j = rng.randint(0, max(1, g - h)), rng.randint(0, max(1, g - w))
        vi, vj = rng.randint(-1, 2), rng.randint(-1, 2)
        if vi == 0 and vj == 0:
            vi = 1
        blocks.append([i, j, h, w, vi, vj])
    occ0 = _rasterize(blocks, g)
    vx = np.zeros((g, g), np.float32)
    vy = np.zeros((g, g), np.float32)
    for i, j, h, w, vi, vj in blocks:
        vx[i:i + h, j:j + w] = vi
        vy[i:i + h, j:j + w] = vj
    ttc = np.full((g, g), np.inf)
    for t in range(1, horizon + 1):
        _step_blocks(blocks, g)
        occ = _rasterize(blocks, g)
        ttc = np.minimum(ttc, np.where(occ, float(t), np.inf))
    tar = np.where(np.isfinite(ttc), ttc / horizon, 1.0).astype(np.float32)
    ext = np.stack([occ0.astype(np.float32), vx, vy])
    return ext, tar, (~occ0)


def make_batch(g, batch, rng, family="random", obs_range=(0, 3), device="cpu",
               norm_c=None, n_goals=1, target="geodesic", horizon=None):
    """Packs a batch of scenes. Returns (ext[B,N,3], target[B,N], valid[B,N]).
    target: geodesic (default) | harmonic | ttc."""
    exts, tars, vals = [], [], []
    tries = 0
    while len(exts) < batch and tries < batch * 20:
        tries += 1
        if target == "harmonic":
            obstacle, start, goal = sample_scene(g, rng, family, obs_range, n_goals)
            ext, tar, val = scene_to_harmonic_targets(obstacle, goal, start)
        elif target == "ttc":
            ext, tar, val = ttc_scene_targets(g, rng, rng.randint(obs_range[0], obs_range[1] + 1) + 1, horizon)
        else:
            obstacle, start, goal = sample_scene(g, rng, family, obs_range, n_goals)
            ext, tar, val = scene_to_targets(obstacle, start, goal, norm_c=norm_c)
        if val.sum() < 2:
            continue
        exts.append(ext.reshape(3, -1).T)
        tars.append(tar.reshape(-1))
        vals.append(val.reshape(-1))
    ext = torch.tensor(np.stack(exts), dtype=torch.float32, device=device)
    tar = torch.tensor(np.stack(tars), dtype=torch.float32, device=device)
    val = torch.tensor(np.stack(vals), dtype=torch.bool, device=device)
    return ext, tar, val


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    for fam, g in [("random", 32), ("u_wall", 32), ("random", 64)]:
        ext, tar, val = make_batch(g, 4, rng, family=fam, obs_range=(2, 4))
        print(f"{fam:7s} G={g}: ext{tuple(ext.shape)} target[{tar.min():.2f},"
              f"{tar.max():.2f}] valid_frac={val.float().mean():.3f}")
