"""Spatial-NCA: a homogeneous cellular field over a 2D workspace grid.

This is the "form change" called out in EXPERIMENT_LOG.md (2026-10-13). The
body-graph NCA's cells are 7 heterogeneous semantic DOFs (px, py, pz, rx, ry,
rz, g) — the role of a cell is hard-wired by *which* DOF it is, so the
emergence/self-organization story is not actually supported. Here cells are
*spatial locations*: every cell on a GxG grid shares one rule and one state
layout, and the network iterates that rule to propagate a plan field.

  * Identical shared local rule on every cell — no type embedding, no
    positional parameters. The only thing that distinguishes cells is the
    external input e_i (goal / obstacle / start) and their neighbors.
  * Three-channel cell state [alpha | plan | morphogen(8)]:
      alpha      activity gate = free-space mask (1 free, 0 obstacle). Frozen
                 cells do not update, so obstacles are hard walls for the field.
      plan       the read-out field (predicted normalized geodesic distance).
      morphogen  cross-iteration scratch memory (warm-started across the K
                 relaxation steps; kept for symmetry with the body-graph NCA).
  * External input e_i (ext_dim channels, task-defined; e.g. goal one-hot,
    obstacle, goal offsets) broadcast per cell.
  * tanh-bounded additive updates + zero-initialized final projection layer.
  * 8-neighbour aggregation; parameters are INDEPENDENT of the grid size G, so
    the same weights run on G=32 and G=64 (the scale-generalization axis).

Execution model: given a scene, initialise plan=morphogen=0, run K relaxation
iterations of the shared rule, read out the plan channel. Iterative local
propagation gives an *unbounded* receptive field with a *fixed* parameter
count — the property a fixed-depth CNN does not have.
"""

import torch
import torch.nn as nn

MORPHOGEN_DIM = 8
EXT_DIM = 3                       # goal, obstacle, start
CELL_STATE_DIM = 1 + 1 + MORPHOGEN_DIM   # alpha + plan + morphogen

# 8-neighbourhood offsets (row, col): 4 orthogonal + 4 diagonal.
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))


def grid_neighbors(g, device="cpu"):
    """Return (nb_idx [N,8] long, nb_valid [N,8] float) for a g x g grid.

    Invalid (out-of-bounds) neighbours are marked with idx 0 and valid 0, so
    they contribute nothing to the degree-normalized aggregate.
    """
    n = g * g
    nb = torch.zeros(n, 8, dtype=torch.long)
    valid = torch.zeros(n, 8, dtype=torch.float32)
    for i in range(g):
        for j in range(g):
            c = i * g + j
            for k, (di, dj) in enumerate(OFFSETS):
                ni, nj = i + di, j + dj
                if 0 <= ni < g and 0 <= nj < g:
                    nb[c, k] = ni * g + nj
                    valid[c, k] = 1.0
    return nb.to(device), valid.to(device)


class _Perception(nn.Module):
    """g(s_j, e_j): per-cell (state, input) -> message. Shared across cells."""
    def __init__(self, in_dim=CELL_STATE_DIM + EXT_DIM, out_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())

    def forward(self, x):          # [B, N, in_dim]
        return self.net(x)         # [B, N, 40]


class _Update(nn.Module):
    """f(own, agg, ext): writes plan + morphogen. Final layer zero-init,
    tanh-bounded (stable identity start)."""
    def __init__(self, in_dim, hidden=96, out_dim=1 + MORPHOGEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.tanh(self.net(x))


class SpatialNCA(nn.Module):
    def __init__(self, k_steps=16, dt=0.2, morphogen_dim=MORPHOGEN_DIM,
                 hidden=96, msg_dim=40, ext_dim=EXT_DIM, obstacle_ch=1,
                 mask_obstacle_msgs=False):
        super().__init__()
        self.k_steps = k_steps
        self.dt = dt
        self.morphogen_dim = morphogen_dim
        self.ext_dim = ext_dim
        self.obstacle_ch = obstacle_ch     # which ext channel gates frozen cells
        self.mask_obstacle_msgs = mask_obstacle_msgs

        # state = [alpha(1) | plan(1) | morphogen(morphogen_dim)]; dims follow the
        # constructor arg (morphogen_dim=8 reproduces the original 13/9 layout).
        cell_state_dim = 2 + morphogen_dim
        self.perception = _Perception(in_dim=cell_state_dim + ext_dim, out_dim=msg_dim)
        update_in = cell_state_dim + msg_dim + ext_dim
        self.update = _Update(update_in, hidden=hidden, out_dim=1 + morphogen_dim)

    def _aggregate(self, msg, nb_idx, nb_valid, alpha=None):
        """Degree-normalized mean of the 8 neighbour messages (sparse gather).

        With mask_obstacle_msgs, frozen (obstacle) neighbours contribute no
        message at all, so a wall is a hard barrier for information flow and
        not just a cell that stops updating."""
        b, n, c = msg.shape
        nb_msg = msg.gather(1, nb_idx.reshape(1, -1, 1).expand(b, -1, c))
        nb_msg = nb_msg.reshape(b, n, 8, c)
        valid = nb_valid[None, :, :]
        if self.mask_obstacle_msgs and alpha is not None:
            idx = nb_idx.reshape(1, -1, 8).expand(b, -1, 8)
            nb_alpha = alpha.expand(b, -1, 8).gather(1, idx)
            valid = valid * nb_alpha
        agg = (nb_msg * valid[..., None]).sum(dim=2)
        cnt = valid.sum(dim=2).clamp(min=1.0)[..., None]
        return agg / cnt

    def forward(self, ext, nb_idx, nb_valid, k_steps=None, return_delta=False):
        """ext: [B, N, ext_dim] per cell (channel 1 = obstacle mask).
        Returns plan field [B, N]; if return_delta, also the final step's
        plan update (the fixed-point residual — zero at a true fixed point)."""
        b, n, _ = ext.shape
        dev, dtype = ext.device, ext.dtype
        k = k_steps if k_steps is not None else self.k_steps

        alpha = (1.0 - ext[..., self.obstacle_ch:self.obstacle_ch + 1])  # free mask
        plan = torch.zeros(b, n, 1, device=dev, dtype=dtype)
        morph = torch.zeros(b, n, self.morphogen_dim, device=dev, dtype=dtype)

        delta_last = torch.zeros(b, n, device=dev, dtype=dtype)
        for _ in range(k):
            s = torch.cat([alpha, plan, morph], dim=-1)     # [B,N,10]
            msg = self.perception(torch.cat([s, ext], dim=-1))
            agg = self._aggregate(msg, nb_idx, nb_valid, alpha)
            d = self.update(torch.cat([s, agg, ext], dim=-1))   # [B,N,9] tanh
            dplan, dmorph = d[..., :1], d[..., 1:]
            delta_last = (self.dt * dplan * alpha)[..., 0]
            plan = plan + self.dt * dplan * alpha
            morph = morph + self.dt * dmorph * alpha

        if return_delta:
            return plan[..., 0], delta_last                 # [B,N], [B,N]
        return plan[..., 0]                                 # [B,N]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
