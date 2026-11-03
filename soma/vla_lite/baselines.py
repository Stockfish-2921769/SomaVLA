"""Controlled field sources for the VLA-lite arms."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import xy_to_cell


class EuclidField(nn.Module):
    """Zero-parameter Euclidean potential; deliberately obstacle-blind."""

    def forward(self, goal_xy, g):
        b = goal_xy.shape[0]
        gi, gj = xy_to_cell(goal_xy, g)
        ii = torch.arange(g, device=goal_xy.device).view(1, g, 1).float()
        jj = torch.arange(g, device=goal_xy.device).view(1, 1, g).float()
        dx = (gj.view(b, 1, 1).float() - jj) / g
        dy = (gi.view(b, 1, 1).float() - ii) / g
        plan = torch.sqrt(dx * dx + dy * dy)
        return plan.unsqueeze(1)


class CNNField(nn.Module):
    """Feed-forward control: fixed-depth 3x3 stack, same nominal width as NCA.

    Emits a `channels`-wide state (channel 0 = plan, rest = latent) so the
    decoder sees the same field bandwidth as the NCA arm; otherwise an
    NCA > CNN gap could just be 32 learned channels beating 1 scalar.
    """

    def __init__(self, ext_dim=77, hidden=32, layers=12, channels=32):
        super().__init__()
        self.inp = nn.Conv2d(ext_dim, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden), nn.GELU())
            for _ in range(layers)])
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ext):
        x = F.gelu(self.inp(ext))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


class VINField(nn.Module):
    """Value Iteration Network (Tamar et al., NeurIPS 2016) on the same ext.

    Reward net R = f(ext) (same hidden width as the NCA rule), then K shared
    Bellman backups  Q = conv3x3([R, V]),  V = max_a Q,  V_0 = 0.
    VIN is the closest prior iterative planner to the NCA, so it is the
    baseline a planning reviewer expects. Kept faithful: no obstacle
    freezing, no gating, no normalization (the NCA freezes obstacle cells via
    alpha; VIN must learn walls through R).

    Output is a `1 + n_actions`-channel state [-V, Q]: channel 0 is
    distance-like (low near the goal), matching the GT plan d/(2G) and the
    other arms, so the same plan loss and `--readout plan` apply.
    """

    def __init__(self, ext_dim=77, hidden=96, n_actions=31, k_steps=12):
        super().__init__()
        self.k_steps = k_steps
        self.reward = nn.Sequential(
            nn.Conv2d(ext_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.q = nn.Conv2d(2, n_actions, 3, padding=1)

    def forward(self, ext, k_steps=None):
        k = self.k_steps if k_steps is None else k_steps
        r = self.reward(ext)
        v = torch.zeros_like(r)
        q = self.q(torch.cat([r, v], dim=1))
        for _ in range(k - 1):                     # K backups in total
            v = q.max(dim=1, keepdim=True).values
            q = self.q(torch.cat([r, v], dim=1))
        v = q.max(dim=1, keepdim=True).values
        return torch.cat([-v, q], dim=1)


def random_smooth_field(b, g, device, dtype=torch.float32, sigma=3.0):
    """Smooth low-frequency random field, normalized per sample to [0,1]."""
    z = torch.randn(b, 1, g, g, device=device, dtype=dtype)
    radius = max(1, int(3 * sigma))
    ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    ker = torch.exp(-(ax * ax) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    k = (ker[:, None] * ker[None, :]).view(1, 1, 2 * radius + 1,
                                            2 * radius + 1)
    z = F.conv2d(F.pad(z, (radius,) * 4, mode="reflect"), k)
    lo = z.amin(dim=(2, 3), keepdim=True)
    hi = z.amax(dim=(2, 3), keepdim=True)
    return (z - lo) / (hi - lo + 1e-6)


def zero_field(b, g, device, dtype=torch.float32):
    return torch.zeros(b, 1, g, g, device=device, dtype=dtype)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
