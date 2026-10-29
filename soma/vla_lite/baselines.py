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
    """Feed-forward control: 12 3x3 layers, same nominal width as NCA."""

    def __init__(self, ext_dim=77, hidden=32, layers=12):
        super().__init__()
        self.inp = nn.Conv2d(ext_dim, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden), nn.GELU())
            for _ in range(layers)])
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ext):
        x = F.gelu(self.inp(ext))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


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
