"""Trainable spatial adapter: SigLIP patch tokens -> 32x32 task grid.

Also predicts the inflated obstacle occupancy used by the NCA.  The obstacle
head is the only source of obstacle information at inference; GT obstacle is
reserved for the upper-bound diagnostic arm.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1 = nn.Conv2d(d, d, 3, padding=1)
        self.n1 = nn.GroupNorm(8, d)
        self.c2 = nn.Conv2d(d, d, 3, padding=1)
        self.n2 = nn.GroupNorm(8, d)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class SpatialAdapter(nn.Module):
    def __init__(self, vis_dim=768, d=256, out_ch=64):
        super().__init__()
        self.vis_dim = vis_dim
        self.out_ch = out_ch
        self.inp = nn.Conv2d(vis_dim, d, 1)
        self.block = ResBlock(d)
        self.out = nn.Conv2d(d, out_ch, 1)
        self.obs = nn.Conv2d(d, 1, 1)

    def forward(self, tokens, size=32):
        b, n, d = tokens.shape
        h = math.isqrt(n)
        if h * h != n:
            raise ValueError(f"token count {n} is not a square grid")
        x = tokens.transpose(1, 2).reshape(b, d, h, h)
        x = self.inp(x)
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False)
        x = self.block(x)
        return self.out(x), self.obs(x)
