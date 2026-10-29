"""Grid-agnostic neural cellular automaton planner.

State per cell: channel 0 = plan scalar, channels 1..C-1 = latent workspace.
The same 3x3 rule is applied K times; obstacle cells are frozen via alpha.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import goal_start_offsets


class NCA(nn.Module):
    def __init__(self, channels=32, vision_ch=64, task_dim=8, hidden=96,
                 k_steps=12, dt=0.5):
        super().__init__()
        self.channels = channels
        self.vision_ch = vision_ch
        self.task_dim = task_dim
        self.ext_dim = 1 + 2 + 2 + vision_ch + task_dim
        self.k_steps = k_steps
        self.dt = dt
        self.rule = nn.Sequential(
            nn.Conv2d(channels + self.ext_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.rule[-1].weight)
        nn.init.zeros_(self.rule[-1].bias)

    def make_ext(self, vision, obstacle_prob, goal_xy, start_xy, task_emb):
        """vision [B,Cv,H,W], obstacle_prob [B,1,H,W], xy [B,2],
        task_emb [B,T] -> ext [B,E,H,W]."""
        b, _, h, w = vision.shape
        gdx, gdy, sdx, sdy = goal_start_offsets(
            goal_xy, start_xy, h, device=vision.device, dtype=vision.dtype)
        task = task_emb.to(vision.dtype).view(b, self.task_dim, 1, 1)
        task = task.expand(b, self.task_dim, h, w)
        return torch.cat([obstacle_prob, gdx, gdy, sdx, sdy, vision, task],
                         dim=1)

    def forward(self, ext, k_steps=None, return_delta=False):
        b, _, h, w = ext.shape
        k = self.k_steps if k_steps is None else k_steps
        alpha = 1.0 - ext[:, :1]
        state = ext.new_zeros(b, self.channels, h, w)
        delta_last = None
        for _ in range(k):
            d = self.rule(torch.cat([state, ext], dim=1))
            d = self.dt * torch.tanh(d) * alpha
            state = state + d
            delta_last = d[:, 0]
        plan = state[:, 0]
        if return_delta:
            return state, plan, delta_last
        return state, plan


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
