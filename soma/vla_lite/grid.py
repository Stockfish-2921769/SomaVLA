"""Workspace <-> grid mapping shared by the VLA-lite modules.

Matches scripts/multi_task.py: x grows with column j, y decreases with row i.
"""
import torch

XL, XH, YL, YH = 0.22, 0.74, 0.22, 0.60


def xy_to_cell(xy, g):
    """xy [B,2] -> integer cell (row i, col j), both [B]."""
    j = ((xy[:, 0] - XL) / (XH - XL) * g).long().clamp(0, g - 1)
    i = ((YH - xy[:, 1]) / (YH - YL) * g).long().clamp(0, g - 1)
    return i, j


def cell_to_xy(i, j, g):
    x = XL + (j.float() + 0.5) / g * (XH - XL)
    y = YH - (i.float() + 0.5) / g * (YH - YL)
    return torch.stack([x, y], dim=-1)


def goal_start_offsets(goal_xy, start_xy, g, device=None, dtype=None):
    """Per-cell relative offsets used as NCA external channels.

    Returns (goal_dx, goal_dy, start_dx, start_dy), each [B,1,H,W] on the
    same device/dtype as the inputs.  Channel order matches E0's
    [dx/G, dy/G] convention (x offset first).
    """
    device = device or goal_xy.device
    dtype = dtype or goal_xy.dtype
    gi, gj = xy_to_cell(goal_xy, g)
    si, sj = xy_to_cell(start_xy, g)
    ii = torch.arange(g, device=device).view(1, g, 1)
    jj = torch.arange(g, device=device).view(1, 1, g)
    b = goal_xy.shape[0]
    gdx = ((gj.view(b, 1, 1).to(dtype) - jj.to(dtype)) / g).expand(b, g, g)
    gdy = ((gi.view(b, 1, 1).to(dtype) - ii.to(dtype)) / g).expand(b, g, g)
    sdx = ((sj.view(b, 1, 1).to(dtype) - jj.to(dtype)) / g).expand(b, g, g)
    sdy = ((si.view(b, 1, 1).to(dtype) - ii.to(dtype)) / g).expand(b, g, g)
    return (gdx.unsqueeze(1), gdy.unsqueeze(1),
            sdx.unsqueeze(1), sdy.unsqueeze(1))
