"""Fair baselines for the spatial plan-field task, matched on parameter budget.

All models take the same per-cell input ext [B, N, in_dim] = [goal, obstacle,
dx, dy, ...] and predict a field [B, N]. They differ only in the inductive bias:

  * cnn         — local, FIXED-DEPTH propagation. Receptive field = 1 + 2*D
                  cells regardless of G. THE key control: it isolates whether
                  *iterative* propagation (unbounded receptive field) buys
                  anything over a fixed-depth local stack.
  * transformer — global attention: unbounded receptive field, but no locality
                  bias and no weight sharing across iterations.
  * mlp         — pointwise (no propagation at all): tests whether propagation
                  is needed, and whether the task is solvable from e_i alone.

SpatialNCA (soma/spatial_nca.py) is the local + iterative-propagation model.
"""

import math

import torch
import torch.nn as nn


class MLPPointwise(nn.Module):
    """No propagation: an independent per-cell MLP on e_i."""
    def __init__(self, hidden=96, in_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ext, g, nb_idx=None, nb_valid=None):
        return self.net(ext)[..., 0]


class CNNBaseline(nn.Module):
    """Fixed-depth local stack; receptive field = 1 + 2*depth cells."""
    def __init__(self, channels=24, depth=4, in_ch=3):
        super().__init__()
        self.depth = depth
        self.in_ch = in_ch
        layers = [nn.Conv2d(in_ch, channels, 3, padding=1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(channels, 1, 1)]
        self.net = nn.Sequential(*layers)

    @property
    def receptive_field(self):
        return 1 + 2 * self.depth

    def forward(self, ext, g, nb_idx=None, nb_valid=None):
        b = ext.shape[0]
        x = ext.transpose(1, 2).reshape(b, self.in_ch, g, g)
        return self.net(x).reshape(b, -1)               # [B,N]


def _pos_enc(g, dim, device, dtype):
    """2D sin/cos positional encoding of grid coords in [0,1]; dim even."""
    i = torch.arange(g, device=device, dtype=dtype) / max(g - 1, 1)
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=dtype)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    out = []
    for coord in (ii.reshape(-1), jj.reshape(-1)):
        ang = coord[:, None] * freqs[None, :]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)[:, :dim]              # [N, dim]


class TransformerBaseline(nn.Module):
    """Global self-attention over the N cells + 2D positional encoding."""
    def __init__(self, dim=32, layers=2, heads=4, ff=64, in_ch=3):
        super().__init__()
        self.inp = nn.Linear(in_ch, dim)
        self.pos_dim = dim
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, ext, g, nb_idx=None, nb_valid=None):
        pos = _pos_enc(g, self.pos_dim, ext.device, ext.dtype)[None, :, :]
        x = self.inp(ext) + pos
        x = self.enc(x)
        return self.out(x)[..., 0]


_NB_CACHE = {}


def _neighbors(g, device):
    key = (g, str(device))
    if key not in _NB_CACHE:
        from soma.spatial_nca import grid_neighbors
        _NB_CACHE[key] = grid_neighbors(g, device)
    return _NB_CACHE[key]


def predict_field(model, name, ext, g, k_steps=None, return_delta=False):
    """Uniform call: ext [B,N,3] -> field [B,N], for any model in this module
    or SpatialNCA. Builds/caches the grid neighbourhood as needed.
    return_delta (nca only) also returns the final-step plan update [B,N]."""
    nb_idx, nb_valid = _neighbors(g, ext.device)
    if name == "nca":
        return model(ext, nb_idx, nb_valid, k_steps=k_steps,
                     return_delta=return_delta)
    return model(ext, g, nb_idx, nb_valid)


def build_model(name, **kw):
    if name == "nca":
        from soma.spatial_nca import SpatialNCA
        return SpatialNCA(**kw)
    if name == "cnn":
        return CNNBaseline(**kw)
    if name == "transformer":
        return TransformerBaseline(**kw)
    if name == "mlp":
        return MLPPointwise(**kw)
    raise ValueError(f"unknown model {name}")
