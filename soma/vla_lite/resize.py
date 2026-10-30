"""Bilinear resize as fixed separable matmuls.

`F.interpolate(mode="bilinear")` has no deterministic CUDA backward, so it
raises under `torch.use_deterministic_algorithms(True)`. Bilinear resizing
(align_corners=False, no antialias) is separable and linear, so it equals
`A_h @ x @ A_w^T` with A built by resizing an identity once; matmul
backward is deterministic. Numerically equivalent to F.interpolate up to
float summation order.
"""
import torch
import torch.nn.functional as F

_CACHE = {}


def _matrix(n_in, n_out, device, dtype):
    key = (n_in, n_out, device, dtype)
    m = _CACHE.get(key)
    if m is None:
        eye = torch.eye(n_in, dtype=torch.float32).unsqueeze(1)   # [in,1,in]
        resp = F.interpolate(eye, size=n_out, mode="linear",
                             align_corners=False)               # [in,1,out]
        m = resp[:, 0].t().contiguous().to(device=device, dtype=dtype)
        _CACHE[key] = m                                          # [out,in]
    return m


def resize_bilinear(x, size):
    """x [B,C,H,W] -> [B,C,size[0],size[1]]."""
    h, w = x.shape[-2:]
    oh, ow = size
    if (h, w) == (oh, ow):
        return x
    a_h = _matrix(h, oh, x.device, x.dtype)
    a_w = _matrix(w, ow, x.device, x.dtype)
    return torch.einsum("oh,bchw,pw->bcop", a_h, x, a_w)
