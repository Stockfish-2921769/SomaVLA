#!/usr/bin/env python3
"""Generic field-regression evaluator, shared by the capability probes.

A scene generator `gen(g, rng)` returns either None or a 4-tuple
(ext [3,g,g], target [g,g], valid [g,g], meta) — meta is whatever the probe's
optional `progress_fn` needs. Reports MSE and per-scene Spearman rho over valid
cells (the primary field-quality metric), plus an optional rollout `prog`.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from soma.spatial_baselines import predict_field


def eval_field(model, name, gen, g, n, rng, device, k_eval=96, progress_fn=None):
    exts, tars, vals, metas = [], [], [], []
    tries = 0
    while len(exts) < n and tries < n * 20:
        tries += 1
        out = gen(g, rng)
        if out is None:
            continue
        ext, tar, val, meta = out
        if val.sum() < 2:
            continue
        exts.append(ext.reshape(3, -1).T)
        tars.append(tar.reshape(-1))
        vals.append(val.reshape(-1))
        metas.append(meta)
    if not exts:
        return {"mse": float("nan"), "rho": float("nan"), "n": 0}
    ext = torch.tensor(np.stack(exts), dtype=torch.float32, device=device)
    tar = np.stack(tars)
    val = np.stack(vals)

    model.eval()
    k = k_eval if name == "nca" else None
    with torch.no_grad():
        pred = predict_field(model, name, ext, g, k_steps=k).cpu().numpy()

    mse = float((((pred - tar) ** 2) * val).sum() / max(val.sum(), 1))
    rhos = []
    for kk in range(len(tars)):
        m = val[kk]
        if m.sum() < 3:
            continue
        r = spearmanr(pred[kk][m], tar[kk][m]).correlation
        if np.isfinite(r):
            rhos.append(r)
    res = {"mse": mse, "rho": float(np.mean(rhos)) if rhos else float("nan"),
           "n": len(tars)}

    if progress_fn is not None:
        progs = []
        for kk in range(len(tars)):
            p = progress_fn(pred[kk].reshape(g, g), metas[kk])
            if p is not None:
                progs.append(p)
        res["prog"] = float(np.mean(progs)) if progs else float("nan")
    return res
