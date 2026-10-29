#!/usr/bin/env python3
"""Shape/behavior unit tests for the VLA-lite skeleton (CPU, fast)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from soma.vla_lite.adapter import SpatialAdapter
from soma.vla_lite.baselines import CNNField, EuclidField, random_smooth_field
from soma.vla_lite.decoder import FieldReadout, FlowActionDecoder
from soma.vla_lite.grid import XL, XH, YL, YH, cell_to_xy, xy_to_cell
from soma.vla_lite.high_level import HighLevelPolicy
from soma.vla_lite.nca import NCA, count_parameters
from soma.vla_lite.policy import VLALiteLowLevel
from soma.vla_lite.vision import FrozenSigLIP


def test_grid():
    for g in (32, 64):
        xy = torch.tensor([[XL + 0.01, YH - 0.01],
                           [0.5 * (XL + XH), 0.5 * (YL + YH)],
                           [XH - 0.01, YL + 0.01]])
        i, j = xy_to_cell(xy, g)
        back = cell_to_xy(i, j, g)
        step = torch.tensor([(XH - XL) / g, (YH - YL) / g]).abs()
        err = (back - xy).abs().max(dim=0).values
        assert torch.all(err <= step + 1e-6), (g, back, xy)
    print("grid ok")


def test_shapes():
    b = 2
    tok = torch.randn(b, 256, 768)
    ad = SpatialAdapter()
    f, o = ad(tok, size=32)
    assert f.shape == (b, 64, 32, 32) and o.shape == (b, 1, 32, 32)
    f64, o64 = ad(tok, size=64)
    assert f64.shape == (b, 64, 64, 64) and o64.shape == (b, 1, 64, 64)

    nca = NCA()
    ext = nca.make_ext(f, torch.sigmoid(o), torch.tensor([[0.4, 0.4]] * b),
                       torch.tensor([[0.5, 0.3]] * b), torch.zeros(b, 8))
    assert ext.shape == (b, 77, 32, 32)
    s, p = nca(ext)
    assert s.shape == (b, 32, 32, 32) and p.shape == (b, 32, 32)
    assert torch.allclose(p, torch.zeros_like(p), atol=1e-6)  # zero init
    s2, p2 = nca(ext, k_steps=1)
    assert s2.shape == s.shape
    ext64 = nca.make_ext(f64, torch.sigmoid(o64),
                         torch.tensor([[0.4, 0.4]] * b),
                         torch.tensor([[0.5, 0.3]] * b), torch.zeros(b, 8))
    s64, p64 = nca(ext64)
    assert s64.shape == (b, 32, 64, 64)

    ro = FieldReadout()
    z = ro(s)
    assert z.shape == (b, 256, 256)
    dec = FlowActionDecoder()
    a = torch.randn(b, 8, 7)
    out = dec(z, torch.zeros(b, 7), torch.zeros(b, 4),
              torch.zeros(b, 256), a, torch.ones(b))
    assert out.shape == (b, 8, 7)
    loss = dec.flow_loss(a, z, torch.zeros(b, 7), torch.zeros(b, 4),
                         torch.zeros(b, 256))
    assert torch.isfinite(loss)
    smp = dec.sample(z, torch.zeros(b, 7), torch.zeros(b, 4),
                     torch.zeros(b, 256), steps=3)
    assert smp.shape == (b, 8, 7)

    eu = EuclidField()
    assert eu(torch.tensor([[0.4, 0.4]] * b), 32).shape == (b, 1, 32, 32)
    rnd = random_smooth_field(b, 32, torch.device("cpu"))
    assert rnd.shape == (b, 1, 32, 32)
    print("shapes ok; NCA params", count_parameters(nca),
          "adapter", count_parameters(ad), "decoder", count_parameters(dec))


def test_high_level():
    b = 2
    hl = HighLevelPolicy(use_field=True)
    vis = torch.randn(b, 256, 768)
    st = torch.zeros(b, 7)
    lg = torch.zeros(b, 256)
    out = hl(vis, st, lg, field=torch.randn(b, 32, 32, 32))
    assert out.shape == (b, 4)
    out2 = hl(vis, st, lg, field=None)
    assert out2.shape == (b, 4)
    print("high-level ok; params", count_parameters(hl))


def test_policy_arms():
    b = 2
    tok = torch.randn(b, 256, 768)
    states = torch.tensor([[0.5, 0.3, 0.76, 0, 0, 0, 1.0]] * b)
    subgoals = torch.tensor([[0.45, 0.35, 0.52, 0.0]] * b)
    gt_plan = torch.rand(b, 32, 32)
    gt_obs = (torch.rand(b, 32, 32) > 0.7).float()
    for arm in ["nca", "gt", "euclid", "cnn", "vis", "none", "zero",
                "random"]:
        pol = VLALiteLowLevel(field_source=arm)
        pol.set_norm(torch.zeros(7), torch.ones(7),
                     torch.zeros(4), torch.ones(4))
        out = pol(vis_tokens=tok, states=states, subgoals=subgoals,
                  gt_plan=gt_plan, gt_obstacle=gt_obs, sample_steps=2)
        assert out["action"].shape == (b, 8, 7), arm
        assert torch.isfinite(out["action"]).all(), arm
        if "loss" in out:
            assert torch.isfinite(out["loss"])
        if arm == "zero":   # compute paid, field hidden from the decoder
            assert out["nca_delta"] is not None
            assert torch.count_nonzero(out["field_tokens"]).item() == 0
        if arm in {"none", "zero"}:
            assert torch.count_nonzero(out["field_tokens"]).item() == 0
        if arm == "vis":
            assert out["field"].shape == (b, 32, 32, 32)
            assert torch.count_nonzero(out["field_tokens"]).item() > 0
        pol2 = VLALiteLowLevel(field_source=arm)
        pol2.set_norm(torch.zeros(7), torch.ones(7),
                      torch.zeros(4), torch.ones(4))
        out2 = pol2(vis_tokens=tok, states=states, subgoals=subgoals,
                    gt_plan=gt_plan, gt_obstacle=gt_obs,
                    actions_norm=torch.randn(b, 8, 7))
        assert torch.isfinite(out2["loss"]), arm
        print(f"  arm {arm}: params {sum(p.numel() for p in pol.parameters())}")
    print("policy arms ok")


def test_siglip():
    vis = FrozenSigLIP(device="cpu")
    x = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
    with torch.no_grad():
        t = vis(x)
    assert t.shape == (2, 256, 768), t.shape
    assert not any(p.requires_grad for p in vis.parameters())
    print("siglip ok")


if __name__ == "__main__":
    test_grid()
    test_shapes()
    test_high_level()
    test_policy_arms()
    test_siglip()
    print("ALL OK")
