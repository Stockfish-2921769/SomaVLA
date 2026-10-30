#!/usr/bin/env python3
"""Shape/behavior unit tests for the VLA-lite skeleton (CPU, fast)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def test_arm_init_parity():
    """Shared stack must be init-identical across arms for a fixed seed."""
    arms = ["nca", "none", "zero", "random", "euclid", "cnn", "vis"]
    shared = {"adapter", "nca", "readout", "decoder", "language"}
    ref = None
    for arm in arms:
        torch.manual_seed(0)
        pol = VLALiteLowLevel(field_source=arm)
        sd = {k: v.clone() for k, v in pol.state_dict().items()
              if k.split(".")[0] in shared}
        if ref is None:
            ref = sd
            continue
        for k, v in sd.items():
            assert torch.equal(v, ref[k]), (arm, k)
    print("arm init parity ok")


def test_field_bandwidth():
    """NCA and CNN expose equal-width states; `plan` readout keeps ch 0."""
    b = 2
    ext = torch.randn(b, 77, 32, 32)
    for layers in (12, 32):
        cnn = CNNField(ext_dim=77, layers=layers)
        assert cnn(ext).shape == (b, 32, 32, 32), layers
    nca, cnn = NCA(), CNNField(ext_dim=77)
    print("  params nca", count_parameters(nca), "cnn12",
          count_parameters(cnn))

    tok = torch.randn(b, 256, 768)
    states = torch.tensor([[0.5, 0.3, 0.76, 0, 0, 0, 1.0]] * b)
    subgoals = torch.tensor([[0.45, 0.35, 0.52, 0.0]] * b)
    for arm in ("nca", "cnn", "vis"):
        torch.manual_seed(0)
        pol = VLALiteLowLevel(field_source=arm, readout_mode="plan")
        for p in pol.parameters():            # break zero init
            torch.nn.init.normal_(p, std=0.05)
        out = pol(vis_tokens=tok, states=states, subgoals=subgoals,
                  sample_steps=1)
        f = out["field"]
        assert f.shape == (b, 32, 32, 32), arm
        assert torch.count_nonzero(f[:, 1:]).item() == 0, arm
        assert torch.count_nonzero(f[:, :1]).item() > 0, arm
    print("field bandwidth ok")


def test_k_and_overrides():
    b = 2
    tok = torch.randn(b, 256, 768)
    states = torch.tensor([[0.5, 0.3, 0.76, 0, 0, 0, 1.0]] * b)
    subgoals = torch.tensor([[0.45, 0.35, 0.52, 0.0]] * b)
    gt_plan = torch.rand(b, 32, 32)
    torch.manual_seed(0)
    pol = VLALiteLowLevel(field_source="nca")
    for p in pol.nca.parameters():
        torch.nn.init.normal_(p, std=0.05)
    pol.eval()
    kw = dict(vis_tokens=tok, states=states, subgoals=subgoals,
              sample_steps=1, noise=torch.zeros(b, 8, 7))
    p4 = pol(k_steps=4, **kw)["plan"]
    p32 = pol(k_steps=32, **kw)["plan"]
    assert not torch.allclose(p4, p32)          # K actually changes the run
    out64 = pol(grid=64, k_steps=24, **kw)
    assert out64["plan"].shape == (b, 64, 64)
    assert out64["field_tokens"].shape == (b, 256, 256)

    z = pol(field_override="zero", **kw)
    assert torch.count_nonzero(z["field_tokens"]).item() == 0
    pg = pol(field_override="plan-gt", gt_plan=gt_plan, **kw)
    assert torch.allclose(pg["field"][:, 0], gt_plan)
    base = pol(**kw)
    assert torch.allclose(pg["field"][:, 1:], base["field"][:, 1:])
    pz = pol(field_override="plan-zero", **kw)
    assert torch.count_nonzero(pz["field"][:, 0]).item() == 0
    pr = pol(field_override="plan-random", **kw)
    assert pr["field"][:, 0].max() <= 1.0 + 1e-5
    try:
        VLALiteLowLevel(field_source="none")(field_override="plan-zero",
                                            **kw)
        raise AssertionError("plan override on no-field arm should fail")
    except ValueError:
        pass
    print("K override + field overrides ok")


def test_resize_matches_interpolate():
    import torch.nn.functional as F
    from soma.vla_lite.resize import resize_bilinear
    for (h, o) in [(16, 32), (32, 16), (16, 64), (64, 16), (32, 32)]:
        x = torch.randn(2, 5, h, h)
        ref = F.interpolate(x, size=(o, o), mode="bilinear",
                            align_corners=False)
        got = resize_bilinear(x, (o, o))
        assert got.shape == ref.shape, (h, o)
        assert torch.allclose(got, ref, atol=1e-5), (h, o,
                                                      (got - ref).abs().max())
    print("resize == F.interpolate ok")


def test_goal_dropout_and_legacy_cnn():
    b = 2
    tok = torch.randn(b, 256, 768)
    states = torch.tensor([[0.5, 0.3, 0.76, 0, 0, 0, 1.0]] * b)
    sg_a = torch.tensor([[0.45, 0.35, 0.52, 0.0]] * b)
    sg_b = torch.tensor([[0.60, 0.50, 0.52, 0.0]] * b)   # xy differs only
    kw = dict(vis_tokens=tok, states=states, sample_steps=1,
              noise=torch.zeros(b, 8, 7))
    torch.manual_seed(0)
    pol = VLALiteLowLevel(field_source="none")
    for p in pol.decoder.parameters():
        torch.nn.init.normal_(p, std=0.05)
    pol.eval()
    a = pol(subgoals=sg_a, **kw)["action"]
    bb = pol(subgoals=sg_b, **kw)["action"]
    assert not torch.allclose(a, bb)          # decoder reads subgoal xy
    a = pol(subgoals=sg_a, hide_goal=True, **kw)["action"]
    bb = pol(subgoals=sg_b, hide_goal=True, **kw)["action"]
    assert torch.allclose(a, bb)              # hidden: xy no longer matters

    torch.manual_seed(0)
    nca = VLALiteLowLevel(field_source="nca")
    for p in nca.parameters():
        torch.nn.init.normal_(p, std=0.05)
    nca.eval()
    a = nca(subgoals=sg_a, hide_goal=True, **kw)["action"]
    bb = nca(subgoals=sg_b, hide_goal=True, **kw)["action"]
    assert not torch.allclose(a, bb)          # goal still reaches the field

    old = VLALiteLowLevel(field_source="cnn", cnn_channels=1)
    assert old.cnn.out.weight.shape[0] == 1
    new = VLALiteLowLevel(field_source="cnn")
    new.load_state_dict(new.state_dict())
    old.load_state_dict(old.state_dict())
    for p in old.cnn.parameters():
        torch.nn.init.normal_(p, std=0.05)
    out = old(subgoals=sg_a, **kw)
    assert out["field"].shape == (b, 32, 32, 32)
    assert torch.count_nonzero(out["field"][:, 1:]).item() == 0
    print("goal dropout / hide-goal / legacy CNN ok")


def test_multi_wall_scene():
    import numpy as np
    from multi_task import _scene_reachable
    from vla_lite_ood import sample_scene_multi_wall
    rng = np.random.RandomState(0)
    for _ in range(5):
        objs, tgts, perm, walls = sample_scene_multi_wall(rng, (2, 6))
        assert len(walls) >= 5, len(walls)
        assert _scene_reachable(objs, tgts, walls)
    print("multi-wall OOD scenes ok")


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
    test_arm_init_parity()
    test_field_bandwidth()
    test_k_and_overrides()
    test_resize_matches_interpolate()
    test_goal_dropout_and_legacy_cnn()
    test_multi_wall_scene()
    test_siglip()
    print("ALL OK")
