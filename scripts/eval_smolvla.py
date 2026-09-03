#!/usr/bin/env python3
"""Track A: SmolVLA-0.5B fine-tuned closed-loop eval on the SAME distributions
as SigVLA-tiny (eval_vla.py) and the NCA baselines.

The fine-tuned SmolVLA replaces the tiny vision controller: a 500M SmolVLM2 VLM
+ flow-matching action expert, mutated to the decision-A task (chunk-8, single
camera, 7-dim state, 4-dim action) and fine-tuned on data/vla_train.npz. It is
measured on the exact hard + long-transport distributions the anchors were
scored on → a directly comparable (params, success) point answering:

  Q1 (capability gap at same data/harness): SmolVLA vs SigVLA-tiny sim %.
  Q2 (is A's long-transport failure architectural/BC, not scale): if a 500M
     model still drops on MuJoCo long-transport while the aware-NCA / planner-NCA
     hold it, the failure is in the open-loop-BC-to-chunk structure, not params.

Everything about the episode loop is inherited verbatim from eval_vla.py
(run_episode_vla / collect / report): perception noise, mid-carry disturbance,
no-break-on-drop, MuJoCo FixedPadsReplay adjudication, long-transport replay.
Only the controller's decode step differs: the fine-tuned SmolVLA policy's
select_action pops one absolute target from its internal n_action_steps=8 queue
(exact receding-horizon semantics), fed the SAME rendered top-down image
(in [0,1], since SmolVLA maps [0,1]->[-1,1] internally) + normalized 7-dim state.

The denormalization stats are NOT in the SmolVLA ckpt (saved norm=None); they
come from ckpts/vla_tiny/best.pt, which holds the exact ch/st mean/std used to
build data/vla_train.npz (verified: denorm recovers physical absolute poses).

Run from repo root (SMOKE first, ~10 hard cells):
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python scripts/eval_smolvla.py --ckpt ckpts/smolvla_ft/best.pt --n-cells 6 --eps 2
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_smolvla.py \
      --ckpt ckpts/smolvla_ft/best.pt --n-cells 12 --eps 3
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_smolvla.py \
      --ckpt ckpts/smolvla_ft/best.pt --long-transport 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from vla_data import siglip_normalize
from train_smolvla import build_policy
from eval_vla import run_episode_vla, collect, report
from eval_implicit_planning import cells
from eval_baseline_hard import (adjudicate, long_transport,
                                DEFAULT_NOISE_MM, DEFAULT_DISTURB_MM)


def as_np(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else np.asarray(x)


class SmolVLAChunkController:
    """Receding-horizon via the policy's own action queue (n_action_steps=8):
    decode a chunk from the current render, execute it open-loop, re-render at
    the next chunk boundary. Identical scene/state interface to
    VLAChunkController, so run_episode_vla is a pure drop-in."""

    def __init__(self, policy, lang_ids, lang_mask, norm, device):
        self.policy = policy
        self.device = device
        self.lang_ids = lang_ids.to(device)
        self.lang_mask = lang_mask.to(device).bool()
        self.chunk_mean = as_np(norm["ch_mean"]).astype(np.float32)
        self.chunk_std = as_np(norm["ch_std"]).astype(np.float32)
        self.state_mean = as_np(norm["st_mean"]).astype(np.float32)
        self.state_std = as_np(norm["st_std"]).astype(np.float32)

    def reset(self):
        self.policy.reset()

    def step(self, state, scene_see, mass, mu):
        from vla_data import render_scene
        img = render_scene(scene_see[:2], scene_see[2:], state[:2], mass, mu)
        # npz images are siglip-normalized CHW [-1,1]; SmolVLA wants [0,1] and
        # maps to [-1,1] internally, so invert only the scale (keep CHW).
        img01 = (siglip_normalize(np.asarray(img, dtype=np.float32) / 255.0)
                 * 0.5) + 0.5
        st_n = ((state - self.state_mean) / self.state_std).astype(np.float32)
        obs = {
            "observation.images.camera1":
                torch.from_numpy(img01[None]).to(self.device),
            "observation.state":
                torch.from_numpy(st_n[None]).to(self.device),
            "observation.language.tokens": self.lang_ids,
            "observation.language.attention_mask": self.lang_mask,
        }
        with torch.no_grad():
            a = self.policy.select_action(obs)      # [1, 4] normalized
        a = a.detach().cpu().numpy().reshape(-1)[:4]
        return a * self.chunk_std + self.chunk_mean  # absolute [x,y,z,open]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpts/smolvla_ft/best.pt")
    ap.add_argument("--norm-ckpt", type=str, default="ckpts/vla_tiny/best.pt",
                    help="carries the ch/st mean/std used to build the npz")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--percept-noise", type=float, default=DEFAULT_NOISE_MM)
    ap.add_argument("--disturb", type=float, default=DEFAULT_DISTURB_MM)
    ap.add_argument("--long-transport", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="also run the clean distribution (σ=0, push=0)")
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    policy, lang_ids, lang_mask = build_policy(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ck["state"], strict=False)
    policy.eval()
    norm = torch.load(args.norm_ckpt, map_location="cpu",
                      weights_only=False)["norm"]
    n_all = sum(p.numel() for p in policy.parameters())
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"SmolVLA-0.5B (ft): total {n_all/1e6:.1f}M, trainable {n_tr/1e6:.2f}M"
          f"  best @step {ck['step']}")

    vla = SmolVLAChunkController(policy, lang_ids, lang_mask, norm, device)

    try:
        import mujoco
        from eval_level_a import FixedPadsReplay
        use_mj = not args.no_mujoco
    except Exception as e:          # noqa: E722
        print(f"warning: mujoco unavailable ({e}); sim-only")
        use_mj = False

    cs = cells(args.n_cells, rng, hard_frac=0.5)
    sigma_m, disturb_m = args.percept_noise / 1000.0, args.disturb / 1000.0

    def run(name, sigma_m, disturb_m):
        sim_ok, drops, cdrop, segs, timeouts, tr_steps = collect(
            vla, cs, rng, device, sigma_m, disturb_m, args.chunk_size,
            args.eps)
        mj, ltr = None, None
        if use_mj:
            mj = [adjudicate(seg, mu, m) if seg is not None else (None, None, None)
                  for mu, m, seg in segs]
            if args.long_transport > 0:
                ltr = long_transport(segs, args.long_transport)
        report(name, cs, args.eps, sim_ok, drops, cdrop, segs, timeouts,
               tr_steps, sigma_m * 1000.0, disturb_m * 1000.0, mj, ltr,
               args.long_transport or 0)
        return sim_ok

    run("hard distribution", sigma_m, disturb_m)
    if args.clean:
        run("clean distribution", 0.0, 0.0)


if __name__ == "__main__":
    main()
