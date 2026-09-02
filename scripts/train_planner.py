#!/usr/bin/env python3
"""Decision B — train the SigLIP scene+physics planner head.

Frozen SigLIP-B16 mean-pooled patch tokens → a ~0.26M MLP head → the 6-vector
[obj_x, obj_y, place_x, place_y, mu, mass] the hierarchical controller needs.
This is the module that removes the NCA actuator's oracle inputs: the image
encodes object position + place target geometrically and mu/mass as shade/size,
so a frozen SigLIP feature readout regresses all six (probe_margin.py showed the
frozen encoder channel carries margin at 0.32 N).

Regression errors map to closed-loop consequences asymmetrically: scene error
shifts the router's xy goals (20mm place_tol / 20mm contact_r slack), while
UNDER-estimating load (predicted m/μ too low) makes the aware actuator
under-contract → true-slip drops on hard cells. So we report per-dim MAE AND the
derived m/μ relative error + under-estimate (dangerous) fraction on the hard
band — the causal knob for the closed-loop outcome.

Run from the repo root:
  python scripts/train_planner.py --data data/planner_train.npz --steps 6000 \
      --out ckpts/planner
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch
import torch.nn as nn

from train_vla_tiny import SigVLATiny, count_params


class PlannerHead(nn.Module):
    """Mean-pooled frozen SigLIP tokens → 6-dim scene+physics."""

    def __init__(self, d=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, 6))

    def forward(self, toks):
        return self.mlp(toks.mean(dim=1))     # [B,256,768] → [B,6]


class SigLIPPlanner(nn.Module):
    """Frozen SigLIP-B16 + PlannerHead. Saves only the head (the vision trunk
    is reconstructed deterministically from the cached safetensors at load)."""

    def __init__(self, freeze_vision=True):
        super().__init__()
        self.vision = SigVLATiny(1, 4).vision
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
        self.head = PlannerHead()

    def forward(self, x):
        return self.head(self.vision.forward_features(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/planner_train.npz")
    ap.add_argument("--out", type=str, default="ckpts/planner")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data)
    im, lab = d["images"], d["labels"]
    print(f"data: {im.shape} {lab.shape}")
    split = int(0.9 * len(im))
    vim, vlab = im[split:], lab[split:]
    im, lab = im[:split], lab[:split]

    # Per-dim normalization (position in m, mu in [0.15,0.6], mass in kg).
    mean = lab.mean(axis=0)
    std = lab.std(axis=0) + 1e-6
    norm = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    lab = (lab - mean) / std
    vlab = (vlab - mean) / std

    planner = SigLIPPlanner().to(device)
    tot, tr = count_params(planner)
    print(f"planner: total {tot/1e6:.1f}M (frozen SigLIP), trainable {tr/1e6:.2f}M")

    opt = torch.optim.AdamW(planner.head.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(200, args.steps // 10)
    lin = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - warmup)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [lin, cos], milestones=[warmup])

    n = len(im)
    def batch(i):
        sl = slice(i * args.batch, (i + 1) * args.batch)
        return (torch.from_numpy(im[sl]).to(device),
                torch.from_numpy(lab[sl]).to(device))

    steps_per_ep = max(1, n // args.batch)
    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)
    for s in range(1, args.steps + 1):
        planner.train()
        img, yy = batch(s % steps_per_ep)
        loss = nn.functional.mse_loss(planner(img), yy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.head.parameters(), 5.0)
        opt.step()
        sched.step()
        if s % 200 == 0 or s == 1:
            planner.eval()
            with torch.no_grad():
                vp = planner(torch.from_numpy(vim).to(device))
                vl = nn.functional.mse_loss(vp, torch.from_numpy(vlab).to(device)).item()
            if vl < best_val:
                best_val = vl
                torch.save({"head": planner.head.state_dict(), "norm": norm,
                            "args": vars(args)},
                           os.path.join(args.out, "best.pt"))
            print(f"step {s:5d}  train {loss.item():.5f}  val {vl:.5f}  "
                  f"best {best_val:.5f}", flush=True)

    # Final per-dim val MAE (denormalized) + m/μ attribution on the best head.
    planner.head.load_state_dict(
        torch.load(os.path.join(args.out, "best.pt"), map_location=device,
                   weights_only=True)["head"])
    planner.eval()
    with torch.no_grad():
        vp = planner(torch.from_numpy(vim).to(device)).cpu().numpy()
    vp = vp * std + mean
    vt = vlab * std + mean
    names = ["obj_x", "obj_y", "place_x", "place_y", "mu", "mass"]
    print("\nper-dim val MAE:")
    for i, nm in enumerate(names):
        unit = "mm" if i < 4 else ""
        scale = 1000.0 if i < 4 else 1.0
        print(f"  {nm:>8}: {np.abs(vp[:, i] - vt[:, i]).mean() * scale:6.2f} "
              f"{unit}  (true std {std[i] * scale:.3f} {unit})")
    # Scene position error as a single number (mm).
    scene_mm = np.abs(vp[:, :4] - vt[:, :4]).mean() * 1000.0
    print(f"  scene xy MAE: {scene_mm:.2f} mm")
    # m/μ relative error + dangerous under-estimate direction.
    r_true = vt[:, 5] / vt[:, 4]
    r_pred = vp[:, 5] / vp[:, 4]
    rel = (r_pred - r_true) / r_true
    hard = r_true >= 1.16
    print(f"  m/μ rel err: mean {np.abs(rel).mean()*100:.1f}%  "
          f"(hard-band {np.abs(rel[hard]).mean()*100:.1f}%)")
    under = (rel[hard] < -0.1).mean() if hard.any() else float("nan")
    print(f"  hard-band UNDER-estimate (>10% low, → under-contract risk): "
          f"{under*100:.0f}%")
    print(f"done. best val {best_val:.5f} at {args.out}/best.pt (head only)")


if __name__ == "__main__":
    main()
