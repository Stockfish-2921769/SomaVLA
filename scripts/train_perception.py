#!/usr/bin/env python3
"""Phase 4f-3 — train the SigLIP perception regression head: image → scene[4].

Frozen SigLIP encoder (ViT-B-16-SigLIP-256) + small MLP head → normalized
scene[4]. Trains on procedurally rendered scenes (scripts/render_scene.py),
regressing object/place meters. Held-out seed validates iid generalization
over the continuous scene distribution. MSE in normalized space, logged in mm.

Run from the repo root:
    python scripts/train_perception.py --steps 20000 --batch 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.perceive import SigLIPPerception, normalize_scene, SCENE_SCALE, preprocess_batch
import proc_sim
from render_scene import render_scene


def make_batch(rng, batch, device):
    imgs, scenes = [], []
    for _ in range(batch):
        obj, place = proc_sim._sample_scene(rng)
        imgs.append(render_scene(obj, place, rng=rng))
        scenes.append(np.concatenate([obj, place]))
    xs = preprocess_batch(imgs, device)
    ys = torch.from_numpy(np.stack([normalize_scene(s) for s in scenes])).to(device)
    return xs, ys


@torch.no_grad()
def evaluate(model, rng, device, n=512):
    xs, ys = make_batch(rng, n, device)
    pred_norm = model(xs)
    err_mm = (pred_norm - ys).abs() * torch.tensor(SCENE_SCALE * 1000, device=device)
    per_dim = err_mm.mean(0).cpu().numpy()
    return float(err_mm.mean().item()), per_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    val_rng = np.random.RandomState(999)   # held-out seed
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model = SigLIPPerception().to(device)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    print(f"trainable params (head only): {model.count_parameters()}")

    for step in range(1, args.steps + 1):
        xs, ys = make_batch(rng, args.batch, device)
        opt.zero_grad()
        pred = model(xs)
        loss = torch.mean((pred - ys) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            tr_err, _ = evaluate(model, rng, device)
            va_err, per_dim = evaluate(model, val_rng, device)
            print(f"[step {step:5d}/{args.steps}] train {tr_err:6.1f}mm  "
                  f"val {va_err:6.1f}mm  per-dim {per_dim.round(1)}")

    ckpt = os.path.join(args.out, "perception.pt")
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "arch": "SigLIPPerception"}, ckpt)
    print(f"saved → {ckpt}")


if __name__ == "__main__":
    main()
