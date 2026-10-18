#!/usr/bin/env python3
"""Tier 1 — train the spatial plan field + fair baselines.

Trains one model to regress the normalized geodesic-to-goal field on a cluttered
G x G grid (supervision = classical Dijkstra, no RL). Models differ only in
inductive bias (see soma/spatial_baselines.py): nca = local + iterative
propagation; cnn = local + fixed depth; transformer = global attention;
mlp = pointwise.

Train distribution: family=random, G=32, obs 0-3. Generalization is measured
elsewhere (eval_spatial_plan.py).

Run from the repo root:
    python scripts/train_spatial_nca.py --model nca --steps 4000
    python scripts/train_spatial_nca.py --model cnn --steps 4000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from soma.spatial_baselines import build_model, predict_field
from spatial_plan_data import make_batch


def masked_mse(pred, tar, val):
    err = (pred - tar) ** 2 * val.float()
    return err.sum() / val.float().sum().clamp(min=1.0)


@torch.no_grad()
def eval_loss(model, name, g, rng, n=64, family="mixed", obs_range=(0, 3),
              device="cpu", norm_c=None, n_goals=1, target="geodesic",
              horizon=None):
    model.eval()
    ext, tar, val = make_batch(g, n, rng, family=family, obs_range=obs_range,
                               device=device, norm_c=norm_c, n_goals=n_goals,
                               target=target, horizon=horizon)
    pred = predict_field(model, name, ext, g)
    mse = masked_mse(pred, tar, val).item()
    mae = ((pred - tar).abs() * val.float()).sum() / val.float().sum().clamp(min=1.0)
    model.train()
    return mse, mae.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nca", "cnn", "transformer", "mlp"],
                    default="nca")
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--family", type=str, default="mixed",
                    help="training scene family: random | wall | mixed")
    ap.add_argument("--obs-lo", type=int, default=0)
    ap.add_argument("--obs-hi", type=int, default=3)
    ap.add_argument("--norm-c", type=float, default=64.0,
                    help="fixed target normalizer (dist/norm_c); scale-consistent "
                         "across G. Use 0 for per-scene max-distance normalization.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-steps", type=int, default=48)
    ap.add_argument("--k-lo", type=int, default=None,
                    help="NCA only: if set with --k-hi, sample a random number "
                         "of relaxation steps per batch in [k_lo, k_hi]. Forces "
                         "the rule to a fixed point (so it transfers to larger "
                         "grids / more iterations).")
    ap.add_argument("--k-hi", type=int, default=None)
    ap.add_argument("--fp-lambda", type=float, default=0.0,
                    help="NCA only: weight on the fixed-point residual loss "
                         "(penalize the final-step plan update, so the rule "
                         "stops at a true fixed point instead of drifting).")
    ap.add_argument("--hidden", type=int, default=96, help="NCA update MLP width")
    ap.add_argument("--morphogen-dim", type=int, default=8, help="NCA scratch dim")
    ap.add_argument("--msg-dim", type=int, default=40, help="NCA perception width")
    ap.add_argument("--cnn-depth", type=int, default=4, help="CNN depth (RF=1+2d)")
    ap.add_argument("--cnn-channels", type=int, default=24, help="CNN width")
    ap.add_argument("--n-goals", type=int, default=1,
                    help="goals per scene (>1 = multi-goal nearest-reach field)")
    ap.add_argument("--n-goals-lo", type=int, default=None,
                    help="if set with --n-goals-hi, sample the goal count per "
                         "batch in [lo, hi] (train a range, test outside it)")
    ap.add_argument("--n-goals-hi", type=int, default=None)
    ap.add_argument("--target", choices=["geodesic", "harmonic", "ttc"],
                    default="geodesic",
                    help="regression target: geodesic distance field | harmonic "
                         "(Laplace) potential | time-to-contact (moving obstacles)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="ttc only: contact-time horizon (default 2*G)")
    ap.add_argument("--tag", type=str, default="", help="suffix for ckpt filename")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda | cuda:0 | cuda:1 | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints_spatial")
    ap.add_argument("--val-every", type=int, default=500)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    norm_c = args.norm_c if args.norm_c and args.norm_c > 0 else None

    obstacle_ch = 0 if args.target == "ttc" else 1
    if args.model == "nca":
        kw = {"k_steps": args.k_steps, "hidden": args.hidden,
              "morphogen_dim": args.morphogen_dim, "msg_dim": args.msg_dim,
              "obstacle_ch": obstacle_ch}
    elif args.model == "cnn":
        kw = {"depth": args.cnn_depth, "channels": args.cnn_channels}
    else:
        kw = {}
    model = build_model(args.model, **kw).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.model}  params={nparam:,}  G={args.g}  device={device}")
    if args.model == "cnn":
        print(f"  (CNN receptive field = {model.receptive_field} cells, fixed)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        ng = args.n_goals
        if args.n_goals_lo and args.n_goals_hi:
            ng = int(rng.randint(args.n_goals_lo, args.n_goals_hi + 1))
        ext, tar, val = make_batch(args.g, args.batch, rng, family=args.family,
                                   obs_range=(args.obs_lo, args.obs_hi),
                                   device=device, norm_c=norm_c, n_goals=ng,
                                   target=args.target, horizon=args.horizon)
        opt.zero_grad()
        k = args.k_steps
        if args.model == "nca" and args.k_lo and args.k_hi:
            k = int(rng.randint(args.k_lo, args.k_hi + 1))
        use_fp = args.model == "nca" and args.fp_lambda > 0
        pred = predict_field(model, args.model, ext, args.g,
                             k_steps=k if args.model == "nca" else None,
                             return_delta=use_fp)
        delta = None
        if use_fp:
            pred, delta = pred
        loss = masked_mse(pred, tar, val)
        if use_fp:
            vf = val.float()
            loss = loss + args.fp_lambda * (delta ** 2 * vf).sum() / vf.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.val_every == 0 or step == 1:
            mse, mae = eval_loss(model, args.model, args.g, rng, device=device,
                                 norm_c=norm_c, n_goals=args.n_goals,
                                 target=args.target, horizon=args.horizon)
            print(f"  step {step:5d}/{args.steps}  train {np.mean(losses[-200:]):.4f}"
                  f"  val mse {mse:.4f} mae {mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    ckpt = os.path.join(args.out, f"spatial_{args.model}{args.tag}.pt")
    torch.save({"model_name": args.model, "model": model.state_dict(),
                "g": args.g, "k_steps": args.k_steps,
                "arch": kw, "tag": args.tag, "n_goals": args.n_goals,
                "args": vars(args)}, ckpt)
    print(f"saved → {ckpt}")


if __name__ == "__main__":
    main()
