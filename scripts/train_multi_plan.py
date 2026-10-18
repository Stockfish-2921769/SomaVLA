#!/usr/bin/env python3
"""Train a plan-field model (NCA / cnn / transformer / mlp) on the multi-object
visit task. The field is supervised by the same Dijkstra solver the routed
executor uses, so models differ only in inductive bias.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from soma.spatial_baselines import build_model, predict_field
from multi_plan_data import make_batch, G, EXT_DIM, OBSTACLE_CH


def masked_mse(pred, tar, val):
    err = (pred - tar) ** 2 * val.float()
    return err.sum() / val.float().sum().clamp(min=1.0)


def ranking_loss(pred, tar, val, nb_idx, nb_valid, margin):
    """Hinge on the greedy-descent readout: the predicted value must be lower
    at the neighbour the true field descends to. Fixes local minima that MSE
    tolerates but waypoint extraction trips on."""
    b, n = pred.shape
    flat = nb_idx.reshape(1, -1).expand(b, -1)
    tnb = tar.gather(1, flat).reshape(b, n, 8)
    vnb = val.gather(1, flat).reshape(b, n, 8) * nb_valid[None]
    tnb = torch.where(vnb.bool(), tnb, torch.full_like(tnb, 1e9))
    best_val, best_k = tnb.min(dim=2)
    best_nb = nb_idx.reshape(1, n, 8).expand(b, -1, -1).gather(
        2, best_k.unsqueeze(-1)).squeeze(-1)
    pred_nb = pred.gather(1, best_nb)
    has = val & (best_val < tar - 1e-9)
    hinge = torch.relu(pred - pred_nb + margin) * has.float()
    return hinge.sum() / has.float().sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, name, rng, device, n=8, batch=32, norm_c=128):
    model.eval()
    se = ae = cnt = 0.0
    for _ in range(n):
        ext, tar, val = make_batch(batch, rng, device=device, norm_c=norm_c)
        pred = predict_field(model, name, ext, G)
        v = val.float()
        se += float(((pred - tar) ** 2 * v).sum())
        ae += float(((pred - tar).abs() * v).sum())
        cnt += float(v.sum())
    model.train()
    return se / max(cnt, 1.0), ae / max(cnt, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nca",
                    choices=["nca", "cnn", "transformer", "mlp"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k-steps", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--morphogen-dim", type=int, default=8)
    ap.add_argument("--msg-dim", type=int, default=40)
    ap.add_argument("--cnn-depth", type=int, default=4)
    ap.add_argument("--cnn-channels", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--norm-c", type=float, default=128.0)
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--rank-margin", type=float, default=0.5)
    ap.add_argument("--out", default="ckpts/multi_plan_nca.pt")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "nca":
        kw = {"k_steps": args.k_steps, "hidden": args.hidden,
              "morphogen_dim": args.morphogen_dim, "msg_dim": args.msg_dim,
              "ext_dim": EXT_DIM, "obstacle_ch": OBSTACLE_CH}
    elif args.model == "cnn":
        kw = {"depth": args.cnn_depth, "channels": args.cnn_channels,
              "in_ch": EXT_DIM}
    elif args.model == "transformer":
        kw = {"in_ch": EXT_DIM}
    else:
        kw = {"in_dim": EXT_DIM}
    model = build_model(args.model, **kw).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.model} params={nparam:,} G={G} device={device}")

    from soma.spatial_nca import grid_neighbors
    nb_idx, nb_valid = grid_neighbors(G, device)
    rank_margin = args.rank_margin / args.norm_c

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    val_rng = np.random.RandomState(12345)
    best = float("inf")
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        ext, tar, val = make_batch(args.batch, rng, device=device,
                                   norm_c=args.norm_c)
        pred = predict_field(model, args.model, ext, G)
        loss = masked_mse(pred, tar, val)
        if args.rank_weight > 0:
            loss = loss + args.rank_weight * ranking_loss(
                pred, tar, val, nb_idx, nb_valid, rank_margin)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.val_every == 0 or step == 1:
            mse, mae = evaluate(model, args.model, val_rng, device,
                                norm_c=args.norm_c)
            if mse < best:
                best = mse
                torch.save({"model_name": args.model, "model": model.state_dict(),
                            "g": G, "k_steps": args.k_steps, "arch": kw,
                            "val_mse": mse, "step": step, "args": vars(args)},
                           args.out)
            print(f"step {step:5d}/{args.steps}  train {np.mean(losses[-200:]):.4f}"
                  f"  val mse {mse:.4f} mae {mae:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"done. best val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
