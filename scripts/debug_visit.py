#!/usr/bin/env python3
"""Offline decision accuracy of a visit-policy checkpoint on the val split."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from train_smolvla import build_policy
from eval_multi import as_np

MULTI_TASK = ("Pick up each object and place it on the matching colored "
              "target, going around the walls.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/visit_base/best.pt")
    ap.add_argument("--data", default="data/multi_visit.npz")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    args = ap.parse_args()

    d = np.load(args.data)
    n = len(d["images"])
    cut = int(0.9 * n)
    idx = (np.arange(cut, n) if args.split == "val" else np.arange(cut))
    rng = np.random.RandomState(0)
    idx = idx[rng.permutation(len(idx))[:args.n]]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, lang_ids, lang_mask = build_policy(device, MULTI_TASK)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ck["state"], strict=False)
    policy.eval()
    norm = ck["norm"]
    ch_mean = as_np(norm["ch_mean"]).astype(np.float32)
    ch_std = as_np(norm["ch_std"]).astype(np.float32)
    st_mean = as_np(norm["st_mean"]).astype(np.float32)
    st_std = as_np(norm["st_std"]).astype(np.float32)
    print(f"ckpt step {ck.get('step')}  val n={len(idx)}")

    im = d["images"][idx].astype(np.float32) / 255.0
    st = d["states"][idx].astype(np.float32)
    cm = d["chunks"][idx][:, 0].astype(np.float32)

    preds = []
    for k in range(0, len(idx), 64):
        b = {
            "observation.images.camera1": torch.from_numpy(im[k:k+64]).to(device),
            "observation.state": torch.from_numpy(st[k:k+64]).to(device),
            "observation.language.tokens": lang_ids.expand(len(im[k:k+64]), -1),
            "observation.language.attention_mask":
                lang_mask.expand(len(im[k:k+64]), -1),
        }
        with torch.no_grad():
            a = policy.predict_action_chunk(b)
        preds.append(a[:, 0].detach().cpu().numpy())
    pred = np.concatenate(preds) * ch_std + ch_mean
    gt = cm * ch_std + ch_mean

    err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    q = np.percentile(err, [10, 25, 50, 75, 90])
    open_acc = ((pred[:, 3] < 0.5) == (gt[:, 3] < 0.5)).mean()
    print(f"xy err cm  p10/25/50/75/90 = {np.round(q*100, 1)}")
    print(f"within 1cm {(err<0.01).mean()*100:.0f}%  2.5cm "
          f"{(err<0.025).mean()*100:.0f}%  5cm {(err<0.05).mean()*100:.0f}%  "
          f"10cm {(err<0.10).mean()*100:.0f}%")
    # identity-level decision: which of the 6 scene points (3 objs + 3 tgts)
    # does the label / the prediction point at?
    meta = d["metas"][idx]
    pts = meta.reshape(-1, 6, 2)                     # 3 objs + 3 targets
    gi = np.argmin(np.linalg.norm(pts - gt[:, None, :2], axis=2), axis=1)
    pi = np.argmin(np.linalg.norm(pts - pred[:, None, :2], axis=2), axis=1)
    ident = (gi == pi).mean()
    same_type = (((gi < 3) == (pi < 3))).mean()
    print(f"open/intent acc {open_acc*100:.0f}%  identity acc {ident*100:.0f}% "
          f"(same obj/tgt slot {same_type*100:.0f}%)")
    for o in (0, 1):
        m = (gt[:, 3] < 0.5) == (o == 0)
        print(f"  intent open={o}: n={m.sum()} xy p50 "
              f"{np.median(err[m])*100:.1f}cm  open mean "
              f"{pred[m, 3].mean():.2f} (gt {gt[m, 3].mean():.2f})")
    # raw state signal: is the hover xy/grip enough?
    s = st * st_std + st_mean
    print(f"  state grip distribution: 0-ish {np.mean(s[:,6]<0.5)*100:.0f}% "
          f"1-ish {np.mean(s[:,6]>=0.5)*100:.0f}%")


if __name__ == "__main__":
    main()
