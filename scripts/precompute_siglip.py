#!/usr/bin/env python3
"""Precompute frozen SigLIP-B/16 patch tokens for a VLA-lite npz.

Output: .npy memmap [N, 256, 768] float16 (16x16 patch grid at 256px).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from soma.vla_lite.vision import FrozenSigLIP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data)
    # NpzFile re-decompresses the whole array on every slice access, so
    # materialize the images once (fits in RAM for these datasets).
    images = d["images"]
    n = len(images)
    if args.limit:
        n = min(n, args.limit)
    vis = FrozenSigLIP(device=args.device).to(args.device).eval()
    feat_dim = vis.feat_dim
    n_tok = vis.n_tokens
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mm = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float16, shape=(n, n_tok, feat_dim))
    with torch.no_grad():
        for i in range(0, n, args.batch):
            j = min(i + args.batch, n)
            x = torch.from_numpy(np.asarray(images[i:j])).to(args.device)
            t = vis(x)
            mm[i:j] = t.cpu().numpy().astype(np.float16)
            if i % (args.batch * 20) == 0:
                print(f"  {i}/{n}", flush=True)
    mm.flush()
    del mm
    meta = dict(data=args.data, out=args.out, n=n, tokens=n_tok,
                dim=feat_dim, dtype="float16", model="ViT-B-16-SigLIP-256")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out} {os.path.getsize(args.out)/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
