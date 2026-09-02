#!/usr/bin/env python3
"""SigVLA-tiny: frozen SigLIP-B16 perception + a compact cross-attention action
decoder, behavior-cloned on rendered single-task pick-and-place demos.

Architecture (the end-to-end tiny-VLA branch of the A/B decision):
  image [B,3,256,256] → SigLIP-B16 trunk.forward_features → patch tokens
  [B,256,768] (frozen). A learned per-chunk query cross-attends to the patch
  tokens (+ an MLP-coded EEF state token) and regresses the chunk of absolute
  target poses [B, chunk, 4] = [x, y, z, open] (rotation holds).

SmolVLA-faithful in interface (image + instruction → action chunk) with the
same SigLIP-B16 encoder; the language model + diffusion head are simplified to
a fixed instruction embedding + a linear head. Swappable for the real
SmolVLA-0.5B when its weights are reachable.

Train from the repo root:
  python scripts/train_vla_tiny.py --data /tmp/vla_smoke.npz --out ckpts/vla_tiny \
      --steps 2000 --batch 32 --lr 1e-3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionDecoder(nn.Module):
    """Chunk queries cross-attend to SigLIP patch tokens + EEF state token."""

    def __init__(self, chunk_size, action_dim, d=256, n_layers=2, n_head=4,
                 state_dim=7):
        super().__init__()
        self.chunk_size = chunk_size
        self.d = d
        self.patch_proj = nn.Linear(768, d)
        self.query = nn.Parameter(torch.randn(chunk_size, d) * 0.02)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(),
                                       nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=n_head,
                                           dim_feedforward=d * 4,
                                           batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, action_dim))

    def forward(self, patch_tokens, state):
        """patch_tokens [B,256,768]; state [B,7]. Returns [B,chunk,action]."""
        B = patch_tokens.shape[0]
        mem = self.patch_proj(patch_tokens)
        q = self.query.unsqueeze(0).expand(B, -1, -1) + \
            self.state_mlp(state).unsqueeze(1)
        out = self.decoder(q, mem)
        return self.head(out)


class SigVLATiny(nn.Module):
    """SigLIP-B16 (frozen) + ActionDecoder."""

    def __init__(self, chunk_size, action_dim, state_dim=7, freeze_vision=True):
        super().__init__()
        import open_clip
        import glob
        f = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP-256/"
            "snapshots/*/open_clip_model.safetensors"))[0]
        # NOTE: keep the full SigLIP in a local var — assigning it as an
        # attribute would auto-register its text tower as trainable params.
        clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP-256", pretrained=f)
        self.vision = clip.visual.trunk
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
        self.decoder = ActionDecoder(chunk_size, action_dim, state_dim=state_dim)

    def vision_tokens(self, x):
        return self.vision.forward_features(x)     # [B, 256, 768]

    def forward(self, image, state):
        toks = self.vision_tokens(image)
        return self.decoder(toks, state)


def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    return total, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/vla_train.npz")
    ap.add_argument("--val-data", type=str, default=None)
    ap.add_argument("--out", type=str, default="ckpts/vla_tiny")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data)
    im, ch, st = d["images"], d["chunks"], d["states"]
    print(f"train: {im.shape} {ch.shape} {st.shape}")
    if args.val_data:
        vd = np.load(args.val_data)
        vim, vch, vst = vd["images"], vd["chunks"], vd["states"]
        print(f"val:   {vim.shape} {vch.shape} {vst.shape}")
    else:
        split = int(0.9 * len(im))
        vim, vch, vst = im[split:], ch[split:], st[split:]
        im, ch, st = im[:split], ch[:split], st[:split]

    # Per-dim normalization from the TRAIN set only; targets are denormalized
    # at eval via the stats saved in the checkpoint. Balances the position
    # dims (std ~5mm) against the gripper-open dim (std ~0.4).
    ch_mean = ch.mean(axis=(0, 1))
    ch_std = ch.std(axis=(0, 1)) + 1e-6
    st_mean = st.mean(axis=0)
    st_std = st.std(axis=0) + 1e-6
    norm = {"ch_mean": torch.from_numpy(ch_mean),
            "ch_std": torch.from_numpy(ch_std),
            "st_mean": torch.from_numpy(st_mean),
            "st_std": torch.from_numpy(st_std)}
    ch = (ch - ch_mean) / ch_std
    st = (st - st_mean) / st_std
    vch = (vch - ch_mean) / ch_std
    vst = (vst - st_mean) / st_std

    model = SigVLATiny(args.chunk_size, 4).to(device)
    tot, tr = count_params(model)
    print(f"SigVLA-tiny: total {tot:,} ({tot/1e6:.1f}M), trainable {tr:,} "
          f"({tr/1e6:.2f}M)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    n = len(im)
    def batch(i):
        sl = slice(i * args.batch, (i + 1) * args.batch)
        return (torch.from_numpy(im[sl]).to(device),
                torch.from_numpy(st[sl]).to(device),
                torch.from_numpy(ch[sl]).to(device))

    steps_per_ep = max(1, n // args.batch)
    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)
    for s in range(1, args.steps + 1):
        model.train()
        i = s % steps_per_ep
        img, stt, chk = batch(i)
        pred = model(img, stt)
        loss = F.mse_loss(pred, chk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == 1:
            model.eval()
            with torch.no_grad():
                vloss, mae = 0.0, 0.0
                for j in range(0, len(vim), 64):
                    vimg = torch.from_numpy(vim[j:j + 64]).to(device)
                    vstt = torch.from_numpy(vst[j:j + 64]).to(device)
                    vchk = torch.from_numpy(vch[j:j + 64]).to(device)
                    vpred = model(vimg, vstt)
                    vloss += F.mse_loss(vpred, vchk).item() * len(vchk)
                    err = (vpred - vchk).abs() * \
                        torch.from_numpy(ch_std).to(device)  # denorm to meters
                    mae += err.mean().item() * len(vchk)
                vloss /= len(vim); mae /= len(vim)
            if vloss < best_val:
                best_val = vloss
                torch.save({"model": model.state_dict(),
                            "args": vars(args), "norm": norm},
                           os.path.join(args.out, "best.pt"))
            print(f"step {s:5d}  train {loss.item():.5f}  "
                  f"val {vloss:.5f} (MAE {mae*1000:.1f}mm)  best {best_val:.5f}",
                  flush=True)
    print(f"done. best val {best_val:.5f} at {args.out}/best.pt")


if __name__ == "__main__":
    main()
