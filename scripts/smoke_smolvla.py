#!/usr/bin/env python3
"""Infra smoke for Track A: SmolVLA-0.5B (lerobot/smolvla_base).

Loads the pretrained policy (0.5B VLM backbone + flow-matching action expert)
and runs ONE select_action on a synthetic single-camera observation, the same
kind of top-down render the decision-A / hierarchical harnesses use. Success =
the policy loads from the local HF cache and returns a FINITE [1,6] action.

This does NOT touch network beyond the HF cache: HF_HUB_OFFLINE=1 forces all
model weights to come from ~/.cache/huggingface/hub (already populated).

Run inside the lerobot venv (has lerobot + the new transformers):
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_smolvla.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "lerobot/smolvla_base"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK = "Put the object on the green ring.\n"


def synthetic_topdown(s=256, seed=0):
    """One top-down RGB render [3,s,s] float in [0,1], grey bg + a dark object."""
    rng = np.random.RandomState(seed)
    img = np.full((s, s, 3), 235, dtype=np.uint8)
    # a "green ring" hint at center-bottom and an object rectangle top-left
    cy, cx = s // 2, s // 2
    img[cy - 6:cy + 6, cx - 6:cx + 6] = (30, 120, 60)
    oy, ox = s // 3, s // 3
    img[oy - 12:oy + 12, ox - 8:ox + 8] = (150, 90, 80)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def tokenize_task(tok, text, max_len=48):
    enc = tok(text, add_special_tokens=True, padding="max_length",
              max_length=max_len, truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def main():
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print(f"[import lerobot policy] {time.time()-t0:.1f}s")

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(REPO)
    print(f"[from_pretrained {REPO}] {time.time()-t0:.1f}s")
    cfg = policy.config

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    print(f"params: total {n_all/1e6:.2f}M, trainable {n_train/1e6:.2f}M")
    print(f"image_features: {cfg.image_features if hasattr(cfg,'image_features') else '?'}")
    print(f"state dim {cfg.max_state_dim}, chunk {cfg.chunk_size}, "
          f"num_steps {cfg.num_steps}, n_action_steps {cfg.n_action_steps}")

    # tokenizer for the language token stream (from the VLM repo, in HF cache)
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    lang_ids, lang_mask = tokenize_task(tok, TASK)
    print(f"[tokenizer] {time.time()-t0:.1f}s  lang_len={lang_ids.shape[-1]}")

    img = synthetic_topdown()
    dev = next(policy.parameters()).device
    obs = {
        "observation.images.camera1": torch.from_numpy(img).unsqueeze(0).to(dev),  # [1,3,256,256]
        "observation.state": torch.zeros(1, 6).to(dev),                            # [1,6]
        "observation.language.tokens": lang_ids.to(dev),
        "observation.language.attention_mask": lang_mask.to(dev).bool(),
    }

    policy.reset()
    t0 = time.time()
    with torch.no_grad():
        a = policy.select_action(obs)
    dt = time.time() - t0
    a = a.detach().cpu().numpy()
    finite = bool(np.isfinite(a).all())
    print(f"[select_action #1] {dt:.1f}s  out shape {a.shape}  finite={finite}")
    print("  action:", np.round(a.ravel(), 4))

    # a second call should come from the cached action queue (fast)
    policy.reset()
    t0 = time.time()
    with torch.no_grad():
        a2 = policy.select_action(obs)
    print(f"[select_action #2 fresh chunk] {time.time()-t0:.1f}s  "
          f"finite={bool(np.isfinite(a2.cpu().numpy()).all())}")

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}  "
              f"alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if not finite:
        sys.exit("NON-FINITE ACTION")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
