#!/usr/bin/env python3
"""Track A: fine-tune SmolVLA-0.5B on decision-A data (learnability + real).

Reuses the decision-A dataset data/vla_train.npz (already per-dim normalized
chunk-8 [x,y,z,open], 7-dim state, siglip-normalized [-1,1] top-down images).
SmolVLA is fed the SAME observations: image inverted back to [0,1]
(stored*0.5+0.5, since SmolVLA internally maps [0,1]->[-1,1]), normalized
state[7], normalized chunk[8,4] as the flow-matching action target, plus a
fixed task-language token stream. Base config mutated post-load to our task:
chunk 8, single camera, 7-dim state, 4-dim action.

--smoke: overfit a 64-sample slice for 200 steps (learnability check), then exit.
Default: full fine-tune on the train split, eval-best by val MSE, save the
trainable state (expert+proj) + config for the closed-loop harness.

Run inside the lerobot venv (cerebvla torch reused):
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=1 python scripts/train_smolvla.py --smoke
  CUDA_VISIBLE_DEVICES=1 python scripts/train_smolvla.py --steps 6000 \
      --out ckpts/smolvla_ft
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "lerobot/smolvla_base"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK = "Move the gripper to pick and place the object.\n"
ACTION_DIM = 4
STATE_DIM = 7
CHUNK = 8


def tokenize_task(tok, text, max_len=48):
    enc = tok(text, add_special_tokens=True, padding="max_length",
              max_length=max_len, truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def build_policy(device):
    """Base policy, config mutated to our single-camera / 7-state / 4-action
    chunk-8 task. Returns (policy, lang_ids, lang_mask)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(REPO)
    cfg = policy.config
    cfg.chunk_size = CHUNK
    cfg.n_action_steps = CHUNK
    # Single top camera + 7-dim state; 4-dim action chunk.
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE,
                                           shape=(STATE_DIM,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL,
                                                    shape=(3, 256, 256)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    policy.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    lang_ids, lang_mask = tokenize_task(tok, TASK)
    return policy, lang_ids.to(device), lang_mask.to(device).bool()


def loss_for_batch(policy, batch):
    """Flow-matching loss masked to the 4 real action dims."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    actions = policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(actions.shape[0], actions.device)
    losses = policy.model.forward(images, img_masks, lang_tokens, lang_masks,
                                  state, actions, noise=noise, time=time)
    return losses[:, :, :ACTION_DIM].mean()


def data_batches(im, ch, st, lang_ids, lang_mask, device, split=0.9, rng=None):
    """Yield (img01[0,1], ch_norm, st_norm) batches over train/val split."""
    n = len(im)
    cut = int(split * n)
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    train, val = idx[:cut], idx[cut:]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/vla_train.npz")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--init", type=str, default="",
                    help="resume from a previous train_smolvla best.pt (state)")
    ap.add_argument("--state-noise-mm", type=float, default=0.0,
                    help="Gaussian proprioception noise (mm, xy-scale) added to "
                         "train state so the expert cannot rely on the state "
                         "shortcut and must read object location from the image "
                         "(closed-loop covariate-shift mitigation)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="ckpts/smolvla_ft")
    ap.add_argument("--norm-ckpt", type=str, default="ckpts/vla_tiny/best.pt",
                    help="raw-state stats (st_std) for state-noise scaling")
    ap.add_argument("--smoke", action="store_true",
                    help="overfit a 64-sample slice for 200 steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    t0 = time.time()
    policy, lang_ids, lang_mask = build_policy(device)
    print(f"[load base] {time.time()-t0:.1f}s")
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        policy.load_state_dict(ck["state"], strict=False)
        print(f"[init from {args.init}] step {ck.get('step')}")
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    print(f"params total {n_all/1e6:.2f}M  trainable {n_tr/1e6:.2f}M")

    d = np.load(args.data)
    im, ch, st = d["images"], d["chunks"], d["states"]
    img01 = (im.astype(np.float32) * 0.5) + 0.5      # [-1,1] -> [0,1]
    print(f"data {im.shape} chunks {ch.shape} states {st.shape}")

    # per-dim raw proprioception noise sigma -> normalized space
    st_noise_sig = None
    if args.state_noise_mm > 0:
        nck = torch.load(args.norm_ckpt, map_location="cpu",
                         weights_only=False)["norm"]
        st_std = np.asarray(nck["st_std"].cpu()) if torch.is_tensor(
            nck["st_std"]) else np.asarray(nck["st_std"])
        scale = np.array([1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 0.8], np.float32)
        st_noise_sig = (scale * (args.state_noise_mm / 1000.0)
                        / np.maximum(st_std, 1e-6)).astype(np.float32)
        print(f"state noise sigma (normalized): {np.round(st_noise_sig, 3)}")

    train, val = data_batches(img01, ch, st, lang_ids, lang_mask, device,
                              split=0.9)
    if args.smoke:
        train = train[:64]
        val = val[:32]
        args.steps = min(args.steps, 200)
    ntr = len(train)

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    warmup = min(200, args.steps // 10)
    lin = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, args.steps - warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [lin, cos], milestones=[warmup])

    def make_batch(idx, noisy=True):
        j = idx[np.random.randint(len(idx), size=args.batch)]
        b_img = torch.from_numpy(img01[j]).to(device)
        b_st = st[j].copy()
        if noisy and st_noise_sig is not None:
            b_st = (b_st + np.random.normal(0.0, st_noise_sig,
                                            b_st.shape).astype(np.float32))
        b_st = torch.from_numpy(b_st).to(device)
        b_ch = torch.from_numpy(ch[j]).to(device)
        return {
            "observation.images.camera1": b_img,
            "observation.state": b_st,
            "action": b_ch,
            "observation.language.tokens": lang_ids.expand(args.batch, -1),
            "observation.language.attention_mask": lang_mask.expand(args.batch, -1),
        }

    def val_loss():
        policy.eval()
        vlosses = []
        with torch.no_grad():
            for _ in range(32):
                vlosses.append(loss_for_batch(policy,
                                              make_batch(val, noisy=False)).item())
        return float(np.mean(vlosses))

    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)
    print(f"{'step':>6} {'loss':>9} {'valMSE':>9} {'lr':>9} {'wall':>6}")
    for s in range(1, args.steps + 1):
        policy.train()
        b = make_batch(train)
        loss = loss_for_batch(policy, b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 10.0)
        opt.step()
        sched.step()
        if s % max(100, args.steps // 10) == 0 or s == args.steps:
            vl = val_loss()
            if vl < best_val:
                best_val = vl
                sd = {k: v.detach().cpu()
                      for k, v in policy.state_dict().items()
                      if "vlm_with_expert.vlm" not in k}
                torch.save({"state": sd, "step": s,
                            "norm": None,
                            "args": vars(args)},
                           os.path.join(args.out, "best.pt"))
            print(f"{s:>6} {loss.item():>9.4f} {vl:>9.4f} "
                  f"{opt.param_groups[0]['lr']:>9.1e} "
                  f"{time.time()-t0:>6.0f}", flush=True)
        elif args.smoke and s % 20 == 0:
            print(f"{s:>6} {loss.item():>9.4f}", flush=True)
    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
