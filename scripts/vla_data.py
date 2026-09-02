#!/usr/bin/env python3
"""VLA training-data generation for the single-task pick-and-place.

Renders the abstract workspace as a 256×256 top-down RGB image and records
(action, image) pairs from the aware-NCA closed loop, so the vision policy can
be trained by behavior cloning on the exact same task the NCA baselines solve.

Per-episode demo (closed loop, clean — no perception noise / disturbance):
  * sample (obj, place) scene, (mu, mass) cell
  * run the aware experts + router for the full pick-and-place, recording
    (state_t, target_t) every step
  * chunk the recorded target sequence into windows of `chunk_size`; at each
    chunk start render the image of (object, place target, current EEF) with
    the object's size encoding mass and its shade encoding friction mu, so the
    vision policy can recover the load-adaptive step plan from pixels.

Output (npz):
  images   [N, 3, 256, 256]  float32 RGB 0..1 (already SigLIP-normalized)
  chunks   [N, chunk_size, 4] float32  absolute target poses [x, y, z, open]
  states   [N, 7] float32  EEF pose at chunk start (pos, rot, open)
  meta     [N, 4] float32  (obj_x, obj_y, place_x, place_y)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
from PIL import Image, ImageDraw

from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from soma.physics import sample_physics
from sim_env import SimEnv
import proc_sim
from eval_implicit_planning import load_experts

XL, XH, YL, YH = 0.28, 0.62, 0.28, 0.56   # workspace bounds (meters)
S = 256
ACTION_DIM = 4                            # [x, y, z, open]; rotation holds


def render_scene(obj, place, eef, mass, mu, s=S):
    """Top-down RGB image. obj/place/eef = (x, y) meters; object size ∝ mass,
    object shade ∝ mu (darker = higher friction). Returns PIL RGB."""
    def P(p):
        x, y = p
        px = (x - XL) / (XH - XL) * (s - 1)
        py = (YH - y) / (YH - YL) * (s - 1)
        return (int(px), int(py))
    img = Image.new("RGB", (s, s), (235, 232, 225))
    d = ImageDraw.Draw(img)
    for g in range(0, s, 32):
        d.line([(g, 0), (g, s)], fill=(220, 216, 208))
        d.line([(0, g), (s, g)], fill=(220, 216, 208))
    px, py = P(place)
    d.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(40, 160, 80), width=3)
    d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(40, 160, 80))
    side = int(8 + mass / 0.35 * 20)       # 12..28 px, mass 0.05..0.35
    ox, oy = P(obj)
    mu_t = min(max((mu - 0.2) / 0.4, 0.0), 1.0)
    r = int(210 - 70 * mu_t); g = int(60 + 25 * mu_t); b = int(50 + 30 * mu_t)
    d.rectangle([ox - side // 2, oy - side // 2, ox + side // 2, oy + side // 2],
                fill=(r, g, b), outline=(120, 30, 30), width=2)
    ex, ey = P(eef)
    d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(20, 20, 25),
              outline=(255, 255, 255), width=1)
    return img


def record_episode(experts, router, mu, mass, rng, device, chunk_size, max_steps=800):
    """One clean closed-loop episode with the aware experts. Returns list of
    (obj_xy, place_xy, mu, mass, state[7], target[7]) per step (all steps)."""
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene = np.concatenate([obj, place]).astype(np.float32)
    ctl = BodyGraphController(experts, router, device=device)
    ctl.reset(scene, env.state.copy())
    out = []
    for _ in range(max_steps):
        target, info = ctl.step(env.state, physics_ctx=env.physics_ctx())
        out.append((obj, place, mu, mass, env.state.copy(), target.copy()))
        env.step(target)
        if info["task_done"] or env.dropped:
            break
    return out


def chunk_and_render(episode, chunk_size, siglip_normalize):
    """Split an episode into (image, chunk) pairs. At each chunk start render
    the scene from the recorded state; chunk = the next chunk_size targets
    (absolute [x,y,z,open]). Returns lists (images, chunks, states, metas)."""
    images, chunks, states, metas = [], [], [], []
    n = len(episode)
    for t in range(0, n, chunk_size):
        obj, place, mu, mass, state, _ = episode[t]
        win = episode[t:t + chunk_size]
        if len(win) < chunk_size:
            break                       # drop partial trailing window
        img = render_scene(obj, place, state[:2], mass, mu)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = siglip_normalize(arr)
        chunk = np.stack([tgt[[0, 1, 2, 6]] for _, _, _, _, _, tgt in win])
        images.append(arr)
        chunks.append(chunk)
        states.append(state.copy())
        metas.append(np.concatenate([obj, place]))
    return images, chunks, states, metas


def siglip_normalize(arr):
    """arr [H,W,3] 0..1 → [3,H,W] with SigLIP mean/std (0.5)."""
    x = arr.transpose(2, 0, 1)
    return (x - 0.5) / 0.5


def sample_cell(rng, hard_frac, band_lo, band_hi):
    """(mu, mass) from the feasible distribution, hard-frac from the tail band."""
    if rng.random() < hard_frac:
        mu = rng.uniform(0.2, min(0.34, 0.35 / band_hi))
        return mu, rng.uniform(band_lo, band_hi) * mu
    return sample_physics(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys_honest")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-episodes", type=int, default=2000)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--band-lo", type=float, default=1.16)
    ap.add_argument("--band-hi", type=float, default=1.26)
    ap.add_argument("--out", type=str, default="data/vla_train.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    experts = load_experts(args.aware_dir, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    images, chunks, states, metas = [], [], [], []
    n_done = 0
    while n_done < args.n_episodes:
        mu, mass = sample_cell(rng, args.hard_frac, args.band_lo, args.band_hi)
        ep = record_episode(experts, router, mu, mass, rng, device,
                            args.chunk_size)
        im, ch, st, me = chunk_and_render(ep, args.chunk_size, siglip_normalize)
        n_done += 1
        if len(ch) >= 1:
            images += im; chunks += ch; states += st; metas += me
    images = np.stack(images).astype(np.float32)
    chunks = np.stack(chunks).astype(np.float32)
    states = np.stack(states).astype(np.float32)
    metas = np.stack(metas).astype(np.float32)
    np.savez_compressed(args.out, images=images, chunks=chunks, states=states,
                        metas=metas)
    print(f"wrote {args.out}: images {images.shape} chunks {chunks.shape} "
          f"states {states.shape} metas {metas.shape} from {n_done} episodes")


if __name__ == "__main__":
    main()
