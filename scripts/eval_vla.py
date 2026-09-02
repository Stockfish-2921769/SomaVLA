#!/usr/bin/env python3
"""SigVLA-tiny closed-loop eval on the SAME distributions as the NCA baselines.

The VLA replaces the aware-NCA closed loop with a learned vision controller:
a frozen SigLIP-B16 encoder + a 2.4M-param cross-attention action decoder that
maps (top-down image, EEF state) → an 8-step chunk of absolute target poses.
It is measured on the exact hard + long-transport distributions the NCA
baselines were scored on, so it produces a directly comparable
(params, success) point for the end-to-end-vs-hierarchical A/B decision:

  blind NCA  0.11M params: hard sim 25/36 (69%), MuJoCo 11/15 (73%) LT drops
  aware NCA  0.12M params: hard sim 36/36 (100%), MuJoCo 0/15 (0%) LT drops
  VLA-tiny   2.44M train / 95.4M total (frozen SigLIP): measured here

Distribution fidelity:
  * perception noise σ: the controller's obj/place coords are corrupted BEFORE
    the image is rendered, so the VLA reads noisy positions from pixels (the
    pixel-level analog of the NCA's noisy scene). EEF state stays truthful
    (proprioception), as does the (mu,mass) size/shade encoding (physics_ctx is
    true for the aware NCA).
  * disturbance: one ±D mm EEF push mid-carry. The VLA only replans at chunk
    boundaries, so a push landing inside a chunk is NOT corrected until the
    next boundary — an honest open-loop-within-chunk limitation to measure.
  * MuJoCo adjudication + long-transport replay: identical to the NCA harness
    (worst-case yaw FixedPadsReplay of the recorded carried-phase segments).

Run from the repo root:
  python scripts/eval_vla.py --ckpt ckpts/vla_tiny/best.pt --n-cells 12 --eps 3
  python scripts/eval_vla.py --ckpt ckpts/vla_tiny/best.pt --long-transport 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from sim_env import SimEnv
import proc_sim
from vla_data import render_scene, siglip_normalize
from train_vla_tiny import SigVLATiny
from eval_implicit_planning import (EVAL_HARD_LO, EVAL_HARD_HI, cells, margin,
                                    load_experts)
from soma.physics import F_MAX
from eval_baseline_hard import (perceived_scene, adjudicate, long_transport,
                                DEFAULT_NOISE_MM, DEFAULT_DISTURB_MM)

LIFT_Z = 0.66            # env lift height: transport/carry is below this
LIFTED_AT = LIFT_Z + 0.02


class VLAChunkController:
    """Receding-horizon: at each chunk boundary, render the scene the
    controller sees and decode an 8-step absolute-pose chunk; execute the chunk
    open-loop, then re-render from the (truthful) observed EEF state."""

    def __init__(self, model, norm, chunk_size, device):
        self.model = model
        self.chunk_mean = np.asarray(norm["ch_mean"].cpu()) if torch.is_tensor(norm["ch_mean"]) else norm["ch_mean"]
        self.chunk_std = np.asarray(norm["ch_std"].cpu()) if torch.is_tensor(norm["ch_std"]) else norm["ch_std"]
        self.state_mean = np.asarray(norm["st_mean"].cpu()) if torch.is_tensor(norm["st_mean"]) else norm["st_mean"]
        self.state_std = np.asarray(norm["st_std"].cpu()) if torch.is_tensor(norm["st_std"]) else norm["st_std"]
        self.chunk_size = chunk_size
        self.device = device
        self.chunk = None
        self.chunk_i = 0

    def reset(self):
        self.chunk = None
        self.chunk_i = 0

    def step(self, state, scene_see, mass, mu):
        """Returns one absolute target [x,y,z,open]."""
        if self.chunk is None or self.chunk_i >= self.chunk_size:
            img = render_scene(scene_see[:2], scene_see[2:], state[:2],
                               mass, mu)
            arr = siglip_normalize(np.asarray(img, dtype=np.float32) / 255.0)
            st_n = ((state - self.state_mean) / self.state_std).astype(np.float32)
            with torch.no_grad():
                pred = self.model(
                    torch.from_numpy(arr[None]).to(self.device),
                    torch.from_numpy(st_n[None]).to(self.device))[0]
            self.chunk = (pred.cpu().numpy() * self.chunk_std
                          + self.chunk_mean)
            self.chunk_i = 0
        target = self.chunk[self.chunk_i]
        self.chunk_i += 1
        return target


def run_episode_vla(vla, mu, mass, rng, device, sigma_m, disturb_m,
                    chunk_size):
    """One closed-loop episode on the hard distribution. Returns
    (success, dropped, drop_carry, transport_steps[], seg or None, timeout)."""
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene_true = np.concatenate([obj, place]).astype(np.float32)
    scene_see = perceived_scene(scene_true, rng, sigma_m)
    vla.reset()
    state = env.state.copy()
    lifted, disturbed = False, False
    transport_steps, tr_seg = [], []
    was_attached = False
    for _ in range(800):
        if not disturbed:
            if state[2] > LIFTED_AT:
                lifted = True
            elif lifted and state[2] < LIFT_Z:
                env.perturb(disturb_m)          # mid-carry push
                disturbed = True
        target = vla.step(state, scene_see, mass, mu)     # [x,y,z,open]
        # Rotation holds: complete the 7-dim pose with the current orientation.
        target_pose = np.concatenate([target[:3], state[3:6], [target[3]]])
        prev = state.copy()
        state, _, info = env.step(target_pose)
        if was_attached:                        # carried phase (incl. slip step)
            transport_steps.append(np.linalg.norm(target_pose[:3] - prev[:3]) * 1000.0)
            tr_seg.append((prev.copy(), target_pose.copy()))
        if env.dropped:
            drop_carry = bool(was_attached)
            break
        if env.success():
            drop_carry = False
            break
        was_attached = info["attached"]
    else:
        drop_carry = bool(was_attached)
    seg = None
    if tr_seg:
        seg = {"states": np.array([s for s, _ in tr_seg], dtype=np.float32),
               "targets": np.array([t for _, t in tr_seg], dtype=np.float32)}
    timeout = not (env.success() or env.dropped)
    return (env.success(), env.dropped, drop_carry, transport_steps, seg,
            timeout)


def collect(vla, cs, rng, device, sigma_m, disturb_m, chunk_size, eps):
    """Per-variant: (sim_ok[], drops[], carry_drops[], segs[(mu,m,seg)],
    timeouts[], tr_steps[] per episode)."""
    sim_ok, drops, cdrop, segs, timeouts, tr_steps = [], [], [], [], [], []
    for mu, m in cs:
        for _ in range(eps):
            ok, dropped, dc, tr, seg, tm = run_episode_vla(
                vla, mu, m, rng, device, sigma_m, disturb_m, chunk_size)
            sim_ok.append(int(ok)); drops.append(int(dropped))
            cdrop.append(int(dc)); segs.append((mu, m, seg))
            timeouts.append(int(tm))
            tr_steps.append((mu, m, float(np.mean(tr)) if tr else float("nan")))
    return sim_ok, drops, cdrop, segs, timeouts, tr_steps


def report(name, cs, eps, sim_ok, drops, cdrop, segs, timeouts, tr_steps,
           sigma_mm, disturb_mm, mj, ltr, Kmax):
    tot = len(sim_ok)
    carry = sum(cdrop)
    held = sum(1 for i, (_, _, s) in enumerate(segs)
               if s is not None and mj is not None and mj[i][0])
    nseg = sum(1 for _, _, s in segs if s is not None)
    slip = (np.mean([mj[i][1] for i in range(len(segs)) if segs[i][2] is not None])
            if (mj is not None and nseg) else float("nan"))
    print(f"\n── SigVLA-tiny, {name} (noise σ={sigma_mm:.1f}mm, "
          f"push ±{disturb_mm:.0f}mm) ──")
    print(f"  sim ok {sum(sim_ok)}/{tot} ({sum(sim_ok)/tot*100:.0f}%)  "
          f"drops {sum(drops)} (carry {carry})  timeouts {sum(timeouts)}  "
          f"MuJoCo transport {held}/{nseg} held (slip {slip:.2f}mm)")
    print("  margin→transport-step (mean carry step, mm):")
    pairs = [(margin(mu, m), st) for mu, m, st in tr_steps if st == st]
    if len(pairs) >= 5:
        lo = np.percentile([p[0] for p in pairs], [0, 20, 40, 60, 80, 100])
        for i in range(5):
            sel = [p[1] for p in pairs if lo[i] <= p[0] < lo[i + 1]
                   or (i == 4 and p[0] >= lo[i])]
            if sel:
                print(f"    margin {lo[i]:4.1f}–{lo[i + 1]:4.1f} N: "
                      f"mean step {np.mean(sel):6.1f} mm (n={len(sel)})")
    if ltr is not None:
        res = ltr
        n_band = sum(1 for r in res if r is not None)
        dropped = sum(1 for r in res if r is not None and r[0] is not None)
        ks = [r[0] for r in res if r is not None and r[0] is not None]
        slips = [r[2] for r in res if r is not None]
        med_slip = np.median(slips) if slips else float("nan")
        kstr = (f"median cycle-to-drop {int(np.median(ks))}" if ks
                else "no cycle dropped")
        print(f"  long-transport ×{Kmax}: {dropped:>3d}/{n_band:<3} dropped  "
              f"{kstr:>24}  median slip {med_slip:6.2f} mm")
    print("  per-cell (sim ok / MuJoCo held of replayed transport segments):")
    for ci, (mu, m) in enumerate(cs):
        ok = sim_ok[ci * eps:(ci + 1) * eps]
        held_n = seg = 0
        for i in range(ci * eps, (ci + 1) * eps):
            if segs[i][2] is not None:
                seg += 1
                if mj is not None and mj[i][0]:
                    held_n += 1
        mjs = f"{held_n}/{seg}" if seg else "n/a"
        print(f"    mu {mu:.2f} m {m:.2f} m/μ {m/mu:.2f} "
              f"margin {margin(mu, m):.1f} | {sum(ok)}/{eps} | mj {mjs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpts/vla_tiny/best.pt")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--percept-noise", type=float, default=DEFAULT_NOISE_MM)
    ap.add_argument("--disturb", type=float, default=DEFAULT_DISTURB_MM)
    ap.add_argument("--long-transport", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="also run the clean distribution (σ=0, push=0)")
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_chunk = ckpt["args"]["chunk_size"]     # query length = train-time chunk
    model = SigVLATiny(model_chunk, 4).to(device).eval()
    model.load_state_dict(ckpt["model"])
    norm = ckpt["norm"]
    vla = VLAChunkController(model, norm, args.chunk_size, device)
    tot, tr = 0, 0
    from train_vla_tiny import count_params
    tot, tr = count_params(model)
    print(f"SigVLA-tiny: total {tot/1e6:.1f}M, trainable {tr/1e6:.2f}M "
          f"(frozen SigLIP-B16 encoder)")

    try:
        import mujoco
        from eval_level_a import FixedPadsReplay
        use_mj = not args.no_mujoco
    except Exception as e:          # noqa: E722
        print(f"warning: mujoco unavailable ({e}); sim-only")
        use_mj = False

    cs = cells(args.n_cells, rng, hard_frac=0.5)
    sigma_m, disturb_m = args.percept_noise / 1000.0, args.disturb / 1000.0

    def run(name, sigma_m, disturb_m):
        sim_ok, drops, cdrop, segs, timeouts, tr_steps = collect(
            vla, cs, rng, device, sigma_m, disturb_m, args.chunk_size,
            args.eps)
        mj, ltr = None, None
        if use_mj:
            mj = [adjudicate(seg, mu, m) if seg is not None else (None, None, None)
                  for mu, m, seg in segs]
            if args.long_transport > 0:
                ltr = long_transport(segs, args.long_transport)
        report(name, cs, args.eps, sim_ok, drops, cdrop, segs, timeouts,
               tr_steps, sigma_m * 1000.0, disturb_m * 1000.0, mj, ltr,
               args.long_transport or 0)
        return sim_ok

    run("hard distribution", sigma_m, disturb_m)
    if args.clean:
        run("clean distribution", 0.0, 0.0)


if __name__ == "__main__":
    main()
