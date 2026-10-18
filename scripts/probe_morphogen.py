#!/usr/bin/env python3
"""Probe what the NCA morphogen memory actually carries.

Question this answers: is the h3 NULL (morphogen ablation does nothing) because
the FORM cannot hold load, or because the TASK never pressures the net to write
it? We separate these by decoding (mu, m, m/mu) from the carried morphogen state
with a linear probe, on closed-loop rollouts of a trained expert.

  * high R^2 (aware)  -> the net DOES stash load in morphogen; the form holds it,
                         and ctx fully observable every step is what makes it
                         redundant (fix = occlusion, not form).
  * ~0 R^2 (aware)    -> the net never wrote load to morphogen at all; there is
                         no pressure even to store it.
  * ~0 R^2 (blind)    -> control: the blind expert cannot see physics, so its
                         morphogen must not encode load either.

We also probe morphogen -> transport step size (does the memory carry
action-relevant info?) and state(pose) -> mu (sanity: pose carries no load).

Run from the repo root:
  python scripts/probe_morphogen.py --aware-dir checkpoints_phys_honest \
      --blind-dir checkpoints_rollin2 --n-cells 16 --eps 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from sim_env import SimEnv
import proc_sim
from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from eval_implicit_planning import load_experts, cells, HELD


def collect(experts, router, cs, eps, rng, device, max_steps=200):
    """Roll out the closed loop; record the morphogen carried across each env
    step plus its labels. Returns a list of per-step dicts."""
    rows = []
    ep_id = 0
    for mu, m in cs:
        for _ in range(eps):
            obj, place = proc_sim._sample_scene(rng)
            env = SimEnv(rng=rng, plant_f=0.5, physics=True, mu=mu, mass=m)
            env.reset((obj, place))
            scene = np.concatenate([obj, place]).astype(np.float32)
            ctl = BodyGraphController(experts, router, device=device)
            ctl.reset(scene, env.state.copy())
            for t in range(max_steps):
                state = env.state.copy()
                ctx = env.physics_ctx()
                target, info = ctl.step(state, physics_ctx=ctx)
                mg = ctl.morphogen
                if mg is not None:
                    rows.append({
                        "ep": ep_id,
                        "skill": ctl.skill,
                        "si": ctl.step_in_skill,
                        "morph": mg[0].cpu().numpy().ravel().copy(),   # [56]
                        "state": state.copy(),                          # [7] pose+open
                        "ctx": ctx.copy(),                              # [9] (has mu,m here)
                        "mu": float(mu), "m": float(m),
                        "step_mm": float(np.linalg.norm(target[:3] - state[:3]) * 1000.0),
                    })
                env.step(target)
                if info["task_done"] or env.dropped:
                    break
            ep_id += 1
    return rows


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form ridge; returns (R^2 on test, MAE on test) in y units."""
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    n, d = Xtr.shape
    A = Xtr.T @ Xtr + lam * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot, float(np.abs(yte - pred).mean())


def probe_rows(rows, label, feat="morph", split=0.8, seed=0):
    """Episode-level train/test split; ridge R^2 for `label` from `feat`."""
    rng = np.random.RandomState(seed)
    eps = sorted(set(r["ep"] for r in rows))
    rng.shuffle(eps)
    ntr = int(split * len(eps))
    train_eps, test_eps = set(eps[:ntr]), set(eps[ntr:])
    tr = [r for r in rows if r["ep"] in train_eps]
    te = [r for r in rows if r["ep"] in test_eps]
    if not tr or not te:
        return float("nan"), float("nan")
    X = lambda rs: np.stack([r[feat] for r in rs])
    y = lambda rs: np.array([label(r) for r in rs], dtype=np.float64)
    return ridge_r2(X(tr), y(tr), X(te), y(te))


def permuted_r2(rows, label, feat="morph", split=0.8, seed=0):
    """Null baseline: shuffle labels WITHIN the test set so any genuine
    label-feature relationship is destroyed; the R^2 should collapse to ~0."""
    rng = np.random.RandomState(seed + 999)
    eps = sorted(set(r["ep"] for r in rows))
    rng.shuffle(eps)
    ntr = int(split * len(eps))
    test_eps = set(eps[ntr:])
    tr = [r for r in rows if r["ep"] in set(eps[:ntr])]
    te = [r for r in rows if r["ep"] in test_eps]
    if not tr or not te:
        return float("nan")
    X = lambda rs: np.stack([r[feat] for r in rs])
    Xtr, Xte = X(tr), X(te)
    ytr = np.array([label(r) for r in tr], dtype=np.float64)
    yte = np.array([label(r) for r in te], dtype=np.float64)
    rng2 = np.random.RandomState(seed + 7)
    rng2.shuffle(ytr)               # break the feature->label link on train
    r2, _ = ridge_r2(Xtr, ytr, Xte, yte)
    return r2


def summarize(name, rows):
    if not rows:
        print(f"\n[{name}] no samples")
        return
    print(f"\n[{name}]  {len(rows)} steps, {len(set(r['ep'] for r in rows))} episodes")
    targets = [
        ("mu",           lambda r: r["mu"]),
        ("mass",         lambda r: r["m"]),
        ("m/mu",         lambda r: r["m"] / r["mu"]),
        ("margin N",     lambda r: r["mu"] * 15.0 - r["m"] * 9.81),
        ("transport mm", lambda r: r["step_mm"]),
    ]
    print(f"  {'label':>14} | {'morphogen':>9} | {'state(pose)':>11} | {'ctx(9d)':>8}")
    for nm, fn in targets:
        # only held-skill steps carry physics; transport-step label only
        # meaningful on transport, but R2 stays comparable across labels.
        sub = rows
        if nm == "transport mm":
            sub = [r for r in rows if r["skill"] == "transport"]
        r2_m, _ = probe_rows(sub, fn, feat="morph")
        r2_s, _ = probe_rows(sub, fn, feat="state")
        r2_c, _ = probe_rows(sub, fn, feat="ctx")
        print(f"  {nm:>14} | {r2_m:>9.3f} | {r2_s:>11.3f} | {r2_c:>8.3f}")
    # per-skill morphogen -> m/mu breakdown (held skills only)
    print(f"  per-skill morphogen R2(m/mu):")
    for sk in HELD:
        sub = [r for r in rows if r["skill"] == sk]
        if sub:
            r2, _ = probe_rows(sub, lambda r: r["m"] / r["mu"], feat="morph")
            print(f"    {sk:>10}: {r2:.3f}  (n={len(sub)})")
    # null baseline: permuted labels destroy any real relationship -> ~0
    r2_perm = permuted_r2(rows, lambda r: r["mu"], feat="morph")
    print(f"  (permuted-label null R2: {r2_perm:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys_honest")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    aware = load_experts(args.aware_dir, args.blind_dir, device)
    blind = load_experts(None, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    cs = cells(args.n_cells, rng, hard_frac=args.hard_frac)
    print(f"cells: {len(cs)} (hard-frac {args.hard_frac})  "
          f"m/μ range [{min(m/mu for mu,m in cs):.2f}, {max(m/mu for mu,m in cs):.2f}]")

    aware_rows = collect(aware, router, cs, args.eps, rng, device)
    blind_rows = collect(blind, router, cs, args.eps, rng, device)

    summarize("AWARE expert (physics ctx, trained)", aware_rows)
    summarize("BLIND expert (control: no physics input)", blind_rows)

    print("\nreading: morphogen R2 >> state R2 on mu/mass/m-mu means the carried")
    print("memory DOES encode the load (form holds it); ~0 means the task never")
    print("pressed the net to store it.")


if __name__ == "__main__":
    main()
