"""Shared gate metrics and training losses for the body-graph NCA.

Losses run in per-cell NORMALIZED units (divided by sigma) so every DOF
contributes O(1) — pos (m), rot (rad) and openness are balanced, matching
the boundary-pushed semantic alignment. Gate (a) reports absolute mm.
"""

import numpy as np
import torch

from soma.bodygraph_nca import N_CELLS, MORPHOGEN_DIM
from soma.skill_experts import SKILL_REGISTRY, SKILLS
import proc_sim


def masked_mse(pred, gt, masks, sigma):
    """Per-cell-normalized masked MSE — balanced gradient across DOFs."""
    m = masks.unsqueeze(1)                       # [B,1,7]
    d = (pred - gt) / sigma                       # O(1) per cell
    return ((m * d) ** 2).mean()


def loss_batch(model, device, rng, skill, T, batch, w3=2.0, w2=1.0, beta=0.02,
               drift_prob=0.5, drift_mode="terminal"):
    states, goals, masks = proc_sim.make_batch(skill, T, batch, rng)
    states = torch.from_numpy(states).to(device)
    goals = torch.from_numpy(goals).to(device)
    masks = torch.from_numpy(masks).to(device)
    sigma = model.sigma
    clean = states.clone()

    drift = rng.random() < drift_prob
    if drift:
        # Per-sample constant bias (matches gate_a) + per-step noise.
        bias = torch.from_numpy(
            rng.uniform(-0.015, 0.015, (batch, 3)).astype(np.float32)).to(device)
        xi = torch.zeros_like(states)
        xi[..., :3] = bias.unsqueeze(1) + torch.from_numpy(
            rng.normal(0, 0.002, states[..., :3].shape)).to(device)
        xi[..., 3:6] = torch.from_numpy(
            rng.normal(0, 0.02, states[..., 3:6].shape)).to(device)
        states = states + xi

    targets = model.unroll_skill(states, goals, masks, T)

    if drift:
        if drift_mode == "dense":
            # DENSE re-convergence: from the perturbed reseed the target must
            # re-plan onto the ORIGINAL path (clean next pose) and land on B.
            next_pose = torch.cat([clean[:, 1:], goals.unsqueeze(1)], dim=1)
            track = masked_mse(targets, next_pose, masks, sigma)
            term = masked_mse(targets[:, -1], goals, masks, sigma)
            loss = w2 * track + w3 * term
        else:  # terminal-only
            loss = w3 * masked_mse(targets[:, -1], goals, masks, sigma)
    else:
        next_pose = torch.cat([states[:, 1:], goals.unsqueeze(1)], dim=1)
        track = masked_mse(targets, next_pose, masks, sigma)
        term = masked_mse(targets[:, -1], goals, masks, sigma)
        loss = track + w3 * term

    # Jerk smoothing on active cells (never penalize output magnitude).
    jerk = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
    m = masks.unsqueeze(1)
    loss = loss + beta * ((m * (jerk / sigma)) ** 2).mean()
    return loss


def loss_loop(model, device, rng, skill, T, batch, w3=2.0, wd=0.3, beta=0.02,
              drift_prob=0.5, plant_f_set=(0.3, 0.5, 1.0),
              pos_noise=0.001, rot_noise=0.01, open_noise=0.01,
              physics=False, w_slip=5.0, slip_safety=0.5, hard_frac=0.0):
    """Closed-loop roll-in training loss.

    Reseed only advances through a lagging plant (reseed += f·(target−reseed))
    applying the expert's OWN targets — so a controller that echoes the state
    never reaches the goal and the terminal loss stays large. The expert must
    learn to drive. plant_f is sampled per batch so the controller is robust
    to actuator lag; a persistent mid-loop pos drift (t0 ~ 30-70%) teaches
    re-convergence (gate b semantics) in the closed loop.

    physics: enable the Coulomb slip modality. A per-episode (μ, m) is sampled
    for every batch element, the physics context channel feeds the NCA each
    step (from the previous step's command — one-step lag), and a slip loss
    w_slip·mean(relu(slip_risk − 1)²) penalizes commanded steps whose inertial
    load exceeds the friction capacity while the object is carried. This is
    the gradient that teaches the network to grip tighter and move slower on
    low-margin cells (implicit action planning).
    """
    if physics:
        states, goals, masks, phy = proc_sim.make_batch(
            skill, T, batch, rng, with_physics=True, hard_frac=hard_frac)
    else:
        states, goals, masks = proc_sim.make_batch(skill, T, batch, rng)
    states = torch.from_numpy(states).to(device)
    goals = torch.from_numpy(goals).to(device)
    masks = torch.from_numpy(masks).to(device)
    sigma = model.sigma
    plant_f = float(rng.choice(plant_f_set))
    drift_t0, drift_bias = None, None
    if rng.random() < drift_prob:
        drift_t0 = int(T * rng.uniform(0.3, 0.7))
        drift_bias = rng.uniform(-0.015, 0.015, (batch, 3)).astype(np.float32)
    if physics:
        from soma.physics import CoulombPhysics
        phys = CoulombPhysics(torch.from_numpy(phy[:, 0]).to(device),
                              torch.from_numpy(phy[:, 1]).to(device), skill)
        targets, slip_terms = model.unroll_loop(
            states, goals, masks, T, plant_f,
            drift_t0=drift_t0, drift_bias=drift_bias,
            pos_noise=pos_noise, rot_noise=rot_noise, open_noise=open_noise,
            physics_fn=phys)
    else:
        targets = model.unroll_loop(
            states, goals, masks, T, plant_f,
            drift_t0=drift_t0, drift_bias=drift_bias,
            pos_noise=pos_noise, rot_noise=rot_noise, open_noise=open_noise)
    term = w3 * masked_mse(targets[:, -1], goals, masks, sigma)
    # Dense hold-at-goal: once the reseed is near the goal the target must stay
    # on it for ALL phases. Terminal-only leaves the mid-roll-in hold steps
    # weakly supervised, so the target can drift away as phase advances (seen
    # as a 7-9mm steady-state bias at f=0.3). Early (far) steps contribute an
    # unsatisfiable-but-gradiented term that only pushes max drive — harmless.
    dense = wd * masked_mse(targets, goals.unsqueeze(1).expand_as(targets), masks, sigma)
    loss = term + dense
    jerk = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
    m = masks.unsqueeze(1)
    loss = loss + beta * ((m * (jerk / sigma)) ** 2).mean()
    if physics:
        from soma.physics import SLIP_THRESH
        # Accumulated slip displacement over the rollout, hinged on the same
        # drop threshold the eval sim uses (safety factor leaves eval margin).
        slip_disp = torch.stack(slip_terms).sum(dim=0)            # [B] m
        slip = w_slip * torch.relu(slip_disp / SLIP_THRESH - slip_safety).mean()
        loss = loss + slip
    return loss


def _drive_loop(model, device, rng, skill, plant_f, drift, tol=5e-3,
                max_steps=200, n=10):
    """One closed-loop drive test: reseed = on-path start; the plant applies
    the expert's OWN targets (reseed += f·(target−reseed) + noise). Success =
    masked pos err < tol within max_steps. drift: optional one-time mid-loop
    pos knock (m). This is the gate the roll-in-trained experts actually
    satisfy — the old on-path gate_a is OOD for them."""
    lo, hi = SKILL_REGISTRY[skill]["duration"]
    model.eval()
    was_firing = model.firing
    model.firing = 1.0
    ok = 0
    for _ in range(n):
        T = int(rng.randint(lo, hi + 1))
        states, goals, masks = proc_sim.make_batch(skill, T, 1, rng)
        st = torch.from_numpy(states)[:, 0].clone().to(device)
        gl = torch.from_numpy(goals).to(device)
        mk = torch.from_numpy(masks).to(device)
        morph = None
        t0 = int(T * rng.uniform(0.3, 0.7)) if drift else -1
        bias = (torch.from_numpy(rng.uniform(-drift, drift, (1, 3)).astype(np.float32)).to(device)
                if drift else None)
        noise = torch.tensor([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01],
                             device=device).unsqueeze(0)
        reached = False
        for t in range(max_steps):
            u = (t + 1) / T
            phase = torch.tensor([np.cos(2 * np.pi * u), np.sin(2 * np.pi * u),
                                  1.0 / T], device=device, dtype=torch.float32).unsqueeze(0)
            target, morph = model.relax(st, gl, mk, phase, morph)
            morph = morph * model.morphogen_decay
            st = st + plant_f * (target - st) + noise * torch.randn(1, 7, device=device)
            st = torch.cat([st[..., :6], st[..., 6:7].clamp(0, 1)], dim=-1)
            if t == t0 and bias is not None:
                st = st + torch.cat([bias, torch.zeros(1, 4, device=device)], dim=-1)
            if t > 5:
                err = (mk[0, :3] * (st[0, :3] - gl[0, :3])).abs().max().item()
                if err < tol:
                    reached = True
                    break
        ok += reached
    model.firing = was_firing
    return ok, n


def gate_a_loop(model, device, rng, n=10):
    """Gate (a) — closed-loop skill convergence (plant_f=0.5, no drift)."""
    return [_drive_loop(model, device, rng, s, 0.5, None, n=n) for s in SKILLS]


def gate_b_loop(model, device, rng, n=10, drift=0.015):
    """Gate (b) — closed-loop drift re-convergence (one-time 15mm mid-loop)."""
    return [_drive_loop(model, device, rng, s, 0.5, drift, n=n) for s in SKILLS]


@torch.no_grad()
def gate_a(model, device, rng, n=10):
    """Re-convergence gate: reseeds pushed OFF the demo path (bias + noise);
    the target must still reach the absolute skill goal B. Echo/reseed random
    init fails with error ≈ |bias| — the honest discriminator."""
    model.eval()
    out = []
    for skill in SKILLS:
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        errs, convs = [], []
        for _ in range(n):
            T = int(rng.randint(lo, hi + 1))
            states, goals, masks = proc_sim.make_batch(skill, T, 1, rng)
            states = states.copy()
            states[:, :, :3] += rng.uniform(-0.015, 0.015, 3).astype(np.float32)
            states[:, :, 3:6] += rng.normal(0, 0.02, states[:, :, 3:6].shape).astype(np.float32)
            states[:, :, :3] += rng.normal(0, 0.002, (T, 3)).astype(np.float32)
            states = torch.from_numpy(states).to(device)
            goals = torch.from_numpy(goals).to(device)
            masks = torch.from_numpy(masks).to(device)
            targets = model.unroll_skill(states, goals, masks, T)
            m = masks[:, :3]
            pos_err = float(((m * (targets[:, -1, :3] - goals[:, :3])) ** 2).sum(-1).sqrt().item() * 1000.0)
            errs.append(pos_err)
            convs.append(pos_err < 10.0)
        out.append((skill, float(np.mean(errs)), float(np.mean(convs))))
    return out


@torch.no_grad()
def gate_b(model, device, rng, epsilons=(0.01, 0.02), firings=(1.0, 0.5),
           n=10):
    """Damage-recovery gate (paper damage-recovery analogy).

    The trajectory starts CLEAN, then a mid-skill step perturbation of size
    epsilon is injected from t0 ~ 30-70% of the episode (the 'damage event'),
    and the unroll runs with optional firing randomness (dropped updates, sim-
    style robustness). The target must still re-converge to the absolute skill
    goal B and NOT diverge. gate_a perturbs from the very start with firing=1.0;
    gate_b tests generalization to a LATER perturbation + damaged updates.
    """
    was_firing = model.firing
    out = []
    for skill in SKILLS:
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        for eps in epsilons:
            for firing in firings:
                errs, convs, divs = [], [], []
                for _ in range(n):
                    T = int(rng.randint(lo, hi + 1))
                    states, goals, masks = proc_sim.make_batch(skill, T, 1, rng)
                    states = states.copy()
                    t0 = int(T * rng.uniform(0.3, 0.7))
                    states[:, t0:, :3] += rng.uniform(-eps, eps, 3).astype(np.float32)
                    states[:, :, :3] += rng.normal(0, 0.002, states[:, :, :3].shape).astype(np.float32)
                    states[:, :, 3:6] += rng.normal(0, 0.02, states[:, :, 3:6].shape).astype(np.float32)
                    states = torch.from_numpy(states).to(device)
                    goals = torch.from_numpy(goals).to(device)
                    masks = torch.from_numpy(masks).to(device)
                    if firing < 1.0:
                        model.train()            # enable update dropout
                    else:
                        model.eval()
                    model.firing = firing
                    targets = model.unroll_skill(states, goals, masks, T)
                    m = masks[:, :3]
                    pos_err = float(((m * (targets[:, -1, :3] - goals[:, :3])) ** 2).sum(-1).sqrt().item() * 1000.0)
                    errs.append(pos_err)
                    convs.append(pos_err < 10.0)
                    divs.append(pos_err > 50.0)
                out.append((skill, float(eps), firing, float(np.mean(errs)),
                            float(np.mean(convs)), float(np.mean(divs))))
    model.eval()
    model.firing = was_firing
    return out


@torch.no_grad()
def gate_c(model, device, rng, T=50):
    """Morphogen emergence: a dim qualifies only if |corr(h,u)|>0.5 across env
    steps AND within-step activity > 0.05 (guards the '16k linear re-implem')."""
    model.eval()
    states, goals, masks = proc_sim.make_batch("approach", T, 1, rng)
    states = torch.from_numpy(states).to(device)
    goals = torch.from_numpy(goals).to(device)
    masks = torch.from_numpy(masks).to(device)
    _, morph_log = model.unroll_skill(states, goals, masks, T, record_morph=True)
    mlog = np.stack([np.stack([mlk[0].cpu().numpy() for mlk in step]) for step in morph_log])  # [T,K,7,8]
    mfin = mlog[:, -1]
    activity = mlog.std(axis=1).mean(axis=0)     # [7,8]
    u = (np.arange(1, T + 1) / T).astype(np.float32)
    report = []
    for cell in range(N_CELLS):
        cors = np.array([np.corrcoef(mfin[:, cell, d], u)[0, 1] if np.std(mfin[:, cell, d]) > 1e-6 else 0.0
                         for d in range(MORPHOGEN_DIM)])
        cors = np.nan_to_num(cors)
        qual = (np.abs(cors) > 0.5) & (activity[cell] > 0.05)
        report.append((cell, float(np.abs(cors).max()),
                       int(qual.sum()), float(activity[cell].max())))
    return report
