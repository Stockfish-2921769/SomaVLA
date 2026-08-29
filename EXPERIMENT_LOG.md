# Experiment Log — Soma

## Phase 6 — Coulomb physics modality + implicit action planning (2026-08-28/29)

Goal: give the homogeneous body-graph NCA a physical modality channel
(friction / dynamics / contact) and probe whether it learns *implicit action
planning* (load-adaptive grip/speed) rather than hand-written rules.

Design (see plan `binary-roaming-gem.md`): analytic differentiable Coulomb
model (`soma/physics.py`), per-step `physics_ctx[9] = [μ, m, F_n, slip_risk] +
one-hot(contact 5)` broadcast into the update rule (54→63 dim), and a slip
loss `relu(Σ relu(risk−1)·|Δx|·held / SLIP_THRESH − 0.5)` training the 4 held
skills (grasp/lift/transport/place). F_t = m·(g0·held + |Δx|/τ²) linear
worst-case-aligned (quadrature form makes the inertial term a 2nd-order
correction — unobservable). τ=0.1s calibrated so at full grip the blind
baseline risk ≈ m/μ, i.e. genuine slip onset at m/μ > 1 (margin ≲ 1.1 N).

### Round A — first build + first gate (h)

- Trained `checkpoints_phys/` (uniform feasible (μ,m), w_slip=5.0, 1500 steps).
- **gate (h) v1: blind 63/80 (79%) vs aware 64/80 (80%) — no h1 gap.**
- Diagnosis:
  1. **Release artifact (both agents)**: `_coulomb_step` checked `F_n < held_req`
     (→ drop) *before* `g ≥ 0.5` (→ release). Since g≥0.5 implies F_n ≤ 7.5 N <
     held_req on every hard cell, the release branch was dead code — every
     placement over the target was scored a drop. → **Fix 1**: reorder
     GRASPED/SLIP branches to check the intentional release first.
  2. **Cell sampling misses the slip band**: uniform feasible (μ,m) has mean
     m/μ≈0.6; the blind only genuinely slips at m/μ>1. With the artifact fixed,
     blind 77/80 vs aware 78/80 — the remaining hard cells (margin 2–3 N) don't
     slip for the blind at its ~52 mm steps.
- Round A already showed the core signatures: **h2** aware transport step
  contracts with margin (16.8→23.4 mm) vs blind flat ~53 mm; **h3** morphogen
  ablation null.

### Round B — sim modeling fixes + hard-tail experiment + final

- **Fix 2 — position-aware detach**: `dropped` is no longer unconditional. The
  object detaches wherever support is lost (F_n < held_req or slip past
  threshold); success is decided by *where* obj_final lands (within place_tol
  of the target = placed). This is what let the aware actually recover the
  extreme cells (its release opening passes through the F_n < held_req window
  before g reaches 0.5).
- **hard_frac experiment (discarded)**: retrained with 50% m/μ∈[1.02,1.32]
  hard-tail samples (`checkpoints_phys2/`). Result: globally conservative —
  transport step flat ~20 mm, h2 contraction lost, still dropped the extreme
  cells. Uniform training is strictly better (keeps margin-conditional
  contraction *and* recovers hard cells).
- Tightened eval hard band to m/μ∈[1.08,1.35] so the blind's genuine slip cells
  are densely sampled.
- **Final gate (h)** (uniform experts + position-aware detach, 40 cells × 3 eps):
  - **h1**: aware 119/120 (99%) vs blind 106/120 (88%); dropped 1 vs 14.
    Aware recovers the m/μ>1 hard tail. Note: ctx-zeroed aware also 119/120 —
    the hard-tail capability comes mostly from the slip-loss-trained
    conservative prior, not from inference-time ctx.
  - **h2**: with ctx, transport step contracts 12.2→23.8 mm with margin (~2×,
    monotone); ctx-zeroed aware flat ~19 mm; blind flat ~53 mm. The
    margin-adaptive plan is driven specifically by the modality channel
    (same-expert on/off ctx control).
  - **h3 (negative)**: morphogen warm-start ablation 119→117/120. Since (μ,m)
    is constant per episode and fully present in the ctx every step, the plan
    is a context→action policy re-derived within a single relax — not carried
    in cross-step morphogen memory.
- **Regression**: `eval_closed_loop.py --ckpt-dir checkpoints_phys` — gates
  e/f/g all pass on the Coulomb sim (20/20; robust 20/20; ~944–951 Hz GPU).
  Blind rollin2 unchanged (100% on the easy default cell).

### Conclusion

A homogeneous NCA given a physical modality channel learns an implicit
load-adaptive speed plan (margin→step contraction, ~2×) and thereby recovers
hard cells the blind baseline drops (99% vs 88%). The plan lives in the
context→action policy, not morphogen memory. Keep `checkpoints_phys/` (uniform
training); `checkpoints_phys2/` (hard_frac) is a discarded experiment.
