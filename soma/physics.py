"""Phase 6 — analytic Coulomb contact model: the physical modality channel.

Models the grasp as a Coulomb friction cone. The gripper's normal force
F_n = F_max·(1−g) (g ∈ [0,1] openness) resists a tangential load
F_t = m·(g0·held + |Δx|/τ²), where Δx = target − reseed is the commanded
EEF displacement and τ is the arm's step-settle time (how long the arm
takes to arrest a commanded step). Slip occurs when F_t exceeds the
friction capacity μ·F_n. The NCA observes a *physics context* vector
[μ, m, F_n, slip_risk] + one-hot contact mode each step (the modality
channel), and the training slip loss backprops the soft slack
relu(F_t − μ·F_n) (equivalently relu(slip_risk − 1)) through g and Δx, so
the network learns to grip tighter (g→0) and move slower (smaller |Δx|)
when its grip margin is thin — implicit action planning.

Load calibration: the plan proposed F_t = sqrt((m·g0)² + (m·|Δx|/τ²)²)
(perpendicular quadrature). With gravity dominating (m·g0 ≈ 3 N vs the
inertial term ≈ 0.3–1 N at realistic steps) the quadrature form makes the
inertial contribution a second-order correction that never crosses the
grip margin — the 'blind slips, aware holds' effect would be unobservable.
The linear (worst-case-aligned) load m·(g0 + |Δx|/τ²) makes the inertial
term directly commensurable with the margin, which is what the gate (h)
experiment needs. τ = 0.1 s is calibrated from the measured closed-loop
per-step displacement (~25–40 mm) so the blind baseline's nominal ~52 mm
steps genuinely slip exactly on the hard tail of the feasible (m, μ) grid —
at full grip risk ≈ m/μ, so slip onset is m/μ > 1 (i.e. margin ≲ 1.1 N),
measured empirically — while easy cells stay far below it. The slowed
motion the aware NCA must learn (steps ~10–20 mm on hard cells) keeps the
risk under 1 while staying within the skill's step budget.

Scene params: μ ~ U(0.15, 0.6), m ~ U(0.05, 0.35) kg, F_max = 15 N,
g0 = 9.81, feasible iff m·g0/μ ≤ 0.9·F_max (the object is graspable at full
grip). The hard tail of that range has ~10% grip margin.

Contact state machine (numpy, eval-time in SimEnv):
  free → contact → grasped → slip → dropped; grasped → released when g ≥ 0.5.
"""

import numpy as np
import torch

# Mirrors scripts/proc_sim.py / scripts/sim_env.py.
CONTACT_Z = 0.52
GRASP_Z = 0.58

F_MAX = 15.0      # N, gripper maximum normal force at full closure
G0 = 9.81         # m/s², gravitational acceleration
TAU = 0.1         # s, arm step-settle time (calibrated, see module docstring)
MU_RANGE = (0.15, 0.6)
MASS_RANGE = (0.05, 0.35)
GRASPABLE_FRAC = 0.9     # feasibility: m·g0/μ ≤ GRASPABLE_FRAC·F_max
SLIP_THRESH = 0.005      # m, accumulated slip displacement before drop
EPS = 1e-3               # F_n floor in slip_risk (avoid div-by-zero)

# Contact modes
FREE, CONTACT, GRASPED, SLIP, DROPPED = range(5)
CONTACT_NAMES = ("free", "contact", "grasped", "slip", "dropped")

# physics_ctx dim = [mu, m, F_n, slip_risk] + one-hot(contact 5)
PHYSICS_CTX_DIM = 9


HARD_LO, HARD_HI = 1.02, 1.32   # m/μ band where a ~52mm blind step slips (risk≈m/μ>1)


def sample_physics(rng, hard_frac=0.0):
    """One feasible (mu, mass) scene. Rejection sample constrained to graspable
    objects (m·g0/μ ≤ 0.9·F_max). With hard_frac>0, that fraction of samples is
    drawn from the HARD tail m/μ ∈ [HARD_LO, HARD_HI] — the band the blind
    baseline's nominal steps genuinely slip on (risk ≈ m/μ > 1 at full grip),
    which uniform feasible sampling almost never hits (mean m/μ ≈ 0.6)."""
    while True:
        if rng.random() < hard_frac:
            mu = rng.uniform(0.2, 0.34)
            m = rng.uniform(HARD_LO, HARD_HI) * mu
        else:
            mu = rng.uniform(*MU_RANGE)
            m = rng.uniform(*MASS_RANGE)
        if 0.05 <= m <= 0.35 and m * G0 / mu <= GRASPABLE_FRAC * F_MAX:
            return float(mu), float(m)


def normal_force(g):
    """F_n = F_max·(1−g), N. Differentiable in g."""
    return F_MAX * (1.0 - g)


def tangential_load(mass, held, a):
    """F_t = m·(g0·held + a), N — worst-case-aligned Coulomb load."""
    return mass * (G0 * float(held) + a)


def slip_risk(F_n, F_t, mu):
    return F_t / (mu * F_n + EPS)


def soft_slack(F_n, F_t, mu):
    """relu(F_t − μ·F_n) normalized to F_max — the differentiable slip signal."""
    return torch.relu(F_t - mu * F_n) / F_MAX


class CoulombPhysics:
    """Per-batch differentiable physics for one training rollout.

    physics_fn(reseed, target, t) → (ctx[B,9], slip_incr[B]).
    ctx is built from the command applied at the PREVIOUS env step
    (reseed/target passed in) — a one-step sensory lag matching the sim:
    the expert sees the slip_risk its last command produced, not the one it
    is about to emit.

    Held gating per skill:
      approach/release  never held (no physics; ctx = free, slip 0).
      grasp             held once the EEF has descended to the object
                        (reseed z < GRASP_Z): the object is being carried,
                        so the network must provide F_n ≥ m·g0/μ by then.
      lift/transport/place  always held (object assumed grasped).
    """

    def __init__(self, mu, mass, skill, tau=TAU, F_max=F_MAX, g0=G0,
                 grasp_z_gate=GRASP_Z + 0.01, eps=EPS, slip_thresh=SLIP_THRESH):
        self.mu = mu                      # [B] tensor
        self.mass = mass                  # [B] tensor
        self.skill = skill
        self.tau = tau
        self.F_max = F_max
        self.g0 = g0
        self.grasp_z_gate = grasp_z_gate
        self.eps = eps
        self.slip_thresh = slip_thresh

    # ── per-step quantities ─────────────────────────────────────────────
    def _held(self, reseed):
        """[B] float 0/1: is the object being carried at this reseed?"""
        if self.skill in ("lift", "transport", "place"):
            return torch.ones_like(self.mu)
        if self.skill == "grasp":
            return (reseed[:, 2] < self.grasp_z_gate).to(self.mu.dtype)
        return torch.zeros_like(self.mu)

    def _contact(self, reseed, held):
        """[B] long contact-mode index (FREE/CONTACT/GRASPED)."""
        if self.skill in ("lift", "transport", "place"):
            return torch.full_like(self.mu, GRASPED, dtype=torch.long)
        if self.skill == "grasp":
            return torch.where(held > 0.5,
                               torch.full_like(self.mu, GRASPED, dtype=torch.long),
                               torch.full_like(self.mu, CONTACT, dtype=torch.long))
        return torch.zeros_like(self.mu, dtype=torch.long)

    def _step(self, reseed, target):
        """(F_n[B], F_t[B], slip_risk[B], held[B], contact[B])."""
        g = reseed[:, 6]
        F_n = self.F_max * (1.0 - g)
        dx_mag = (target[:, :3] - reseed[:, :3]).norm(dim=-1)   # m
        a = dx_mag / self.tau ** 2                               # m/s²
        held = self._held(reseed)
        F_t = self.mass * (self.g0 * held + a)
        risk = F_t / (self.mu * F_n + self.eps)
        contact = self._contact(reseed, held)
        return F_n, F_t, risk, held, contact

    def __call__(self, reseed, target, t):
        """→ (ctx[B,9], slip_incr[B]). ctx values normalized to O(1);
        slip_incr = relu(slip_risk−1)·|Δx|·held is the per-step slip
        DISPLACEMENT accumulation increment (same quantity the eval sim
        accumulates before dropping the object) — the training slip loss sums
        these over the rollout and hinges on the drop threshold, so the
        gradient mirrors the real drop condition instead of a diluted
        per-step mean."""
        F_n, F_t, risk, held, contact = self._step(reseed, target)
        onehot = torch.nn.functional.one_hot(
            contact, num_classes=5).to(self.mu.dtype)             # [B,5]
        ctx = torch.stack([
            self.mu / 0.6, self.mass / 0.35, F_n / self.F_max,
            risk.clamp(0.0, 5.0) / 5.0,
        ], dim=-1)                                                 # [B,4]
        ctx = torch.cat([ctx, onehot], dim=-1)                     # [B,9]
        dx = (target[:, :3] - reseed[:, :3]).norm(dim=-1)          # m
        slip_incr = torch.relu(risk - 1.0) * dx * held             # [B]
        return ctx, slip_incr


def numpy_slip_metrics(state, target, mu, mass, held):
    """numpy F_n / F_t / slip_risk for a single step (SimEnv uses this)."""
    g = float(state[6])
    F_n = F_MAX * (1.0 - g)
    dx = np.asarray(target[:3], dtype=np.float64) - np.asarray(state[:3], dtype=np.float64)
    a = float(np.linalg.norm(dx)) / TAU ** 2
    F_t = mass * (G0 * held + a)
    risk = F_t / (mu * F_n + EPS)
    return F_n, F_t, risk
