"""Phase 6 — analytic Coulomb contact model: the physical modality channel.

Models the grasp as a Coulomb friction cone. The gripper's normal force
F_n = F_max·(1−g) (g ∈ [0,1] openness) resists a tangential load
F_t = m·sqrt((g0·held)² + a²) — the vector-sum of the carried weight and the
arm's inertial load — where a = K2·|Δx| is the PEAK pseudo-acceleration the
plant exerts on the object during a commanded step Δx = target − reseed, and
τ is the arm's step-settle time. Slip occurs when F_t exceeds the friction
capacity μ·F_n. The NCA observes a *physics context* vector
[μ, m, F_n, slip_risk] + one-hot contact mode each step (the modality
channel), and the training slip loss backprops the soft slack
relu(F_t − μ·F_n) (equivalently relu(slip_risk − 1)) through g and Δx, so
the network learns to grip tighter (g→0) and move slower (smaller |Δx|)
when its grip margin is thin — implicit action planning.

Honest plant reconstruction: the damped plant
s(t) = target + (state−target)·(1−plant_f)^(t/τ) has continuous acceleration
a(t) = K2·|Δx|·(1−plant_f)^(t/τ) with K2 = (−ln(1−plant_f)/τ)². For the
eval/replay plant (PLANT_F = 0.5) the PEAK acceleration is a = K2·|Δx| — NOT
the older Δx/τ² = 100·Δx form (a ~2× overestimate at τ=0.1 that made the
blind baseline's drops a parameterization artifact: MuJoCo independent replay
showed 0 slip). Two honesty corrections align the analytic load with the
MuJoCo pseudo-force model (scripts/eval_level_a.py):
  (1) a = K2·Δx with the true k² ≈ 48·Δx at τ=0.1 (192·Δx at τ=0.05), and
  (2) vector-sum F_t = m·sqrt((g0·held)² + a²) — weight and inertial load
  are perpendicular, not worst-case scalar-aligned.
τ = 0.05 s is a recalibrated fast-arm regime: peak acceleration ~192·Δx makes
the blind baseline's nominal ~52 mm steps (a ≈ 10 m/s² ≈ 1g) genuinely slip
on the hard tail m/μ > 1.07 of the feasible grid while the aware's contracted
~27 mm steps (boundary m/μ ≈ 1.35) hold — a real, MuJoCo-reproducible band.

The abstract sim has no pad orientation, so the analytic model adopts the
WORST-CASE orientation: the commanded horizontal acceleration is fully
pad-tangential (entirely friction-loaded). The MuJoCo replay implements this
via a 90° yaw of the recorded trajectory (transport along x would otherwise
be borne by the contact normal — capture — and never slip).

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
TAU = 0.05        # s, arm step-settle time (fast-arm regime, see docstring)
PLANT_F = 0.5     # eval/replay plant lag; peak accel a = K2·|Δx| matches MuJoCo
K2 = (-np.log(1.0 - PLANT_F) / TAU) ** 2   # ≈ 192 m/s² per m of step
MU_RANGE = (0.15, 0.6)
MASS_RANGE = (0.05, 0.35)
GRASPABLE_FRAC = 0.9     # feasibility: m·g0/μ ≤ GRASPABLE_FRAC·F_max
SLIP_THRESH = 0.015      # m, accumulated slip displacement before drop (= MuJoCo replay threshold)
EPS = 1e-3               # F_n floor in slip_risk (avoid div-by-zero)

# Contact modes
FREE, CONTACT, GRASPED, SLIP, DROPPED = range(5)
CONTACT_NAMES = ("free", "contact", "grasped", "slip", "dropped")

# physics_ctx dim = [mu, m, F_n, slip_risk] + one-hot(contact 5)
PHYSICS_CTX_DIM = 9


HARD_LO, HARD_HI = 1.10, 1.35   # m/μ band where a ~52mm blind step slips at τ=0.05 (risk>1); recalibrated


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
    """F_t = m·sqrt((g0·held)² + a²), N — vector-sum of weight + inertial load."""
    return mass * np.sqrt((G0 * float(held)) ** 2 + a ** 2 + EPS)


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
        a = K2 * dx_mag                                          # m/s², peak plant pseudo-accel
        held = self._held(reseed)
        # +eps inside the sqrt: at (held=0, a=0) — the first grasp unroll step,
        # where physics_fn is seeded with prev_reseed == prev_target — the bare
        # sqrt(0) backprops d/d(a²) = inf · 0 = nan and corrupts every param.
        F_t = self.mass * torch.sqrt((self.g0 * held) ** 2 + a ** 2 + self.eps)
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
    a = K2 * float(np.linalg.norm(dx))
    F_t = mass * np.sqrt((G0 * held) ** 2 + a ** 2 + EPS)
    risk = F_t / (mu * F_n + EPS)
    return F_n, F_t, risk
