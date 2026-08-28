"""Body-graph NCA: a homogeneous cellular field over the 7 EEF-pose DOFs.

Adapts the Smart Cellular Bricks recipe (Nat. Commun. 2026) to a 7-cell body
graph (px, py, pz, rx, ry, rz, g):

  * Identical shared local rule on every cell — NO type embedding. Role
    differentiation is expected to EMERGE from graph topology + morphogens.
  * Three-channel cell state [alpha | value | morphogen(8)]:
      alpha      activity gate set by the skill boundary; never written by
                 the network, never read out. An inactive cell is frozen
                 (its DOF holds position / openness).
      value      normalized displacement of the DOF from its current reseed.
      morphogen  cross-step warm-start trajectory memory (LayerNorm'd).
  * tanh-bounded additive updates + zero-initialized final projection layer
    (near-zero initial output = safe identity/attractor start).
  * Stochastic firing in training (robustness to dropped updates), firing=1
    at inference (deterministic).

Execution model (receding horizon): each env step, reseed = current absolute
state; value re-seeds to 0; K relaxation steps push value toward the
normalized goal (skill target B relative to reseed); readout
target = reseed + value·sigma. Morphogen carries memory across env steps
(×decay warm-start, gradients kept); value has no cross-step gradient.
"""

import numpy as np
import torch
import torch.nn as nn

N_CELLS = 7
PX, PY, PZ, RX, RY, RZ, G = range(7)
MORPHOGEN_DIM = 8
CELL_STATE_DIM = 1 + 1 + MORPHOGEN_DIM  # alpha + value + morphogen

# 10 undirected edges: kinematic chain + geometric shortcuts.
EDGES = (
    (PX, PY), (PY, PZ), (PZ, RX), (RX, RY), (RY, RZ), (RZ, G),
    (PX, RX), (PY, RY), (PZ, RZ), (PZ, G),
)

# Per-DOF physical scale pushed to the boundary (the rule only sees the
# normalized abstract displacement dynamics): pos in m, rot in rad, grip in
# openness units. Readout multiplies value back by sigma.
SIGMA = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)


def _adjacency():
    adj = [[] for _ in range(N_CELLS)]
    for a, b in EDGES:
        adj[a].append(b)
        adj[b].append(a)
    return tuple(tuple(a) for a in adj)


# Degree per cell: px→2, py→3, pz→3, rx→3, ry→3, rz→3, g→2
ADJACENCY = _adjacency()


class _Perception(nn.Module):
    """g(s_j): per-cell state → message. Shared across all cells."""
    def __init__(self, in_dim=CELL_STATE_DIM, out_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())

    def forward(self, s):          # [B, 7, 10]
        return self.net(s)         # [B, 7, 40]


class _Update(nn.Module):
    """f(own, agg, goal, phase): writes value + morphogen. Final layer is
    zero-initialized and the output is tanh-bounded (stable start)."""
    def __init__(self, in_dim, hidden=96, out_dim=1 + MORPHOGEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.tanh(self.net(x))


class BodyGraphNCA(nn.Module):
    def __init__(self, k_steps=8, firing=0.5, morphogen_dim=MORPHOGEN_DIM,
                 attractor_lambda=0.1, dt=0.2, morphogen_decay=0.95, hidden=96,
                 phase_dim=3):
        super().__init__()
        self.k_steps = k_steps
        self.firing = firing
        self.morphogen_dim = morphogen_dim
        self.attractor_lambda = attractor_lambda
        self.dt = dt
        self.morphogen_decay = morphogen_decay
        self.phase_dim = phase_dim

        self.perception = _Perception()
        update_in = CELL_STATE_DIM + 40 + 1 + phase_dim  # own + agg + goal + phase
        self.update = _Update(update_in, hidden=hidden)

        self.register_buffer("sigma", torch.tensor(SIGMA, dtype=torch.float32))
        # Adjacency as a [7,7] degree-normalized mean matrix for a single
        # einsum (vs a 7-iter Python gather loop — 65x faster on GPU).
        adj_w = torch.zeros(N_CELLS, N_CELLS, dtype=torch.float32)
        for i, nb in enumerate(ADJACENCY):
            adj_w[i, nb] = 1.0
        self.register_buffer("adj_w", adj_w / adj_w.sum(dim=-1, keepdim=True))

    # ─────────────────────────────────────────────────────────────────────
    # Cell graph
    # ─────────────────────────────────────────────────────────────────────
    def _aggregate(self, msg):
        """Degree-normalized neighbor message mean — single einsum."""
        return torch.einsum("ij,bjd->bid", self.adj_w, msg)   # [B,7,40]

    # ─────────────────────────────────────────────────────────────────────
    # Single env step: K relaxation iterations
    # ─────────────────────────────────────────────────────────────────────
    def relax(self, reseed, goal_abs, alpha_mask, phase, morphogen=None,
              record_morph=False):
        """One env step. Returns absolute target [B,7] and morphogen_out [B,7,8].

        reseed:    [B,7] current absolute state (pos+rot+openness)
        goal_abs:  [B,7] absolute skill target
        alpha_mask:[B,7] 0/1 activity gate
        phase:     [B,3] cyclic phase + 1/duration
        morphogen: [B,7,8] warm-start from the previous env step (None → zeros)
        """
        B = reseed.shape[0]
        dev, dtype = reseed.device, reseed.dtype
        goal_norm = (goal_abs - reseed) / self.sigma               # [B,7]
        alpha = (alpha_mask > 0.5).to(dtype).unsqueeze(-1)         # [B,7,1]
        v = torch.zeros(B, N_CELLS, 1, device=dev, dtype=dtype)
        m = (morphogen if morphogen is not None
             else torch.zeros(B, N_CELLS, self.morphogen_dim, device=dev, dtype=dtype))
        ph = phase.unsqueeze(1).expand(B, N_CELLS, self.phase_dim)  # [B,7,3]
        gn = goal_norm.unsqueeze(-1)                               # [B,7,1]

        mlog = [] if record_morph else None
        for _ in range(self.k_steps):
            s = torch.cat([alpha, v, m], dim=-1)                    # [B,7,10]
            msg = self.perception(s)                                # [B,7,40]
            agg = self._aggregate(msg)                              # [B,7,40]
            u = torch.cat([s, agg, gn, ph], dim=-1)                 # [B,7,54]
            d = self.update(u)                                      # [B,7,9] tanh-bound
            if self.training and self.firing < 1.0:
                gate = alpha * (torch.rand(B, N_CELLS, 1, device=dev) < self.firing).to(dtype)
            else:
                gate = alpha
            dv, dm = d[..., :1], d[..., 1:]
            # dt scaling → fixed point v* = dt·tanh(f)/λ ≈ 2·tanh(f) (bounded),
            # not tanh(f)/λ ≈ 10·tanh(f) which blows up to ±0.5 m.
            v = (v + self.dt * dv * gate) - self.attractor_lambda * v
            m = m + self.dt * dm * gate
            if record_morph:
                mlog.append(m.clone())

        target = reseed + v[..., 0] * self.sigma                   # [B,7] abs pose+openness
        return (target, m, mlog) if record_morph else (target, m)

    # ─────────────────────────────────────────────────────────────────────
    # BPTT over a full skill (receding horizon, morphogen warm-start)
    # ─────────────────────────────────────────────────────────────────────
    def unroll_skill(self, states, goal_abs, alpha_mask, duration,
                     record_morph=False):
        """states: [B,T,7] actual states (reseed each step), goal_abs: [B,7],
        alpha_mask: [B,7], duration: scalar expected length.
        Returns targets [B,T,7]; optionally morph log {env_step: [K] tensors}.
        """
        B, T, _ = states.shape
        dev, dt = states.device, states.dtype
        morphogen = torch.zeros(B, N_CELLS, self.morphogen_dim, device=dev, dtype=dt)
        targets = []
        morph_log = [] if record_morph else None
        for t in range(T):
            u = (t + 1) / duration
            phase = torch.tensor(
                [np.cos(2 * np.pi * u), np.sin(2 * np.pi * u), 1.0 / duration],
                device=dev, dtype=dt).unsqueeze(0).expand(B, -1)
            if record_morph:
                target, morphogen, mlog = self.relax(
                    states[:, t], goal_abs, alpha_mask, phase, morphogen, record_morph=True)
                morph_log.append(mlog)
            else:
                target, morphogen = self.relax(
                    states[:, t], goal_abs, alpha_mask, phase, morphogen)
            targets.append(target)
            morphogen = morphogen * self.morphogen_decay           # warm-start, grad kept
        targets = torch.stack(targets, dim=1)                      # [B,T,7]
        return (targets, morph_log) if record_morph else targets

    # ─────────────────────────────────────────────────────────────────────
    # Closed-loop roll-in (BPTT through the plant) — the training mode that
    # forces the expert to DRIVE. unroll_skill reseeds from on-path states,
    # so the reseed walks to the goal by itself and the terminal loss is
    # satisfied by echoing (target ≈ reseed). Here the reseed only advances
    # through a lagging plant applying this expert's own targets, so echo
    # fails catastrophically and the network must learn to converge on goal.
    # ─────────────────────────────────────────────────────────────────────
    def unroll_loop(self, states, goal_abs, alpha_mask, duration, plant_f,
                    drift_t0=None, drift_bias=None,
                    pos_noise=0.001, rot_noise=0.01, open_noise=0.01):
        """states: [B,T,7] (only states[:, 0] seeds the loop), goal_abs: [B,7],
        alpha_mask: [B,7], duration: scalar, plant_f: scalar actuator gain.
        Returns targets [B,T,7]. Reseed evolves as reseed += f·(target−reseed)
        + measurement noise; a mid-loop pos drift (drift_t0, drift_bias) is
        optionally injected to teach re-convergence."""
        B, T, _ = states.shape
        dev, dt = states.device, states.dtype
        reseed = states[:, 0].clone()
        morphogen = torch.zeros(B, N_CELLS, self.morphogen_dim, device=dev, dtype=dt)
        noise_sigma = torch.tensor([pos_noise] * 3 + [rot_noise] * 3 + [open_noise],
                                   device=dev, dtype=dt)
        if drift_bias is not None:
            drift_bias = torch.as_tensor(drift_bias, device=dev, dtype=dt)
        targets = []
        for t in range(T):
            u = (t + 1) / duration
            phase = torch.tensor(
                [np.cos(2 * np.pi * u), np.sin(2 * np.pi * u), 1.0 / duration],
                device=dev, dtype=dt).unsqueeze(0).expand(B, -1)
            target, morphogen = self.relax(reseed, goal_abs, alpha_mask, phase, morphogen)
            targets.append(target)
            reseed = reseed + plant_f * (target - reseed)
            reseed = reseed + noise_sigma * torch.randn(B, N_CELLS, device=dev, dtype=dt)
            if drift_t0 is not None and drift_bias is not None and t == drift_t0:
                # One-time mid-loop disturbance (matches eval perturb): the state
                # is knocked off once and must re-converge to the goal.
                reseed = reseed + torch.cat(
                    [drift_bias, torch.zeros(B, N_CELLS - 3, device=dev, dtype=dt)], dim=-1)
            reseed = torch.cat([reseed[..., :6], reseed[..., 6:7].clamp(0.0, 1.0)],
                               dim=-1)
            morphogen = morphogen * self.morphogen_decay
        return torch.stack(targets, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
