"""Phase 5a — event-driven closed-loop controller over MoE router + NCA experts.

Each env step: if the current skill's active-DOF pose is within tolerance
(goal reached) or it has run past its duration, re-invoke the router
(event-driven transition); otherwise run the routed skill's NCA expert from
the current state (receding-horizon reseed) and return its absolute target
pose. The expert IS the low-level controller — there are no hand-tuned
feedback gains in this loop.

Completion is judged against the router's goal masked by the skill's alpha
mask, in physical units (pos m / rot rad / openness). A per-skill route log
is kept for eval reporting. The terminal skill (release) is never routed
away from; is_task_done() signals the end of a pick-and-place episode.
"""

import numpy as np
import torch

# completion tolerances: pos (m), rot (rad), openness
DEFAULT_TOL = (5e-3, 0.05, 0.05)


class BodyGraphController:
    def __init__(self, experts, router, device="cuda", completion_tol=DEFAULT_TOL):
        self.experts = {k: m.to(device).eval() for k, m in experts.items()}
        self.router = router.to(device).eval()
        self.device = device
        self.pos_tol, self.rot_tol, self.open_tol = completion_tol
        self.scene = None
        self.state = None
        self.boundary = None
        self.skill = None
        self.step_in_skill = 0
        self.morphogen = None
        self.route_log = []

    # ── routing ─────────────────────────────────────────────────
    def _route(self, completed=None):
        if self.boundary is not None:
            p, r, o = self._active_err()
            self.route_log.append({
                "skill": self.skill, "steps": self.step_in_skill,
                "completed": bool(p < self.pos_tol and r < self.rot_tol
                                  and o < self.open_tol)})
        # Constrain candidates to the task grammar so the stateless router
        # can't skip the carry when the place target sits near the grasp
        # point (its grasp/release ambiguity). On completion only the
        # canonical next skill is legal (forced transition — the router's
        # self-logit dominates at its own completion pose); on timeout self
        # stays legal for retry. Argmax within set.
        self.boundary = self.router.boundary(self.scene, self.state,
                                             prev_skill=self.skill,
                                             completed=completed)
        # Every skill demo holds orientation (proc_sim S_rot == B_rot), so the
        # correct goal rot is the CURRENT rot. The router's learned rot
        # constant (≈0, mean of the random start rotation) is OOD for the
        # experts — they were trained with goal_rot == reseed_rot (hold) and a
        # nonzero goal_norm rot freezes the relaxation. Enforce hold here.
        self.boundary["goal"] = np.asarray(self.boundary["goal"],
                                           dtype=np.float32).copy()
        self.boundary["goal"][3:6] = self.state[3:6]
        self.skill = self.boundary["skill"]
        self.step_in_skill = 0
        self.morphogen = None

    def reset(self, scene, state):
        self.scene = np.asarray(scene, dtype=np.float32)
        self.state = np.asarray(state, dtype=np.float32)
        self.route_log = []
        self._route()

    # ── completion ──────────────────────────────────────────────
    def _active_err(self):
        d = np.abs(self.state - self.boundary["goal"])
        m = self.boundary["alpha_mask"]
        pos = d[:3][m[:3] > 0]
        rot = d[3:6][m[3:6] > 0]
        opn = d[6:] if m[6] > 0 else np.array([])
        # MAX per group, not mean: a skill only "completes" when every active
        # DOF is inside tolerance. Mean hides a single-axis miss (transport
        # completing 11mm off in y while px/pz sit on goal), which compounds
        # with perception error into misplaced objects.
        return (pos.max() if pos.size else 0.0,
                rot.max() if rot.size else 0.0,
                opn.max() if opn.size else 0.0)

    def skill_complete(self):
        p, r, o = self._active_err()
        return bool(p < self.pos_tol and r < self.rot_tol and o < self.open_tol)

    def is_task_done(self):
        return self.skill == "release" and self.skill_complete()

    # ── control step ────────────────────────────────────────────
    def step(self, state=None, physics_ctx=None):
        """One closed-loop control step → (target_pose np[7], info dict).

        state: latest env observation (None → reuse internal state).
        physics_ctx: [P] modality vector from the env (Coulomb physics); the
        expert uses it to shape its command (Phase 6). None → zeros for
        physics-capable experts, ignored for Phase 5 experts."""
        if state is not None:
            self.state = np.asarray(state, dtype=np.float32)
        info = {"skill": self.skill, "step_in_skill": self.step_in_skill,
                "routed": False, "task_done": False}
        if self.boundary is not None and self.step_in_skill > 0:
            if self.skill == "release" and self.skill_complete():
                info["task_done"] = True
                return self.state.copy(), info
            if self.step_in_skill >= int(self.boundary["duration"]) or self.skill_complete():
                self._route(completed=self.skill_complete())
                info["routed"] = True

        u = (self.step_in_skill + 1) / self.boundary["duration"]
        phase = np.array([np.cos(2 * np.pi * u), np.sin(2 * np.pi * u),
                          1.0 / self.boundary["duration"]], dtype=np.float32)
        dev = self.device
        st = torch.as_tensor(self.state, dtype=torch.float32, device=dev).unsqueeze(0)
        gl = torch.as_tensor(self.boundary["goal"], dtype=torch.float32, device=dev).unsqueeze(0)
        mk = torch.as_tensor(self.boundary["alpha_mask"], dtype=torch.float32, device=dev).unsqueeze(0)
        ph = torch.as_tensor(phase, dtype=torch.float32, device=dev).unsqueeze(0)
        ctx_t = (torch.as_tensor(physics_ctx, dtype=torch.float32, device=dev).unsqueeze(0)
                 if physics_ctx is not None else None)
        exp = self.experts[self.skill]
        with torch.no_grad():
            target, morphogen = exp.relax(st, gl, mk, ph, self.morphogen,
                                          physics_ctx=ctx_t)
            self.morphogen = morphogen * exp.morphogen_decay
        target_np = target[0].cpu().numpy()
        self.step_in_skill += 1
        info["target"] = target_np
        return target_np, info
