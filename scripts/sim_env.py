"""Phase 5b + Phase 6 — self-contained closed-loop pick-and-place environment.

The controller's absolute target_pose is integrated by a damped plant: each
step the state moves a fraction `plant_f` of the commanded delta plus small
measurement noise, so the receding-horizon NCA expert is exercised under a
lagging actuator. Phase 6 replaces the deterministic grasp model with an
analytic Coulomb one (soma/physics.py): the object attaches only when the
gripper's normal force F_n = F_max·(1−g) can support its weight (F_n ≥ m·g0/μ)
at contact depth, and drops if the tangential load (gravity + commanded-step
inertia) exceeds the friction capacity μ·F_n for long enough. `--physics off`
falls back to the old attach model (regression path).

The env also publishes a per-step physics_ctx (the modality channel the
controller feeds the NCA): [μ, m, F_n, slip_risk] + one-hot contact mode.

State layout matches proc_sim: [pos(3) m, axis-angle(3) rad, openness(1) ∈
[0,1]] (1 = open). All parameters are fresh — nothing imported from the old
controller.
"""

import numpy as np

from soma.physics import (FREE, CONTACT, GRASPED, SLIP, DROPPED, CONTACT_NAMES,
                          F_MAX, G0, TAU, SLIP_THRESH, EPS, PHYSICS_CTX_DIM,
                          numpy_slip_metrics)

CONTACT_Z = 0.52
GRASP_Z = 0.58
LIFT_Z = 0.66


class SimEnv:
    def __init__(self, rng=None, plant_f=0.5, contact_r=0.02, place_tol=0.02,
                 pos_noise=0.001, rot_noise=0.01, open_noise=0.01,
                 physics=True, mu=0.4, mass=0.1):
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.plant_f = plant_f
        self.contact_r = contact_r
        self.place_tol = place_tol
        self.noise = np.array([pos_noise] * 3 + [rot_noise] * 3 + [open_noise],
                              dtype=np.float32)
        self.physics_on = physics
        self.mu = mu
        self.mass = mass
        self.state = None
        self.obj_xy = None
        self.place_xy = None
        self.attached = False
        self.contact_mode = FREE
        self.slip_disp = 0.0
        self.dropped = False
        self.obj_final = None
        self.ctx = None

    def reset(self, scene, physics=None):
        """scene = (obj_xy, place_xy). Starts EEF near obj, high up, open.
        physics = optional (mu, mass); defaults to the ctor values."""
        obj, place = scene
        pos = np.array([obj[0] + self.rng.uniform(-0.08, 0.08),
                        obj[1] + self.rng.uniform(-0.08, 0.08),
                        self.rng.uniform(0.72, 0.78)], dtype=np.float32)
        rot = self.rng.uniform(-0.3, 0.3, 3).astype(np.float32)
        self.state = np.concatenate([pos, rot, [1.0]]).astype(np.float32)
        self.obj_xy = np.asarray(obj, dtype=np.float32).copy()
        self.place_xy = np.asarray(place, dtype=np.float32).copy()
        if physics is not None:
            self.mu, self.mass = float(physics[0]), float(physics[1])
        self.attached = False
        self.contact_mode = FREE
        self.slip_disp = 0.0
        self.dropped = False
        self.obj_final = None
        self.ctx = self._make_ctx(0.0)
        return self.state.copy()

    def physics_ctx(self):
        """The [P] modality vector for the controller's next relax call."""
        return self.ctx.copy()

    def _make_ctx(self, slip_risk):
        """[P] numpy ctx from the current contact mode + a slip_risk estimate."""
        F_n = F_MAX * (1.0 - float(self.state[6]))
        risk = min(float(slip_risk), 5.0)
        onehot = np.zeros(5, dtype=np.float32)
        onehot[self.contact_mode] = 1.0
        return np.concatenate([
            [self.mu / 0.6, self.mass / 0.35, F_n / F_MAX, risk / 5.0], onehot
        ]).astype(np.float32)

    def _near_obj(self, s):
        return (np.linalg.norm(s[:2] - self.obj_xy) < self.contact_r
                and s[2] < GRASP_Z + 0.03)

    def _detach(self, s):
        """Object loses gripper support. Whether this is a clean placement or a
        failed drop is decided by WHERE it happens: the object simply stops
        being carried and its final position — the EEF's at detach time —
        determines success (within place_tol of the target = placed). A hard
        `dropped` flag that ignores position would mis-score a gripper opening
        over the place target as a drop, because the F_n < held_req window is
        crossed before g reaches the 0.5 release threshold on hard cells."""
        self.attached = False
        self.obj_final = s[:2].copy()
        self.dropped = bool(np.linalg.norm(s[:2] - self.place_xy) >= self.place_tol)
        return FREE

    def _coulomb_step(self, prev, target, s):
        """Advance the Coulomb contact state machine. Returns (obj_xy, final)."""
        g = float(s[6])
        F_n, F_t, risk = numpy_slip_metrics(s, target, self.mu, self.mass,
                                            held=(self.contact_mode == GRASPED))
        held_req = self.mass * G0 / self.mu        # F_n needed to carry the weight
        dx = float(np.linalg.norm(target[:3] - prev[:3]))
        mode = self.contact_mode

        if mode == FREE:
            if self._near_obj(s):
                mode = CONTACT
        elif mode == CONTACT:
            if F_n >= held_req and self._near_obj(s):
                mode = GRASPED
                self.attached = True
                self.obj_xy = s[:2].copy()
        elif mode == GRASPED:
            self.obj_xy = s[:2].copy()                      # carried
            if g >= 0.5:
                mode = self._detach(s)          # intentional release (open gripper)
            elif F_n < held_req:
                mode = self._detach(s)          # can't support the weight
            elif risk > 1.0:
                mode = SLIP
                self.slip_disp += (risk - 1.0) * dx
                if self.slip_disp > SLIP_THRESH:
                    mode = self._detach(s)      # slipped out mid-carry
        elif mode == SLIP:
            self.obj_xy = s[:2].copy()
            self.slip_disp += (risk - 1.0) * dx
            if g >= 0.5:
                mode = self._detach(s)
            elif self.slip_disp > SLIP_THRESH or F_n < held_req:
                mode = self._detach(s)
            elif risk <= 1.0:
                mode = GRASPED
        # FREE after detach; DROPPED mode no longer a terminal label — the
        # failure signal is self.dropped (detached away from the place target).

        self.contact_mode = mode
        return self.obj_xy.copy(), (None if self.obj_final is None
                                    else self.obj_final.copy())

    def _old_grasp_step(self, target, s):
        """Legacy deterministic attach (regression path, physics=False)."""
        if not self.attached:
            near = np.linalg.norm(s[:2] - self.obj_xy) < self.contact_r
            if s[6] < 0.5 and near:
                self.attached = True
        if self.attached:
            self.obj_xy = s[:2].copy()
            if s[6] >= 0.5:
                self.attached = False
                self.obj_final = self.obj_xy.copy()
        return self.obj_xy.copy(), (None if self.obj_final is None
                                    else self.obj_final.copy())

    def step(self, target_pose):
        """Integrate target_pose through the damped plant + grasp model.

        Returns (state[7], obj_xy, info{attached, contact, slip_risk,
        dropped, obj_final})."""
        target = np.asarray(target_pose, dtype=np.float32)
        prev = self.state.copy()
        self.state = self.state + self.plant_f * (target - self.state)
        self.state = self.state + self.noise * self.rng.randn(7).astype(np.float32)
        self.state[6] = float(np.clip(self.state[6], 0.0, 1.0))
        s = self.state

        if self.physics_on:
            obj_xy, final = self._coulomb_step(prev, target, s)
        else:
            obj_xy, final = self._old_grasp_step(target, s)
        if final is not None:
            self.obj_final = final

        # Publish the modality for the controller's NEXT step: contact mode +
        # the slip_risk the just-applied command produced.
        _, _, risk = numpy_slip_metrics(s, target, self.mu, self.mass,
                                        held=(self.contact_mode == GRASPED))
        self.ctx = self._make_ctx(risk)

        info = {"attached": self.attached,
                "contact": CONTACT_NAMES[self.contact_mode],
                "slip_risk": float(risk),
                "dropped": self.dropped,
                "obj_final": (None if self.obj_final is None
                              else self.obj_final.copy())}
        return s.copy(), obj_xy, info

    def success(self):
        """Object released within place_tol of the place location, never dropped."""
        if self.dropped or self.obj_final is None:
            return False
        return float(np.linalg.norm(self.obj_final - self.place_xy)) < self.place_tol

    def perturb(self, eps=0.01):
        """Mid-episode drift injection (robustness probe). Returns new state."""
        self.state[:3] += self.rng.uniform(-eps, eps, 3).astype(np.float32)
        return self.state.copy()
