"""Phase 5b — self-contained closed-loop pick-and-place environment.

The controller's absolute target_pose is integrated by a damped plant: each
step the state moves a fraction `plant_f` of the commanded delta plus small
measurement noise, so the receding-horizon NCA expert is exercised under a
lagging actuator (a real closed-loop stability check — no instant pose
setting). A minimal grasp model attaches the object to the EEF when the
gripper closes near it, carries it through lift/transport, and leaves it at
the place location when the gripper opens on release.

State layout matches proc_sim: [pos(3) m, axis-angle(3) rad, openness(1) ∈
[0,1]] (1 = open). All parameters are fresh — nothing imported from the old
controller.
"""

import numpy as np

CONTACT_Z = 0.52
GRASP_Z = 0.58
LIFT_Z = 0.66


class SimEnv:
    def __init__(self, rng=None, plant_f=0.5, contact_r=0.02, place_tol=0.02,
                 pos_noise=0.001, rot_noise=0.01, open_noise=0.01):
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.plant_f = plant_f
        self.contact_r = contact_r
        self.place_tol = place_tol
        self.noise = np.array([pos_noise] * 3 + [rot_noise] * 3 + [open_noise],
                              dtype=np.float32)
        self.state = None
        self.obj_xy = None
        self.place_xy = None
        self.attached = False
        self.obj_final = None

    def reset(self, scene):
        """scene = (obj_xy, place_xy). Starts EEF near obj, high up, open."""
        obj, place = scene
        pos = np.array([obj[0] + self.rng.uniform(-0.08, 0.08),
                        obj[1] + self.rng.uniform(-0.08, 0.08),
                        self.rng.uniform(0.72, 0.78)], dtype=np.float32)
        rot = self.rng.uniform(-0.3, 0.3, 3).astype(np.float32)
        self.state = np.concatenate([pos, rot, [1.0]]).astype(np.float32)
        self.obj_xy = np.asarray(obj, dtype=np.float32).copy()
        self.place_xy = np.asarray(place, dtype=np.float32).copy()
        self.attached = False
        self.obj_final = None
        return self.state.copy()

    def step(self, target_pose):
        """Integrate target_pose through the damped plant + grasp model.

        Returns (state[7], obj_xy, info{attached, obj_final})."""
        target = np.asarray(target_pose, dtype=np.float32)
        self.state = self.state + self.plant_f * (target - self.state)
        self.state = self.state + self.noise * self.rng.randn(7).astype(np.float32)
        self.state[6] = float(np.clip(self.state[6], 0.0, 1.0))

        if not self.attached:
            near = np.linalg.norm(self.state[:2] - self.obj_xy) < self.contact_r
            if self.state[6] < 0.5 and near:
                self.attached = True
        if self.attached:
            self.obj_xy = self.state[:2].copy()
            if self.state[6] >= 0.5:
                self.attached = False
                self.obj_final = self.obj_xy.copy()

        info = {"attached": self.attached,
                "obj_final": None if self.obj_final is None else self.obj_final.copy()}
        return self.state.copy(), self.obj_xy.copy(), info

    def success(self):
        """Object released within place_tol of the place location."""
        if self.obj_final is None:
            return False
        return float(np.linalg.norm(self.obj_final - self.place_xy)) < self.place_tol

    def perturb(self, eps=0.01):
        """Mid-episode drift injection (robustness probe). Returns new state."""
        self.state[:3] += self.rng.uniform(-eps, eps, 3).astype(np.float32)
        return self.state.copy()
