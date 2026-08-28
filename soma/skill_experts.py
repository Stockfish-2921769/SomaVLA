"""Skill registry — the MoE→NCA boundary-condition contract.

For each primitive this defines:
  * alpha_mask — which of the 7 body-graph cells are active. An inactive
    cell's value is frozen at its reseed: that DOF holds position/openness.
    This is the Smart-Bricks alpha gating — the cell neither updates nor is
    driven toward a goal, and the loss is masked to active cells only.
  * openness_goal — gripper target (1 = open, 0 = closed).
  * duration — expected skill length in env steps (random-rollout training
    prior; the MoE emits a concrete duration at runtime).
  * Target pose semantics are defined in scripts/proc_sim.py per skill.

Cell order: [px, py, pz, rx, ry, rz, g].
"""

import numpy as np

# cell indices
PX, PY, PZ, RX, RY, RZ, G = range(7)

SKILLS = ["approach", "grasp", "lift", "transport", "place", "release"]

# Task grammar: the legal next skills for each current skill. The state-only
# router is stateless — at a grasp-completion pose that coincides with a
# pre-release pose (object placed next to where it was picked up) it cannot
# tell "just grasped" from "about to release" and may skip the carry. The
# controller constrains the router's candidates to this chain (self allowed
# for timeout retry); the router still picks the argmax within the set.
TRANSITIONS = {
    "approach": ("approach", "grasp"),
    "grasp": ("grasp", "lift"),
    "lift": ("lift", "transport"),
    "transport": ("transport", "place"),
    "place": ("place", "release"),
    "release": ("release",),
}

SKILL_REGISTRY = {
    # Move to the pregrasp pose above the object. All pose DOFs drive;
    # the gripper is held open (g inactive → frozen at reseed openness).
    "approach": {
        "alpha_mask": np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32),
        "openness_goal": 1.0,
        "duration": (70, 120),
    },
    # Sink pz onto the object and close the gripper. xy/orientation hold.
    "grasp": {
        "alpha_mask": np.array([0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        "openness_goal": 0.0,
        "duration": (25, 50),
    },
    # Raise the grasped object straight up. xy/orientation hold.
    "lift": {
        "alpha_mask": np.array([0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        "openness_goal": 0.0,
        "duration": (25, 50),
    },
    # Move to the place position at lift height, holding the grasp.
    "transport": {
        "alpha_mask": np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        "openness_goal": 0.0,
        "duration": (70, 120),
    },
    # Lower onto the place target. xy/orientation hold, grasp held.
    "place": {
        "alpha_mask": np.array([0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        "openness_goal": 0.0,
        "duration": (25, 50),
    },
    # Open the gripper; pose holds exactly.
    "release": {
        "alpha_mask": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        "openness_goal": 1.0,
        "duration": (15, 30),
    },
}


def get_mask(skill: str) -> np.ndarray:
    return SKILL_REGISTRY[skill]["alpha_mask"]


def get_duration_range(skill: str) -> tuple:
    return SKILL_REGISTRY[skill]["duration"]


def get_openness_goal(skill: str) -> float:
    return SKILL_REGISTRY[skill]["openness_goal"]
