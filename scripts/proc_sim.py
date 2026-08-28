"""Procedural skill trajectory generator — the NCA training corpus.

For each of the 6 primitives, samples a plausible start pose S and an
absolute target pose B (per the skill's semantics relative to an object and
a place location), then produces a smooth T-step state trajectory from S→B.
Only alpha-active DOFs move; inactive DOFs hold (their B_i = S_i). Small
measurement noise is added so the receding-horizon reseed never equals the
clean path exactly.

The generator is online: sample_skill() draws fresh episodes on demand, so
training never overfits a static dataset.

State layout: [pos(3) m, rot(3) axis-angle rad, openness(1) ∈ [0,1]].
"""

import numpy as np

from soma.skill_experts import SKILL_REGISTRY, SKILLS

# Workspace / placement constants (arbitrary but plausible LIBERO-like meters).
CONTACT_Z = 0.52     # object on the table
GRASP_Z = 0.58       # pregrasp height above the object
LIFT_Z = 0.66        # lifted clearance
POS_NOISE = (0.002, 0.002, 0.002)   # meters
ROT_NOISE = (0.01, 0.01, 0.01)      # radians
OPEN_NOISE = 0.01


def _smooth(x):
    """3x² − 2x³ smoothstep, monotonic 0→1 with smooth accel/decel."""
    return x * x * (3.0 - 2.0 * x)


def _rot_target(rng):
    """Small random axis-angle orientation (upright-ish gripper)."""
    return rng.uniform(-0.3, 0.3, 3).astype(np.float32)


def _sample_scene(rng: np.random.RandomState):
    """Random pick-and-place scene. Returns (obj_xy, place_xy) in meters."""
    obj = np.array([rng.uniform(0.36, 0.48), rng.uniform(0.40, 0.52)], dtype=np.float32)
    place = np.array([rng.uniform(0.46, 0.58), rng.uniform(0.34, 0.46)], dtype=np.float32)
    return obj, place


def sample_skill(skill: str, rng: np.random.RandomState, T=None, scene=None):
    """Draw one episode. Returns (states[T,7], goal[7], alpha_mask[7], T).

    T is drawn from the skill's duration range unless given. scene is a
    (obj_xy, place_xy) pair; None draws a fresh scene per call. Start and
    target poses are derived from the scene anchors per the skill's semantics
    (heights stay at the fixed CONTACT/GRASP/LIFT constants), so the corpus is
    scene-parameterized — the router gets a learnable scene → goal signal and
    the NCA experts train over varied object/place positions.
    """
    cfg = SKILL_REGISTRY[skill]
    lo, hi = cfg["duration"]
    if T is None:
        T = int(rng.randint(lo, hi + 1))
    mask = cfg["alpha_mask"].copy()
    rot = _rot_target(rng)
    obj, place = scene if scene is not None else _sample_scene(rng)

    # Skill-specific start → target pose semantics relative to the scene.
    if skill == "approach":
        S_pos = np.array([obj[0] + rng.uniform(-0.08, 0.08),
                          obj[1] + rng.uniform(-0.08, 0.08),
                          rng.uniform(0.72, 0.78)], dtype=np.float32)
        B_pos = np.array([obj[0], obj[1], GRASP_Z], dtype=np.float32)
        S_open, B_open = 1.0, 1.0
        S_rot, B_rot = rot.copy(), rot.copy()
    elif skill == "grasp":
        S_pos = np.array([obj[0], obj[1], GRASP_Z + 0.02], dtype=np.float32)
        B_pos = np.array([obj[0], obj[1], CONTACT_Z], dtype=np.float32)
        S_open, B_open = 1.0, 0.0
        S_rot, B_rot = rot.copy(), rot.copy()
    elif skill == "lift":
        S_pos = np.array([obj[0], obj[1], CONTACT_Z + 0.01], dtype=np.float32)
        B_pos = np.array([obj[0], obj[1], LIFT_Z], dtype=np.float32)
        S_open, B_open = 0.0, 0.0
        S_rot, B_rot = rot.copy(), rot.copy()
    elif skill == "transport":
        S_pos = np.array([obj[0], obj[1], LIFT_Z], dtype=np.float32)
        B_pos = np.array([place[0], place[1], LIFT_Z], dtype=np.float32)
        S_open, B_open = 0.0, 0.0
        S_rot, B_rot = rot.copy(), rot.copy()
    elif skill == "place":
        S_pos = np.array([place[0], place[1], LIFT_Z], dtype=np.float32)
        B_pos = np.array([place[0], place[1], CONTACT_Z], dtype=np.float32)
        S_open, B_open = 0.0, 0.0
        S_rot, B_rot = rot.copy(), rot.copy()
    else:  # release — pose holds, gripper opens
        S_pos = np.array([place[0], place[1], CONTACT_Z + 0.01], dtype=np.float32)
        B_pos = S_pos.copy()
        S_open, B_open = 0.0, 1.0
        S_rot, B_rot = rot.copy(), rot.copy()

    # Smooth trajectory: active DOFs interpolate S→B, inactive hold.
    prof = _smooth(np.linspace(0.0, 1.0, T))
    pos = S_pos[None, :] + (B_pos - S_pos)[None, :] * prof[:, None]
    rotp = S_rot[None, :] + (B_rot - S_rot)[None, :] * prof[:, None]
    openp = S_open + (B_open - S_open) * prof
    states = np.concatenate([pos, rotp, openp[:, None]], axis=-1).astype(np.float32)

    # Measurement noise (so reseed never equals the clean path).
    states += rng.normal(0, POS_NOISE + ROT_NOISE + (OPEN_NOISE,), size=states.shape).astype(np.float32)

    goal = np.concatenate([B_pos, B_rot, [B_open]]).astype(np.float32)
    return states, goal, mask, T


def make_batch(skill: str, T: int, batch: int, rng: np.random.RandomState):
    """Vectorized batch builder: all samples share duration T."""
    states, goals, masks = [], [], []
    for _ in range(batch):
        s, g, m, _ = sample_skill(skill, rng, T=T)
        states.append(s)
        goals.append(g)
        masks.append(m)
    return (np.stack(states).astype(np.float32),
            np.stack(goals).astype(np.float32),
            np.stack(masks).astype(np.float32))


def router_sample(rng: np.random.RandomState, skill=None):
    """One MoE-router training sample at a skill boundary.

    Returns (scene[4], start_state[7], skill_idx, goal[7], mask[7], T):
      scene       [obj_x, obj_y, place_x, place_y] ground-truth scene (the
                  future VLM perceive() contract replaces this).
      start_state the (noisy) EEF pose the router observes at invocation.
      goal        the skill's absolute target B (boundary condition).
    """
    scene = _sample_scene(rng)
    if skill is None:
        skill = SKILLS[rng.randint(len(SKILLS))]
    states, goal, mask, T = sample_skill(skill, rng, scene=scene)
    scene_vec = np.concatenate([scene[0], scene[1]]).astype(np.float32)
    return scene_vec, states[0].copy(), SKILLS.index(skill), goal, mask, T
