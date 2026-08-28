"""State-only MoE router: scene + EEF state -> skill + boundary conditions.

First stage of the MoE router (gate d) uses ground-truth scene params, no
images. `perceive()` is the boundary a lightweight VLM image encoder will
later replace with the same contract (scene[4] = [obj_x, obj_y, place_x,
place_y] in meters).

Boundary-condition contract to the NCA experts (see soma/skill_experts.py):
  skill      top-1 of the 6 primitives
  goal       absolute target pose [7]
  alpha      activity mask via SKILL_REGISTRY[skill] lookup
  duration   expected skill length, clamped to the skill's duration range
"""

import numpy as np
import torch
import torch.nn as nn

from soma.skill_experts import SKILLS, SKILL_REGISTRY, TRANSITIONS


class StateRouter(nn.Module):
    def __init__(self, scene_dim=4, state_dim=7, n_skills=6, hidden=128):
        super().__init__()
        self.n_skills = n_skills
        self.backbone = nn.Sequential(
            nn.Linear(scene_dim + state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.skill_head = nn.Linear(hidden, n_skills)
        # Goal assembly: per-skill learned constant base + per-skill scene→xy
        # readout. Only px/py are ever scene-derived (obj for approach, place
        # for transport); every other ACTIVE goal DOF is a per-skill constant
        # (heights, openness, zero-rot). Keeping constants as pure learned
        # scalars (no scene input) removes the (w,b) degeneracy that made full
        # linear goal heads overfit constant targets to input-dependent
        # features (poor held-out generalization).
        self.goal_base = nn.Parameter(torch.zeros(n_skills, 7))
        self.xy_heads = nn.ModuleList([nn.Linear(scene_dim, 2) for _ in range(n_skills)])
        self.duration_head = nn.Linear(hidden, 1)

    def forward(self, scene, state):
        """scene [B,4], state [B,7].

        Returns (skill_logits, goals_all, duration):
          goals_all  [B, n_skills, 7] — each skill's assembled goal. HARD
                     routing picks goals_all[arange, argmax(logits)]; each
                     skill's xy head + base rows are supervised only on its own
                     skill's samples (see train_moe_router.py).
        """
        B = scene.shape[0]
        x = torch.cat([scene, state], dim=-1)
        emb = self.backbone(x)
        logits = self.skill_head(emb)
        xys = torch.stack([h(scene) for h in self.xy_heads], dim=1)         # [B,6,2]
        goals_all = self.goal_base.unsqueeze(0).expand(B, -1, -1).clone()
        goals_all[:, :, 0:2] = xys
        duration = self.duration_head(emb).squeeze(-1)                      # [B]
        return logits, goals_all, duration

    @torch.no_grad()
    def boundary(self, scene, state, prev_skill=None, completed=None):
        """Runtime boundary conditions for the routed skill (single sample).

        scene[4] / state[7] as np arrays. Routes HARD (top-1 skill) and emits
        that skill's own goal head. Returns a dict with the routed skill,
        absolute goal, alpha mask (registry lookup) and duration.
        prev_skill: if given, the router's candidates are restricted to the
        task-grammar TRANSITIONS[prev_skill]; argmax within that set.
        Resolves the stateless router's grasp/release ambiguity when the
        place target sits next to the grasp point.
        completed: whether the previous skill reached its goal. On completion
        only the canonical next skill is legal (forced transition — at its
        own completion pose the router's self-logit dominates, so allowing
        self would re-pick the finished skill and stall the chain); on
        timeout (completed=False) self stays legal for retry.
        """
        dev = next(self.parameters()).device
        scene_t = torch.as_tensor(np.asarray(scene, dtype=np.float32), device=dev).unsqueeze(0)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=dev).unsqueeze(0)
        logits, goals_all, duration = self.forward(scene_t, state_t)
        if prev_skill is not None:
            nexts = TRANSITIONS[prev_skill]
            valid = [SKILLS.index(nexts[-1])] if completed else \
                    [SKILLS.index(s) for s in nexts]
            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid] = logits[0, valid]
            logits = mask
        skill_idx = int(logits.argmax(-1).item())
        skill = SKILLS[skill_idx]
        lo, hi = SKILL_REGISTRY[skill]["duration"]
        return {
            "skill_idx": skill_idx,
            "skill": skill,
            "goal": goals_all[0, skill_idx].cpu().numpy(),
            "alpha_mask": SKILL_REGISTRY[skill]["alpha_mask"],
            "duration": float(np.clip(duration.item(), lo, hi)),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
