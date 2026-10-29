"""End-to-end low-level VLA-lite policy.

`field_source` selects the controlled field arm while everything else
(adapter, obstacle head, readout, decoder, data, labels) stays identical.

`readout_mode` controls field bandwidth into the decoder:
  full  learned generators (nca/cnn) expose their whole 32-ch state; scalar
        sources (gt/euclid/random) fill channel 0 only; `vis` is the
        equal-bandwidth no-iteration perception control.
  plan  every arm exposes channel 0 (the plan scalar) only, so learned and
        scalar field sources are compared at identical bandwidth.
"""
import torch
import torch.nn as nn

from .adapter import SpatialAdapter
from .baselines import CNNField, EuclidField, random_smooth_field
from .decoder import FieldReadout, FlowActionDecoder
from .language import LanguageSlot
from .nca import NCA


_ARM_ALIASES = {
    "nca": "nca", "gt": "gt", "gtfield": "gt", "gt-field": "gt",
    "euclid": "euclid", "cnn": "cnn", "vis": "vis", "vision": "vis",
    "feat": "vis", "none": "none", "nofield": "none", "no-field": "none",
    "zero": "zero", "zerofield": "zero", "zero-field": "zero",
    "random": "random", "randomfield": "random", "random-field": "random",
}

READOUT_MODES = ("full", "plan")
# Inference-time probes (task book 7.3): hide the whole field, or replace
# only its plan channel while keeping any latent channels.
FIELD_OVERRIDES = ("zero", "plan-zero", "plan-gt", "plan-random")


class VLALiteLowLevel(nn.Module):
    def __init__(self, vision=None, field_source="nca", grid=32,
                 vis_dim=768, vision_ch=64, task_dim=8, n_tasks=1,
                 d=256, chunk=8, action_dim=7, readout_mode="full",
                 cnn_layers=12):
        super().__init__()
        arm = _ARM_ALIASES.get(field_source.lower(), field_source.lower())
        if arm not in {"nca", "gt", "euclid", "cnn", "vis",
                       "none", "zero", "random"}:
            raise ValueError(f"unknown field_source {field_source!r}")
        if readout_mode not in READOUT_MODES:
            raise ValueError(f"unknown readout_mode {readout_mode!r}")
        self.field_source = arm
        self.readout_mode = readout_mode
        self.grid = grid
        self.vision = vision
        self.adapter = SpatialAdapter(vis_dim, d, vision_ch)
        self.nca = NCA(channels=32, vision_ch=vision_ch, task_dim=task_dim)
        self.readout = FieldReadout(in_ch=32, d=d, out_grid=16)
        self.decoder = FlowActionDecoder(d=d, chunk=chunk,
                                         action_dim=action_dim)
        self.language = LanguageSlot(d=d, task_dim=task_dim, n_tasks=n_tasks)
        # Built after the shared stack: an arm-specific parameterized module
        # here would otherwise consume RNG and shift readout/decoder init,
        # confounding cross-arm comparisons for a fixed seed.
        self.cnn = (CNNField(ext_dim=self.nca.ext_dim, layers=cnn_layers,
                             channels=self.nca.channels)
                    if arm == "cnn" else None)
        self.euclid = EuclidField() if arm == "euclid" else None
        # equal-bandwidth perception control: raw adapter features through the
        # same readout, no iterative field computation
        self.vis_field = nn.Conv2d(vision_ch, 32, 1) if arm == "vis" else None
        self.register_buffer("st_mean", torch.zeros(7))
        self.register_buffer("st_std", torch.ones(7))
        self.register_buffer("ch_mean", torch.zeros(4))
        self.register_buffer("ch_std", torch.ones(4))

    def set_norm(self, st_mean, st_std, ch_mean, ch_std):
        self.st_mean.copy_(torch.as_tensor(st_mean, dtype=torch.float32))
        self.st_std.copy_(torch.as_tensor(st_std, dtype=torch.float32))
        self.ch_mean.copy_(torch.as_tensor(ch_mean, dtype=torch.float32))
        self.ch_std.copy_(torch.as_tensor(ch_std, dtype=torch.float32))

    def _zero_tokens(self, b, like):
        return like.new_zeros(b, self.decoder.grid_tokens,
                              self.decoder.state_proj[0].out_features)

    def _apply_override(self, field, override, gt_plan, grid):
        """Replace the plan channel of `field` (plan-* probes)."""
        if field is None:
            raise ValueError(f"override {override!r} needs a field; arm "
                             f"{self.field_source!r} has none")
        field = field.clone()
        b = field.shape[0]
        if override == "plan-zero":
            field[:, 0] = 0.0
        elif override == "plan-gt":
            if gt_plan is None:
                raise ValueError("plan-gt override needs gt_plan")
            field[:, 0] = gt_plan.to(field.dtype)
        elif override == "plan-random":
            field[:, :1] = random_smooth_field(b, grid, field.device,
                                               field.dtype)
        return field

    def forward(self, images=None, vis_tokens=None, states=None,
                subgoals=None, task_ids=None, gt_plan=None, gt_obstacle=None,
                actions_norm=None, noise=None, time=None, sample_steps=10,
                grid=None, field_dropout=0.0, k_steps=None,
                field_override=None):
        grid = self.grid if grid is None else grid
        if field_override is not None and \
                field_override not in FIELD_OVERRIDES:
            raise ValueError(f"unknown field_override {field_override!r}")
        if vis_tokens is None:
            if self.vision is None:
                raise ValueError("images given but policy.vision is None")
            vis_tokens = self.vision(images)
        b = vis_tokens.shape[0]
        feat, obs_logits = self.adapter(vis_tokens, size=grid)
        p_obs = torch.sigmoid(obs_logits)
        goal_xy = subgoals[:, :2]
        start_xy = states[:, :2]
        task8, lang_token = self.language(task_ids, batch_size=b,
                                          device=states.device)

        field = None
        nca_delta = None
        if self.field_source in {"nca", "zero"}:
            ext = self.nca.make_ext(feat, p_obs, goal_xy, start_xy, task8)
            state, plan, nca_delta = self.nca(ext, k_steps=k_steps,
                                              return_delta=True)
            # "zero" pays the NCA compute but hides the field from the decoder
            field = state if self.field_source == "nca" else None
        elif self.field_source == "cnn":
            ext = self.nca.make_ext(feat, p_obs, goal_xy, start_xy, task8)
            field = self.cnn(ext)
            plan = field[:, 0]
        elif self.field_source == "gt":
            if gt_plan is None or gt_obstacle is None:
                raise ValueError("gt arm needs gt_plan and gt_obstacle")
            plan = gt_plan.unsqueeze(1).to(feat.dtype)
            field = feat.new_zeros(b, 32, grid, grid)
            field[:, :1] = plan
            p_obs = gt_obstacle.unsqueeze(1).to(feat.dtype)
        else:
            if self.field_source == "euclid":
                plan = self.euclid(goal_xy, grid)
            elif self.field_source == "random":
                plan = random_smooth_field(b, grid, feat.device, feat.dtype)
            else:                                   # vis / none
                plan = feat.new_zeros(b, 1, grid, grid)
            if self.field_source == "vis":
                field = self.vis_field(feat)
            elif self.field_source != "none":
                field = feat.new_zeros(b, 32, grid, grid)
                field[:, :1] = plan

        if field is not None and self.readout_mode == "plan":
            field = torch.cat([field[:, :1],
                               field.new_zeros(b, field.shape[1] - 1,
                                               grid, grid)], dim=1)
        if field_override not in (None, "zero"):
            field = self._apply_override(field, field_override, gt_plan, grid)

        if field is None or field_override == "zero":
            tokens = self._zero_tokens(b, feat)
        else:
            tokens = self.readout(field)
        if self.training and field_dropout > 0:
            keep = (torch.rand(b, 1, 1, device=tokens.device) >=
                    field_dropout).to(tokens.dtype)
            tokens = tokens * keep
        state_norm = (states - self.st_mean) / self.st_std
        subgoal_norm = (subgoals - self.ch_mean) / self.ch_std

        out = dict(plan=plan, obstacle_logits=obs_logits,
                   obstacle_prob=p_obs, field_tokens=tokens, field=field,
                   nca_delta=nca_delta, feat=feat)
        if actions_norm is not None:
            loss = self.decoder.flow_loss(
                actions_norm, tokens, state_norm, subgoal_norm, lang_token,
                noise=noise, time=time)
            out["loss"] = loss
        else:
            out["action"] = self.decoder.sample(
                tokens, state_norm, subgoal_norm, lang_token,
                steps=sample_steps, noise=noise)
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
