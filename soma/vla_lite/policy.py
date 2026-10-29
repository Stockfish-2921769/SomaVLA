"""End-to-end low-level VLA-lite policy.

`field_source` selects the controlled field arm while everything else
(adapter, obstacle head, readout, decoder, data, labels) stays identical.
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


class VLALiteLowLevel(nn.Module):
    def __init__(self, vision=None, field_source="nca", grid=32,
                 vis_dim=768, vision_ch=64, task_dim=8, n_tasks=1,
                 d=256, chunk=8, action_dim=7):
        super().__init__()
        arm = _ARM_ALIASES.get(field_source.lower(), field_source.lower())
        if arm not in {"nca", "gt", "euclid", "cnn", "vis",
                       "none", "zero", "random"}:
            raise ValueError(f"unknown field_source {field_source!r}")
        self.field_source = arm
        self.grid = grid
        self.vision = vision
        self.adapter = SpatialAdapter(vis_dim, d, vision_ch)
        self.nca = NCA(channels=32, vision_ch=vision_ch, task_dim=task_dim)
        self.cnn = CNNField(ext_dim=self.nca.ext_dim) if arm == "cnn" else None
        self.euclid = EuclidField() if arm == "euclid" else None
        # equal-bandwidth perception control: raw adapter features through the
        # same readout, no iterative field computation
        self.vis_field = nn.Conv2d(vision_ch, 32, 1) if arm == "vis" else None
        self.readout = FieldReadout(in_ch=32, d=d, out_grid=16)
        self.decoder = FlowActionDecoder(d=d, chunk=chunk,
                                         action_dim=action_dim)
        self.language = LanguageSlot(d=d, task_dim=task_dim, n_tasks=n_tasks)
        self.register_buffer("st_mean", torch.zeros(7))
        self.register_buffer("st_std", torch.ones(7))
        self.register_buffer("ch_mean", torch.zeros(4))
        self.register_buffer("ch_std", torch.ones(4))

    def set_norm(self, st_mean, st_std, ch_mean, ch_std):
        self.st_mean.copy_(torch.as_tensor(st_mean, dtype=torch.float32))
        self.st_std.copy_(torch.as_tensor(st_std, dtype=torch.float32))
        self.ch_mean.copy_(torch.as_tensor(ch_mean, dtype=torch.float32))
        self.ch_std.copy_(torch.as_tensor(ch_std, dtype=torch.float32))

    def _grid_tokens(self, field):
        if self.field_source in {"none", "zero"}:
            b = field.shape[0]
            return field.new_zeros(b, self.decoder.grid_tokens,
                                   self.decoder.state_proj[0].out_features)
        return self.readout(field)

    def forward(self, images=None, vis_tokens=None, states=None,
                subgoals=None, task_ids=None, gt_plan=None, gt_obstacle=None,
                actions_norm=None, noise=None, time=None, sample_steps=10,
                grid=None, field_dropout=0.0):
        grid = self.grid if grid is None else grid
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
            state, plan, nca_delta = self.nca(ext, return_delta=True)
            # "zero" pays the NCA compute but hides the field from the decoder
            field = state
        elif self.field_source == "gt":
            if gt_plan is None or gt_obstacle is None:
                raise ValueError("gt arm needs gt_plan and gt_obstacle")
            plan = gt_plan.unsqueeze(1).to(feat.dtype)
            field = feat.new_zeros(b, 32, grid, grid)
            field[:, :1] = plan
            p_obs = gt_obstacle.unsqueeze(1).to(feat.dtype)
        else:
            if self.field_source == "cnn":
                ext = self.nca.make_ext(feat, p_obs, goal_xy, start_xy, task8)
                plan = self.cnn(ext)
            elif self.field_source == "euclid":
                plan = self.euclid(goal_xy, grid)
            elif self.field_source == "random":
                plan = random_smooth_field(b, grid, feat.device, feat.dtype)
            else:                                   # vis / none / zero
                plan = feat.new_zeros(b, 1, grid, grid)
            if self.field_source == "vis":
                field = self.vis_field(feat)
            elif self.field_source not in {"none", "zero"}:
                field = feat.new_zeros(b, 32, grid, grid)
                field[:, :1] = plan

        tokens = self._grid_tokens(field if field is not None
                                   else feat.new_zeros(b, 32, grid, grid))
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
