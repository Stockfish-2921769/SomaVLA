"""Visit-level high-level policy.

Queried once per visit at hover; predicts [gx, gy, gz, open] in normalized
visit-command space.  Optionally reads a pooled NCA/GT field.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelPolicy(nn.Module):
    def __init__(self, vis_dim=768, d=256, layers=2, heads=8,
                 field_ch=32, field_pool=8, use_field=False):
        super().__init__()
        self.use_field = use_field
        self.field_pool = field_pool
        self.vis_proj = nn.Linear(vis_dim, d)
        self.state_proj = nn.Sequential(
            nn.Linear(7, d), nn.SiLU(), nn.Linear(d, d))
        self.lang_proj = nn.Linear(d, d)
        self.field_proj = nn.Conv2d(field_ch, d, 1)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d, 4)

    def forward(self, vis_tokens, state_norm, lang_token, field=None):
        b = vis_tokens.shape[0]
        tok = [self.vis_proj(vis_tokens),
               self.state_proj(state_norm).unsqueeze(1),
               self.lang_proj(lang_token).unsqueeze(1)]
        if self.use_field and field is not None:
            f = self.field_proj(field)
            f = F.adaptive_avg_pool2d(f, (self.field_pool, self.field_pool))
            tok.append(f.flatten(2).transpose(1, 2))
        # query is last, so field tokens can be inserted before it
        tok.append(self.query.expand(b, 1, -1))
        h = self.tr(torch.cat(tok, dim=1))
        return self.out(h[:, -1])
