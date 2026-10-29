"""Action decoder: NCA field tokens -> flow-matching action chunk."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldReadout(nn.Module):
    def __init__(self, in_ch=32, d=256, out_grid=16):
        super().__init__()
        self.out_grid = out_grid
        self.c1 = nn.Conv2d(in_ch, d, 1)
        self.c2 = nn.Conv2d(d, d, 3, padding=1)
        self.norm = nn.GroupNorm(8, d)

    def forward(self, field):
        x = F.gelu(self.c1(field))
        x = F.interpolate(x, size=(self.out_grid, self.out_grid),
                          mode="bilinear", align_corners=False)
        x = F.gelu(self.norm(self.c2(x)))
        return x.flatten(2).transpose(1, 2)      # [B, out_grid^2, d]


class FlowActionDecoder(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, grid_tokens=256,
                 chunk=8, action_dim=7, grid_dim=256):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.grid_tokens = grid_tokens
        self.state_proj = nn.Sequential(
            nn.Linear(7 + 4, d), nn.SiLU(), nn.Linear(d, d))
        self.lang_proj = nn.Linear(d, d)
        self.time_proj = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.action_proj = nn.Linear(chunk * action_dim, d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d, chunk * action_dim)

    def _tokens(self, grid_tokens, state_norm, subgoal_norm, lang_token,
                x_t, t):
        b = grid_tokens.shape[0]
        st = self.state_proj(torch.cat([state_norm, subgoal_norm], dim=-1))
        lg = self.lang_proj(lang_token)
        tm = self.time_proj(t.view(b, 1))
        ac = self.action_proj(x_t.reshape(b, -1))
        return torch.stack([st, lg, tm, ac], dim=1)

    def forward(self, grid_tokens, state_norm, subgoal_norm, lang_token,
                x_t, t):
        b = grid_tokens.shape[0]
        ctx = self._tokens(grid_tokens, state_norm, subgoal_norm, lang_token,
                           x_t, t)
        q = self.query.expand(b, 1, -1)
        h = torch.cat([grid_tokens, ctx, q], dim=1)
        h = self.tr(h)
        return self.out(h[:, -1]).view(b, self.chunk, self.action_dim)

    def flow_loss(self, actions, grid_tokens, state_norm, subgoal_norm,
                  lang_token, noise=None, time=None):
        b = actions.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            beta = torch.distributions.Beta(
                torch.tensor(1.5, device=actions.device),
                torch.tensor(1.0, device=actions.device))
            time = beta.sample((b,)).to(actions.device)
        tv = time.view(b, 1, 1)
        x_t = tv * noise + (1 - tv) * actions
        target = noise - actions
        pred = self.forward(grid_tokens, state_norm, subgoal_norm,
                            lang_token, x_t, time)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, grid_tokens, state_norm, subgoal_norm, lang_token,
               steps=10, noise=None):
        b = grid_tokens.shape[0]
        if noise is None:
            noise = torch.randn(b, self.chunk, self.action_dim,
                                device=grid_tokens.device)
        x_t = noise
        t = 1.0
        dt = -1.0 / steps
        while t >= -dt / 2:
            tv = torch.full((b,), t, device=grid_tokens.device)
            v = self.forward(grid_tokens, state_norm, subgoal_norm,
                             lang_token, x_t, tv)
            x_t = x_t + dt * v
            t += dt
        return x_t
