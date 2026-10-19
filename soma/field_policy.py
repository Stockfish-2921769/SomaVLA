"""Field-token injection for SmolVLA's action expert (method 2 / E0-E1).

The plan field is encoded by a small CNN into 64 tokens which are inserted
into the prefix immediately before the state token:

    [image | language | field(64) | state] -> VLM/expert prefix

Field tokens carry pad_mask=1 and att_mask=0 (same block as the state token),
so they are visible to the state token and, through the KV cache, to every
denoising step.  Only the field encoder and the (already trainable) action
expert receive gradients; the VLM stays frozen.

The insertion is implemented by wrapping ``VLAFlowMatching.embed_prefix`` on
the policy instance instead of editing the installed lerobot package.
"""

import torch
import torch.nn as nn


class FieldEncoder(nn.Module):
    """[B,4,G,G] -> [B,64,H]: 4->32->64->128 (all stride 2), 8x8 tokens."""

    def __init__(self, in_ch=4, out_dim=960, widths=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1),
            nn.GroupNorm(4, c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.GroupNorm(8, c3), nn.GELU(),
        )
        self.proj = nn.Linear(c3, out_dim)

    def forward(self, field):
        x = field.float()
        x = self.net(x)                       # [B,128,8,8]
        x = x.flatten(2).transpose(1, 2)      # [B,64,128]
        return self.proj(x)


class FieldConditionedPolicy(nn.Module):
    """Policy wrapper owning the field encoder and the prefix splice."""

    def __init__(self, policy, field_encoder=None):
        super().__init__()
        self.policy = policy
        self.field_encoder = field_encoder
        self._field = None
        self._orig_embed_prefix = policy.model.embed_prefix
        policy.model.embed_prefix = self._embed_prefix

    @property
    def n_field_tokens(self):
        return 0 if self.field_encoder is None else 64

    def set_field(self, field):
        self._field = field

    def clear_field(self):
        self._field = None

    def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks,
                      state=None):
        embs, pad_masks, att_masks = self._orig_embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)

        if self._field is None or self.field_encoder is None:
            return embs, pad_masks, att_masks

        field = self._field
        if field.shape[0] != embs.shape[0]:
            raise ValueError(
                f"field batch {field.shape[0]} != prefix batch {embs.shape[0]}")
        tokens = self.field_encoder(field.to(embs.device)).to(embs.dtype)
        if tokens.shape[-1] != embs.shape[-1]:
            raise ValueError(
                f"field token dim {tokens.shape[-1]} != VLM hidden "
                f"{embs.shape[-1]}")

        b, n_tok = tokens.shape[:2]
        # state is the final token of the original prefix; insert before it.
        embs = torch.cat([embs[:, :-1], tokens, embs[:, -1:]], dim=1)
        pad_masks = torch.cat([
            pad_masks[:, :-1],
            torch.ones(b, n_tok, dtype=pad_masks.dtype,
                       device=pad_masks.device),
            pad_masks[:, -1:],
        ], dim=1)
        att_masks = torch.cat([
            att_masks[:, :-1],
            torch.zeros(b, n_tok, dtype=att_masks.dtype,
                        device=att_masks.device),
            att_masks[:, -1:],
        ], dim=1)
        return embs, pad_masks, att_masks


def attach_field_encoder(policy, vlm_hidden=None, device=None):
    """Build the encoder matching the policy's VLM width and wrap the policy."""
    if vlm_hidden is None:
        vlm_hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
    enc = FieldEncoder(out_dim=vlm_hidden)
    if device is not None:
        enc = enc.to(device)
    return FieldConditionedPolicy(policy, enc)
