"""Frozen SigLIP-B/16 vision front-end.

Uses the locally cached open_clip ViT-B-16-SigLIP-256 (webli) weights.
Input 256x256 -> 16x16x768 patch tokens (256 tokens).  The text tower is not
kept; VLA-lite has no LLM.
"""
import math

import torch
import torch.nn as nn


class FrozenSigLIP(nn.Module):
    def __init__(self, model_name="ViT-B-16-SigLIP-256", pretrained="webli",
                 device="cpu"):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.visual = model.visual
        del model
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()
        self.feat_dim = 768
        self.grid = 16
        self.n_tokens = self.grid * self.grid

    @torch.no_grad()
    def forward(self, images):
        """images [B,3,H,W] uint8 (0..255) or float (0..1) -> [B,N,768]."""
        if images.dtype == torch.uint8:
            x = images.float() / 127.5 - 1.0
        else:
            x = images.float() * 2.0 - 1.0
        trunk = getattr(self.visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward_features"):
            tokens = trunk.forward_features(x)
        elif hasattr(self.visual, "forward_features"):
            tokens = self.visual.forward_features(x)
        else:
            tokens = self.visual(x)
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        return tokens

    def train(self, mode=True):
        # keep the frozen tower in eval mode even if the parent is trained
        super().train(False)
        self.visual.eval()
        return self
