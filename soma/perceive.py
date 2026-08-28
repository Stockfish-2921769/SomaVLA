"""Phase 4f-2 — lightweight SigLIP perception: image → scene vector.

Replaces the ground-truth scene at the `perceive()` boundary of the MoE
router (gate d stage 2). A frozen SigLIP vision encoder (ViT-B-16-SigLIP-256,
webli) is followed by a small regression head predicting the normalized
scene[4] = [obj_x, obj_y, place_x, place_y] (meters).

SigLIP weights are pulled on first load. huggingface.co is unreachable from
this host, so the HF mirror endpoint is set before the open_clip download.
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn

# Scene distribution (meters), matching proc_sim._sample_scene. Normalizing
# each dim to ~[-1,1] keeps the regression head well-conditioned.
SCENE_MEAN = np.array([0.42, 0.46, 0.52, 0.40], dtype=np.float32)
SCENE_SCALE = np.array([0.06, 0.06, 0.06, 0.06], dtype=np.float32)

MODEL_ARCH = "ViT-B-16-SigLIP-256"
PRETRAINED = "webli"


def normalize_scene(scene):
    return (np.asarray(scene, dtype=np.float32) - SCENE_MEAN) / SCENE_SCALE


def denormalize_scene(norm):
    return np.asarray(norm, dtype=np.float32) * SCENE_SCALE + SCENE_MEAN


def preprocess_batch(imgs, device):
    """np.uint8 [B,H,W,3] → torch [B,3,256,256] on device.

    Vectorized equivalent of open_clip's preprocess for already-256×256 RGB
    uint8 input: [0,255] → [-1,1] (SigLIP Normalize(0.5,0.5,0.5)). Skips the
    PIL resize because the renderer already emits 256×256.
    """
    x = torch.from_numpy(np.asarray(imgs, dtype=np.float32)).to(device)
    x = (x / 127.5) - 1.0
    return x.permute(0, 3, 1, 2).contiguous()


class SigLIPPerception(nn.Module):
    """Frozen SigLIP encoder + MLP → normalized scene[4].

    Constructing the model triggers the SigLIP weight download (HF mirror).
    """

    def __init__(self, hidden=128):
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_ARCH, pretrained=PRETRAINED)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.encoder = self.model.visual
        self.head = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )

    @torch.no_grad()
    def encode_image(self, img):
        """img [B,3,256,256] → SigLIP features [B,768] (frozen, no grad)."""
        return self.encoder(img)

    def forward(self, img):
        """img [B,3,256,256] → normalized scene pred [B,4]."""
        return self.head(self.encode_image(img))

    @torch.no_grad()
    def perceive(self, img_np):
        """Single RGB image np.uint8 [H,W,3] → scene[4] meters (the boundary)."""
        dev = next(self.parameters()).device
        pil = _np_to_pil(img_np)
        x = self.preprocess(pil).unsqueeze(0).to(dev)
        norm = self.forward(x)
        return denormalize_scene(norm[0].cpu().numpy())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _np_to_pil(arr):
    from PIL import Image
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
