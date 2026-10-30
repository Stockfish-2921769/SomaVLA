"""Seeding and deterministic-training switch for VLA-lite.

The pilot (reports/VLA_LITE_PILOT.md §6.2) found that a fixed-seed rerun
with identical init moved nca s0 from 33.0% to 42.5% closed-loop success:
nondeterministic CUDA kernels alone add ~±5–10 pp. Arm comparisons need
`enable_determinism()` (or several reruns per config).
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism():
    """Call before the first CUDA op. Any op without a deterministic kernel
    then raises RuntimeError instead of silently drifting."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # fused SDPA backward kernels are not guaranteed deterministic; the
    # decoders are small, so the math path costs little
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def loader_generator(seed):
    """Dedicated shuffle RNG: data order then depends only on the seed, not
    on how much global RNG the arm's module construction consumed."""
    return torch.Generator().manual_seed(seed)
