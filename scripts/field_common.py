"""Shared SmolVLA field-conditioned policy builder for E0/E1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "lerobot/smolvla_base"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK = ("Pick up each object and place it on the matching colored target, "
        "going around the walls.\n")
STATE_DIM = 7
SUBGOAL_DIM = 4
ACTION_DIM = 7
FIELD_STATE_DIM = STATE_DIM + SUBGOAL_DIM
CHUNK = 8
N_ACTION_STEPS = 4
NUM_DENOISE = 10


def tokenize_task(tok, text, max_len=48):
    enc = tok(text, add_special_tokens=True, padding="max_length",
              max_length=max_len, truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def build_field_policy(device="cuda", arm="gt-plan", task=TASK):
    """Returns (policy, field_wrapper_or_None, lang_ids, lang_mask).

    arm='no-field' builds no encoder; every other arm shares the same encoder
    architecture and token count.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoTokenizer
    from soma.field_policy import FieldConditionedPolicy, FieldEncoder

    policy = SmolVLAPolicy.from_pretrained(REPO)
    cfg = policy.config
    cfg.chunk_size = CHUNK
    cfg.n_action_steps = N_ACTION_STEPS
    cfg.num_steps = NUM_DENOISE
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE,
                                           shape=(FIELD_STATE_DIM,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL,
                                                    shape=(3, 256, 256)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    policy.eval()

    hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
    if arm == "no-field":
        wrapper = FieldConditionedPolicy(policy, None)
    else:
        wrapper = FieldConditionedPolicy(
            policy, FieldEncoder(out_dim=hidden).to(device))
    wrapper = wrapper.to(device)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    lang_ids, lang_mask = tokenize_task(tok, task)
    return wrapper, lang_ids.to(device), lang_mask.to(device).bool()


def load_norm(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n = ck["norm"]
    out = {}
    for k in ("ch_mean", "ch_std", "st_mean", "st_std"):
        v = n[k]
        out[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return out


def load_field_ckpt(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def save_field_ckpt(path, wrapper, extra):
    policy = wrapper.policy
    sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()
          if "vlm_with_expert.vlm" not in k}
    enc = None if wrapper.field_encoder is None else \
        {k: v.detach().cpu() for k, v in wrapper.field_encoder.state_dict().items()}
    payload = {"state": sd, "field": enc, **extra}
    torch.save(payload, path)
