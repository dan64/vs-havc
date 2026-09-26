# Native PyTorch implementation of the DINOv3 ViT-B/16 backbone used by
# Segmentor_DINOv3 (see resnet.py), frozen inference only.
#
# This file is a derivative work: it re-implements, for the inference path,
# the DINOv3ViT model of HuggingFace transformers 4.57.6
# (transformers/models/dinov3_vit/modeling_dinov3_vit.py, Copyright 2025
# Meta AI and The HuggingFace Inc. team, Apache License 2.0:
# http://www.apache.org/licenses/LICENSE-2.0), removing every dependency on
# transformers / huggingface_hub / tokenizers.
#
# Parameter names, module structure and forward math are kept identical to
# the HuggingFace implementation, so that:
#   - the fused CMNET2 checkpoints (keys 'key_encoder.network2.backbone.*')
#     and the local weights/dinov3-vitb16/model.safetensors file load with
#     strict=True;
#   - the extracted features are numerically identical to the ones produced
#     by transformers (same weights, same inputs).
#
# Deliberately NOT implemented (never used by the filter, which runs
# inference only): training-time coordinate augmentation of the rope
# embeddings, drop-path (identity in eval anyway), gated MLP, mask-token
# handling (bool_masked_pos is a pretraining-only feature, the mask token is
# kept only as a parameter for checkpoint compatibility), gradient
# checkpointing, attention backends other than torch's
# scaled_dot_product_attention (which is what the transformers
# implementation used at runtime here: attn_implementation="sdpa").

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_patches_center_coordinates(num_patches_h, num_patches_w, dtype=torch.float32, device=None):
    """2D centers of the image patches, normalized to [-1, +1] (same as transformers)."""
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Rotary position embedding applied to the patch tokens only, ignoring
    the prefix tokens (cls + register tokens) - same as transformers."""
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)
    return q, k


class DINOv3ViTEmbeddings(nn.Module):
    """CLS token + register tokens + patch embeddings (conv patchify).
    Token sequence layout: [cls, register_0..N, patch_0..M]."""

    def __init__(self, hidden_size, num_channels, patch_size, num_register_tokens):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # mask_token is a pretraining-only feature (bool_masked_pos): kept as
        # a parameter only so that checkpoints containing it load strictly.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        # (batch, channels, height, width) -> (batch, num_patches, hidden_size)
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)
        return embeddings


class DINOv3ViTRopePositionEmbedding(nn.Module):
    """2D rotary position embeddings for the patch tokens (cos/sin pair of
    tensors shaped (num_patches, head_dim))."""

    def __init__(self, head_dim, base, patch_size):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.patch_size = patch_size
        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # not part of state_dict (as in transformers)
        self._coords_cache = {}  # plain python cache (never in state_dict)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # force float32, as in transformers
            key = (num_patches_h, num_patches_w, str(device))
            patch_coords = self._coords_cache.get(key)
            if patch_coords is None:
                if len(self._coords_cache) > 64:  # bound the cache across many input resolutions
                    self._coords_cache.clear()
                patch_coords = get_patches_center_coordinates(
                    num_patches_h, num_patches_w, dtype=torch.float32, device=device)
                self._coords_cache[key] = patch_coords

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DINOv3ViTAttention(nn.Module):
    """Multi-head self-attention with 2D rope on the patch tokens."""

    def __init__(self, hidden_size, num_heads, q_bias, k_bias, v_bias, o_bias):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False  # read by the sdpa attention path (kept for reference)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=v_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=q_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=o_bias)

    def forward(self, hidden_states, position_embeddings):
        batch_size, num_tokens, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Same call the transformers sdpa path made at runtime (no mask, no
        # causal masking, dropout 0.0: the backbone always runs in eval mode).
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, num_tokens, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        # exact Gaussian error linear unit (same as transformers ACT2FN["gelu"])
        return self.down_proj(F.gelu(self.up_proj(x)))


class DINOv3ViTLayer(nn.Module):
    """Transformer block: attention and MLP with layer-scale and residuals
    (drop-path rate is 0.0 for this checkpoint: identity in eval, omitted)."""

    def __init__(self, hidden_size, num_heads, intermediate_size, layer_norm_eps,
                 q_bias, k_bias, v_bias, o_bias, mlp_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = DINOv3ViTAttention(hidden_size, num_heads, q_bias, k_bias, v_bias, o_bias)
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size, mlp_bias)
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size)

    def forward(self, hidden_states, position_embeddings):
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


@dataclass
class DINOv3ViTOutput:
    """Same fields the transformers model returned (BaseModelOutputWithPooling)."""
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    # 13 entries: [0] = embeddings output, [i] = output of layer i-1
    # (identical layout to the transformers implementation).
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class DINOv3ViT(nn.Module):
    """DINOv3 ViT backbone, frozen inference. Build it with
    :meth:`from_pretrained_dir`, which reads config.json + model.safetensors
    from a local directory (no network access, no HuggingFace cache)."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        intermediate_size = config["intermediate_size"]
        layer_norm_eps = config["layer_norm_eps"]
        self.embeddings = DINOv3ViTEmbeddings(
            hidden_size, config["num_channels"], config["patch_size"], config["num_register_tokens"])
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(
            head_dim=hidden_size // num_heads, base=config["rope_theta"], patch_size=config["patch_size"])
        self.layer = nn.ModuleList([
            DINOv3ViTLayer(hidden_size, num_heads, intermediate_size, layer_norm_eps,
                           config["query_bias"], config["key_bias"], config["value_bias"],
                           config["proj_bias"], config["mlp_bias"])
            for _ in range(config["num_hidden_layers"])
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values, output_hidden_states=True):
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)

        hidden_states = self.embeddings(pixel_values)
        all_hidden_states = [hidden_states] if output_hidden_states else None
        position_embeddings = self.rope_embeddings(pixel_values)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_embeddings)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return DINOv3ViTOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    @classmethod
    def from_pretrained_dir(cls, weights_dir, map_location="cpu"):
        """Load the backbone from a self-contained local directory
        (config.json + model.safetensors)."""
        config_path = os.path.join(weights_dir, "config.json")
        weights_path = os.path.join(weights_dir, "model.safetensors")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"DINOv3 backbone config not found: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DINOv3 backbone weights not found: {weights_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if config.get("model_type") != "dinov3_vit":
            raise ValueError(f"unexpected model_type in {config_path}: {config.get('model_type')!r}")
        if config.get("use_gated_mlp", False):
            raise NotImplementedError("gated-MLP backbone variants are not supported by this implementation")
        if config.get("hidden_act", "gelu") != "gelu":
            raise NotImplementedError(f"unsupported hidden_act: {config.get('hidden_act')!r}")

        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load the DINOv3 backbone (model.safetensors): "
                "pip install safetensors") from exc

        model = cls(config)
        state_dict = load_file(weights_path, device=map_location)
        model.load_state_dict(state_dict, strict=True)
        return model
