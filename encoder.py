"""Encoder-only Michelangelo, extracted from the vendored sal_perceiver.py."""
import math
from typing import Optional

import torch
from torch import nn
from einops import repeat

from .checkpoint import checkpoint
from .embedder import FourierEmbedder
from .transformer_blocks import ResidualCrossAttentionBlock, Transformer


class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        """Build learned shape queries, point projection, and attention blocks."""
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """Cross-attend learned queries to point features and refine the shape tokens."""

        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """Run point-to-shape attention with optional activation checkpointing."""

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class AlignedShapeEncoder(nn.Module):
    """Encode xyz/normal clouds into one global and 256 local shape tokens."""
    def __init__(self, *, num_latents=256, width=768, heads=12, layers=8,
                 num_freqs=8, point_feats=3, init_scale=.25, device=None, dtype=None):
        """Build only the frozen encoder branch of the released shape model."""
        super().__init__()
        embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=False)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype, num_latents=num_latents + 1,
            fourier_embedder=embedder, point_feats=point_feats, width=width,
            heads=heads, layers=layers, init_scale=init_scale * math.sqrt(1.0 / width),
            qkv_bias=False, flash=False, use_ln_post=True, use_checkpoint=False)

    def forward(self, cloud):
        """Return shape tokens from a batch of normalized xyz/normal clouds."""
        features, _ = self.encoder(cloud[..., :3], cloud[..., 3:])
        return features
