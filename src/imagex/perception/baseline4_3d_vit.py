"""Baseline 4: 3D Vision Transformer over volumetric patches.

Input is the full preprocessed volume. The volume is divided into
non-overlapping 3D patches, each patch is linearly embedded, a class
token is prepended, and a standard transformer encoder runs over the
sequence. Classification reads from the class token.

Deliberately minimal. No learned positional embedding interpolation,
no register tokens, no multi-scale. If this does not beat Baseline 3,
the additional complexity is not earning its place in the ladder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patch embedding via a strided convolution."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, Z, Y, X) -> (N, embed_dim, Dz, Dy, Dx)
        x = self.proj(x)
        # -> (N, embed_dim, Dz*Dy*Dx) -> (N, tokens, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class Baseline4_3DViT(nn.Module):
    """Small 3D ViT classifier.

    Input:  (N, 1, Z, Y, X)
    Output: (N, 2) logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        patch_size: tuple[int, int, int] = (4, 8, 8),
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5D input (N, C, Z, Y, X), got {x.shape}")

        tokens = self.patch_embed(x)
        n, t, _ = tokens.shape
        if t + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"volume produces {t} tokens, exceeds configured max "
                f"{self.pos_embed.size(1) - 1}"
            )

        cls = self.cls_token.expand(n, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : t + 1, :]
        tokens = self.pos_drop(tokens)

        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)
