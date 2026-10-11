"""Tests for Baseline 4 (3D ViT). Shape only, no training."""

import pytest
import torch

from imagex.perception.baseline4_3d_vit import Baseline4_3DViT, PatchEmbed3D


def test_patch_embed_token_count() -> None:
    pe = PatchEmbed3D(in_channels=1, embed_dim=16, patch_size=(4, 8, 8))
    x = torch.randn(1, 1, 8, 16, 16)
    out = pe(x)
    # 8/4 * 16/8 * 16/8 = 2 * 2 * 2 = 8 tokens
    assert out.shape == (1, 8, 16)


def test_vit_output_shape() -> None:
    model = Baseline4_3DViT(embed_dim=32, depth=2, num_heads=4, patch_size=(4, 8, 8))
    x = torch.randn(2, 1, 8, 16, 16)
    y = model(x)
    assert y.shape == (2, 2)


def test_vit_accepts_different_depths() -> None:
    model = Baseline4_3DViT(
        embed_dim=32, depth=2, num_heads=4, patch_size=(2, 8, 8)
    )
    x = torch.randn(1, 1, 8, 16, 16)
    y = model(x)
    assert y.shape == (1, 2)


def test_vit_rejects_4d_input() -> None:
    model = Baseline4_3DViT(embed_dim=32, depth=2, num_heads=4)
    x = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError):
        model(x)


def test_vit_rejects_indivisible_embed_dim() -> None:
    with pytest.raises(ValueError):
        Baseline4_3DViT(embed_dim=30, num_heads=7)


def test_vit_rejects_oversized_volume() -> None:
    model = Baseline4_3DViT(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=(2, 8, 8),
        max_tokens=4,
    )
    x = torch.randn(1, 1, 8, 16, 16)  # 4 * 2 * 2 = 16 tokens > 4
    with pytest.raises(ValueError):
        model(x)


def test_vit_parameter_count_is_small() -> None:
    model = Baseline4_3DViT(embed_dim=32, depth=2, num_heads=4)
    n = sum(p.numel() for p in model.parameters())
    # Keep this small on purpose for a baseline that should be beatable.
    assert n < 2_000_000
