"""Tests for Baseline 3 architecture. Shape only, no training."""

import pytest
import torch

from imagex.perception.baseline3_3d import Baseline3_3DCNN


def test_output_shape_small_volume() -> None:
    model = Baseline3_3DCNN()
    x = torch.randn(2, 1, 16, 32, 32)
    y = model(x)
    assert y.shape == (2, 2)


def test_output_shape_odd_dimensions() -> None:
    model = Baseline3_3DCNN()
    x = torch.randn(1, 1, 9, 17, 21)
    y = model(x)
    assert y.shape == (1, 2)


def test_accepts_larger_volumes() -> None:
    model = Baseline3_3DCNN()
    x = torch.randn(1, 1, 64, 128, 128)
    y = model(x)
    assert y.shape == (1, 2)


def test_rejects_4d_input() -> None:
    model = Baseline3_3DCNN()
    x = torch.randn(1, 1, 64, 64)
    with pytest.raises(ValueError):
        model(x)


def test_rejects_6d_input() -> None:
    model = Baseline3_3DCNN()
    x = torch.randn(1, 1, 4, 4, 4, 4)
    with pytest.raises(ValueError):
        model(x)


def test_parameter_count_is_reasonable() -> None:
    model = Baseline3_3DCNN()
    n = sum(p.numel() for p in model.parameters())
    # 3D convs are expensive relative to 2D; bound is looser but still small.
    assert n < 5_000_000
