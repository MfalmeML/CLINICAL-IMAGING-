"""Tests for Baseline 2 architecture. Shape only, no training."""

import pytest
import torch

from imagex.perception.baseline2_2p5d import Baseline2_2p5DCNN


def test_output_shape_default_three_slices() -> None:
    model = Baseline2_2p5DCNN(n_slices=3)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 2)


def test_output_shape_five_slices() -> None:
    model = Baseline2_2p5DCNN(n_slices=5)
    x = torch.randn(1, 5, 128, 128)
    y = model(x)
    assert y.shape == (1, 2)


def test_accepts_larger_slices() -> None:
    model = Baseline2_2p5DCNN(n_slices=3)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    assert y.shape == (1, 2)


def test_rejects_non_4d_input() -> None:
    model = Baseline2_2p5DCNN(n_slices=3)
    x = torch.randn(1, 3, 64, 64, 4)
    with pytest.raises(ValueError):
        model(x)


def test_rejects_wrong_channel_count() -> None:
    model = Baseline2_2p5DCNN(n_slices=3)
    x = torch.randn(1, 4, 64, 64)
    with pytest.raises(ValueError):
        model(x)


def test_rejects_bad_n_slices() -> None:
    with pytest.raises(ValueError):
        Baseline2_2p5DCNN(n_slices=0)


def test_parameter_count_is_small() -> None:
    model = Baseline2_2p5DCNN(n_slices=3)
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000
