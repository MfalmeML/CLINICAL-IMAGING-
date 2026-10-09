"""Tests for Baseline 1 architecture. Shape only, no training."""

import pytest
import torch

from imagex.perception.baseline1_2d import Baseline1_2DCNN


def test_output_shape() -> None:
    model = Baseline1_2DCNN()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 2)


def test_accepts_larger_slices() -> None:
    model = Baseline1_2DCNN()
    x = torch.randn(1, 1, 512, 512)
    y = model(x)
    assert y.shape == (1, 2)


def test_rejects_non_4d_input() -> None:
    model = Baseline1_2DCNN()
    x = torch.randn(1, 64, 64)
    with pytest.raises(ValueError):
        model(x)


def test_parameter_count_is_small() -> None:
    model = Baseline1_2DCNN()
    n = sum(p.numel() for p in model.parameters())
    # Sanity bound: baseline should be small, not a transformer.
    assert n < 1_000_000
