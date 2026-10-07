"""Tests for spacing resampling."""

import numpy as np
import pytest

from imagex.preprocessing.resample import resample_to_spacing


def test_identity_resample_preserves_shape() -> None:
    vol = np.random.rand(8, 16, 16).astype(np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.shape == vol.shape


def test_upsample_doubles_xy() -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 0.5, 0.5))
    assert out.shape[1] == 32
    assert out.shape[2] == 32
    assert out.shape[0] == 8


def test_downsample_halves_xy() -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 2.0, 2.0))
    assert out.shape[1] == 8
    assert out.shape[2] == 8


def test_dtype_is_float32() -> None:
    vol = np.ones((4, 4, 4), dtype=np.int16)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.dtype == np.float32


def test_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        resample_to_spacing(np.zeros((4, 4), dtype=np.float32), (1, 1, 1), (1, 1, 1))


def test_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError):
        resample_to_spacing(np.zeros((4, 4, 4), dtype=np.float32), (1, 1, 1), (1, 0, 1))
