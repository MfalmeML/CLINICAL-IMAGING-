"""Tests for resample_to_spacing.

Volumes are (z, y, x). Spacing tuples are (x, y, z), the SimpleITK convention.
"""

import numpy as np
import pytest

from imagex.preprocessing.resample import resample_to_spacing


def test_identity_resample_preserves_shape() -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.shape == (8, 16, 16)


def test_upsample_doubles_xy() -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (0.5, 0.5, 1.0))  # x, y, z
    assert out.shape == (8, 32, 32)


def test_downsample_halves_xy() -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (2.0, 2.0, 1.0))  # x, y, z
    assert out.shape == (8, 8, 8)


def test_each_axis_uses_its_own_target_spacing() -> None:
    vol = np.zeros((8, 16, 24), dtype=np.float32)  # z, y, x
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 0.5, 2.0))  # x, y, z
    assert out.shape == (4, 32, 24)


def test_anisotropic_source_spacing_is_read_as_xyz() -> None:
    vol = np.zeros((4, 8, 16), dtype=np.float32)  # z, y, x
    out = resample_to_spacing(vol, (1.0, 1.0, 2.0), (1.0, 1.0, 1.0))  # x, y, z
    assert out.shape == (8, 8, 16)


def test_dtype_is_float32() -> None:
    vol = np.zeros((4, 8, 8), dtype=np.int16)
    out = resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.dtype == np.float32


def test_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        resample_to_spacing(np.zeros((16, 16)), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_rejects_non_positive_target() -> None:
    vol = np.zeros((4, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        resample_to_spacing(vol, (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))


def test_rejects_non_positive_source() -> None:
    vol = np.zeros((4, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        resample_to_spacing(vol, (1.0, 0.0, 1.0), (1.0, 1.0, 1.0))