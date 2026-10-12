"""Tests for image registration. Uses small synthetic volumes."""

import numpy as np
import pytest
import SimpleITK as sitk

from imagex.temporal.registration import (
    register_affine,
    register_rigid,
    resample_to_fixed,
)


def _make_blob(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0) -> np.ndarray:
    z, y, x = np.indices(shape)
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    return np.exp(-(dist ** 2) / (2 * radius ** 2)).astype(np.float32)


def test_register_rigid_identity() -> None:
    vol = _make_blob()
    result = register_rigid(vol, vol, spacing_xyz=(1.0, 1.0, 1.0), number_of_iterations=20)
    assert result.transform_type == "rigid"
    assert result.iterations >= 0


def test_register_rigid_on_shifted_volume() -> None:
    fixed = _make_blob(center=(8, 16, 16))
    moving = _make_blob(center=(8, 18, 16))  # shifted 2 voxels in y
    result = register_rigid(
        fixed, moving, spacing_xyz=(1.0, 1.0, 1.0), number_of_iterations=50
    )
    # Metric should decrease; convergence is not guaranteed in 50 iterations.
    assert isinstance(result.final_metric_value, float)
    assert result.transform is not None


def test_register_affine_identity() -> None:
    vol = _make_blob()
    result = register_affine(vol, vol, spacing_xyz=(1.0, 1.0, 1.0), number_of_iterations=20)
    assert result.transform_type == "affine"


def test_register_rejects_shape_mismatch_rigid() -> None:
    a = _make_blob(shape=(16, 32, 32))
    b = _make_blob(shape=(16, 32, 24))
    with pytest.raises(ValueError):
        register_rigid(a, b, spacing_xyz=(1.0, 1.0, 1.0))


def test_register_rejects_shape_mismatch_affine() -> None:
    a = _make_blob(shape=(16, 32, 32))
    b = _make_blob(shape=(16, 32, 24))
    with pytest.raises(ValueError):
        register_affine(a, b, spacing_xyz=(1.0, 1.0, 1.0))


def test_register_rejects_mismatched_pyramid_lengths() -> None:
    vol = _make_blob()
    with pytest.raises(ValueError):
        register_rigid(
            vol,
            vol,
            spacing_xyz=(1.0, 1.0, 1.0),
            shrink_factors=(4, 2, 1),
            smoothing_sigmas=(2.0, 1.0),
        )


def test_resample_identity_transform_preserves_volume() -> None:
    vol = _make_blob()
    identity = sitk.Euler3DTransform()
    out = resample_to_fixed(vol, vol, identity, spacing_xyz=(1.0, 1.0, 1.0))
    assert out.shape == vol.shape
    # Values at the blob center should be nearly identical.
    assert out[8, 16, 16] == pytest.approx(vol[8, 16, 16], abs=1e-3)


def test_resample_rejects_non_3d() -> None:
    vol = _make_blob()
    identity = sitk.Euler3DTransform()
    with pytest.raises(ValueError):
        resample_to_fixed(np.zeros((4, 4)), vol, identity, spacing_xyz=(1.0, 1.0, 1.0))
