"""Voxel spacing resampling. The single largest source of scanner confound."""

from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk


def _validate_spacing(name: str, spacing: tuple[float, float, float]) -> None:
    if len(spacing) != 3:
        raise ValueError(f"{name} spacing must be length 3, got {len(spacing)}")
    if not all(math.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"{name} spacing must be positive and finite, got {spacing}")


def resample_to_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
) -> np.ndarray:
    """Resample a 3D array to a target voxel spacing.

    ``volume`` is a (z, y, x) array. ``source_spacing`` and ``target_spacing``
    are in (x, y, z) order, the SimpleITK convention, and are used as given:
    no reordering happens inside this function.

    Returns a (z', y', x') float32 array.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {volume.shape}")
    _validate_spacing("source", source_spacing)
    _validate_spacing("target", target_spacing)

    arr = np.asarray(volume, dtype=np.float32)

    # A (z, y, x) array becomes an image whose size is (x, y, z).
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in source_spacing))

    # Physical extent per axis divided by the new spacing; all in (x, y, z).
    new_size = [
        max(1, int(round(n * src / tgt)))
        for n, src, tgt in zip(img.GetSize(), source_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)

    out = resampler.Execute(img)
    return sitk.GetArrayFromImage(out).astype(np.float32)