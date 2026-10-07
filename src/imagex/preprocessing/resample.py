"""Voxel spacing resampling. The single largest source of scanner confound."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def resample_to_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
) -> np.ndarray:
    """Resample a 3D array to a target voxel spacing.

    Input volume is (z, y, x). source_spacing and target_spacing are (x, y, z)
    to match SimpleITK convention; they are reversed internally.

    Returns a (z', y', x') float32 array.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {volume.shape}")
    if len(source_spacing) != 3 or len(target_spacing) != 3:
        raise ValueError("spacing must be length 3")
    if any(s <= 0 for s in target_spacing):
        raise ValueError("target spacing must be positive")

    arr = np.asarray(volume, dtype=np.float32)

    # SimpleITK expects (x, y, z); we hold (z, y, x).
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in reversed(source_spacing)))

    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()
    new_size = [
        max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i])))
        for i in range(3)
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
