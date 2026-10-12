"""Image registration: align a prior CT volume to a current one.

Registration is the first stage of the temporal engine. It answers:
where in the current volume does each point in the prior volume map?

Two registration modes are supported:

- rigid: rotation and translation only. Fast, robust, appropriate when
  the patient is positioned similarly between studies.
- affine: adds scaling and shear. More flexible, still linear. Appropriate
  when positioning differs, but not when anatomy has deformed.

Deformable registration is out of scope for V1. It is a later extension
if rigid and affine prove insufficient on NLST data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class RegistrationResult:
    """Result of registering a moving volume to a fixed volume."""
    transform: sitk.Transform
    final_metric_value: float
    iterations: int
    converged: bool
    transform_type: str


def _to_sitk(volume: np.ndarray, spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    """Wrap a (z, y, x) array as a SimpleITK image with xyz spacing."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume.shape}")
    img = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    return img


def register_rigid(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    number_of_iterations: int = 200,
    learning_rate: float = 1.0,
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
) -> RegistrationResult:
    """Rigid registration of moving to fixed. Returns the transform.

    fixed, moving: (z, y, x) float arrays of the same anatomical region.
    spacing_xyz: voxel spacing shared by both volumes.

    The returned transform maps points in the fixed image's physical space
    to the moving image's physical space, per SimpleITK convention.
    """
    if fixed.shape != moving.shape:
        raise ValueError(
            f"fixed {fixed.shape} and moving {moving.shape} must have the same shape"
        )
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError("shrink_factors and smoothing_sigmas must have the same length")

    fixed_img = _to_sitk(fixed, spacing_xyz)
    moving_img = _to_sitk(moving, spacing_xyz)

    initial = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=number_of_iterations,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(list(shrink_factors))
    registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial, inPlace=False)

    final_transform = registration.Execute(fixed_img, moving_img)
    final_metric = float(registration.GetMetricValue())
    iterations = int(registration.GetOptimizerIteration())
    converged = registration.GetOptimizerStopConditionDescription() != ""

    return RegistrationResult(
        transform=final_transform,
        final_metric_value=final_metric,
        iterations=iterations,
        converged=converged,
        transform_type="rigid",
    )


def register_affine(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    number_of_iterations: int = 200,
    learning_rate: float = 1.0,
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
) -> RegistrationResult:
    """Affine registration of moving to fixed. Adds scale and shear to rigid."""
    if fixed.shape != moving.shape:
        raise ValueError(
            f"fixed {fixed.shape} and moving {moving.shape} must have the same shape"
        )
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError("shrink_factors and smoothing_sigmas must have the same length")

    fixed_img = _to_sitk(fixed, spacing_xyz)
    moving_img = _to_sitk(moving, spacing_xyz)

    initial = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=number_of_iterations,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(list(shrink_factors))
    registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial, inPlace=False)

    final_transform = registration.Execute(fixed_img, moving_img)
    final_metric = float(registration.GetMetricValue())
    iterations = int(registration.GetOptimizerIteration())
    converged = registration.GetOptimizerStopConditionDescription() != ""

    return RegistrationResult(
        transform=final_transform,
        final_metric_value=final_metric,
        iterations=iterations,
        converged=converged,
        transform_type="affine",
    )


def resample_to_fixed(
    moving: np.ndarray,
    fixed: np.ndarray,
    transform: sitk.Transform,
    spacing_xyz: tuple[float, float, float],
    default_value: float = 0.0,
) -> np.ndarray:
    """Apply a transform to resample moving into the fixed image's frame."""
    fixed_img = _to_sitk(fixed, spacing_xyz)
    moving_img = _to_sitk(moving, spacing_xyz)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)

    out = resampler.Execute(moving_img)
    return sitk.GetArrayFromImage(out).astype(np.float32)
