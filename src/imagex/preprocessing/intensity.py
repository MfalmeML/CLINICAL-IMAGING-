"""Hounsfield unit conversion and intensity normalization."""

import numpy as np


def to_hounsfield(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert raw DICOM pixel values to Hounsfield units.

    HU = pixel * slope + intercept. Returns float32.
    """
    arr = np.asarray(pixel_array, dtype=np.float32)
    return arr * float(slope) + float(intercept)


def clip_hu(hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU values to a fixed window. Raises if min >= max."""
    if hu_min >= hu_max:
        raise ValueError(f"hu_min ({hu_min}) must be < hu_max ({hu_max})")
    return np.clip(hu, hu_min, hu_max)


def normalize_minmax(clipped: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Map clipped HU to [0, 1]. Raises if window is degenerate."""
    if hu_max <= hu_min:
        raise ValueError("degenerate window")
    return (clipped - hu_min) / (hu_max - hu_min)
