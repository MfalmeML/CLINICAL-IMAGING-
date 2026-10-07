"""Series selection and slice ordering for CT volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom


def select_ct_series(
    datasets: list[pydicom.Dataset],
    min_slices: int,
    max_slice_thickness_mm: float,
) -> list[pydicom.Dataset]:
    """Filter to CT slices meeting thickness and count requirements.

    Raises ValueError if no qualifying series is present.
    """
    if not datasets:
        raise ValueError("empty dataset list")

    ct = [d for d in datasets if getattr(d, "Modality", None) == "CT"]
    if not ct:
        raise ValueError("no CT slices found")

    def thickness(d: pydicom.Dataset) -> float:
        v = getattr(d, "SliceThickness", None)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    qualifying = [d for d in ct if thickness(d) <= max_slice_thickness_mm]
    if len(qualifying) < min_slices:
        raise ValueError(
            f"only {len(qualifying)} slices under {max_slice_thickness_mm}mm; "
            f"need {min_slices}"
        )
    return qualifying


def _slice_position(ds: pydicom.Dataset) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) != 3:
        raise ValueError("slice missing ImagePositionPatient")
    return float(ipp[2])


def order_slices(datasets: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    """Return slices sorted by z position ascending. Raises on missing positions."""
    return sorted(datasets, key=_slice_position)


def stack_volume(slices: list[pydicom.Dataset]) -> np.ndarray:
    """Stack ordered slices into a 3D array of raw pixel values.

    Returns shape (z, y, x) as int16-compatible numpy array.
    """
    if not slices:
        raise ValueError("empty slice list")
    arrays = [s.pixel_array for s in slices]
    return np.stack(arrays, axis=0)
