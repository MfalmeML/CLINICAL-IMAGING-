"""End-to-end preprocessing pipeline: DICOM directory to preprocessed tensor.

Composes the units built in earlier modules. No new logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imagex.config import load_config
from imagex.preprocessing.deidentify import deidentify_dataset
from imagex.preprocessing.dicom_loader import load_dicom_series
from imagex.preprocessing.intensity import clip_hu, normalize_minmax, to_hounsfield
from imagex.preprocessing.resample import resample_to_spacing
from imagex.preprocessing.series import order_slices, select_ct_series, stack_volume


def preprocess_study(dicom_dir: Path, config_path: Path) -> dict[str, Any]:
    """Run full preprocessing on a single study directory.

    Returns a dict with:
      - 'volume': (z, y, x) float32 in [0, 1]
      - 'raw_hu_volume': (z, y, x) float32 before clipping/normalization
      - 'spacing': effective (x, y, z) spacing after resample
      - 'config': the config used, for provenance
    """
    cfg = load_config(config_path)

    datasets = load_dicom_series(dicom_dir)
    datasets = [deidentify_dataset(d) for d in datasets]

    sel = cfg["series_selection"]
    datasets = select_ct_series(
        datasets,
        min_slices=int(sel["min_slices"]),
        max_slice_thickness_mm=float(sel["max_slice_thickness_mm"]),
    )
    datasets = order_slices(datasets)

    raw = stack_volume(datasets)

    slope = float(getattr(datasets[0], "RescaleSlope", 1.0))
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0.0))
    hu = to_hounsfield(raw, slope, intercept)

    intensity = cfg["intensity"]
    clipped = clip_hu(hu, float(intensity["hu_min"]), float(intensity["hu_max"]))

    spacing = getattr(datasets[0], "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(datasets[0], "SliceThickness", 1.0))
    source_spacing = (float(spacing[0]), float(spacing[1]), slice_thickness)

    target_spacing = tuple(float(s) for s in cfg["geometry"]["target_spacing_mm"])
    resampled_hu = resample_to_spacing(clipped, source_spacing, target_spacing)

    normalized = normalize_minmax(
        resampled_hu, float(intensity["hu_min"]), float(intensity["hu_max"])
    )

    return {
        "volume": normalized.astype(np.float32),
        "raw_hu_volume": resampled_hu.astype(np.float32),
        "spacing": target_spacing,
        "config": cfg,
    }
