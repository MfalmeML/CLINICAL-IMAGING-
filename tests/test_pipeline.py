"""Tests for the end-to-end preprocessing pipeline.

Uses synthetic DICOM written to tmp_path. No real data required.
"""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from imagex.preprocessing.pipeline import preprocess_study


def _write_slice(path: Path, z: float, spacing: tuple[float, float], thickness: float) -> None:
    ds = Dataset()
    ds.Modality = "CT"
    ds.SliceThickness = thickness
    ds.PixelSpacing = list(spacing)
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PatientName = "Doe^John"
    ds.PatientID = "abc123"

    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()

    arr = (np.ones((4, 4)) * (z * 100 + 1024)).astype(np.int16)
    ds.PixelData = arr.tobytes()

    ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def synthetic_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    for i in range(50):
        _write_slice(study / f"slice_{i:03d}.dcm", z=float(i), spacing=(1.0, 1.0), thickness=1.0)
    return study


def test_pipeline_returns_expected_keys(synthetic_study: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "series_selection:\n"
        "  min_slices: 10\n"
        "  max_slice_thickness_mm: 3.0\n"
        "geometry:\n"
        "  target_spacing_mm: [1.0, 1.0, 1.0]\n"
        "intensity:\n"
        "  hu_min: -1000\n"
        "  hu_max: 400\n"
    )
    out = preprocess_study(synthetic_study, cfg)
    assert set(out.keys()) == {"volume", "raw_hu_volume", "spacing", "config"}
    assert out["volume"].dtype == np.float32
    assert out["volume"].ndim == 3
    assert out["volume"].min() >= 0.0
    assert out["volume"].max() <= 1.0


def test_pipeline_output_spacing(synthetic_study: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "series_selection:\n"
        "  min_slices: 10\n"
        "  max_slice_thickness_mm: 3.0\n"
        "geometry:\n"
        "  target_spacing_mm: [1.0, 1.0, 1.0]\n"
        "intensity:\n"
        "  hu_min: -1000\n"
        "  hu_max: 400\n"
    )
    out = preprocess_study(synthetic_study, cfg)
    assert out["spacing"] == (1.0, 1.0, 1.0)


def test_pipeline_rejects_too_few_slices(synthetic_study: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "series_selection:\n"
        "  min_slices: 500\n"
        "  max_slice_thickness_mm: 3.0\n"
        "geometry:\n"
        "  target_spacing_mm: [1.0, 1.0, 1.0]\n"
        "intensity:\n"
        "  hu_min: -1000\n"
        "  hu_max: 400\n"
    )
    with pytest.raises(ValueError):
        preprocess_study(synthetic_study, cfg)
