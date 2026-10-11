"""Tests for the preprocessing CLI. Uses synthetic DICOM."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from imagex.cli.preprocess_dataset import main


def _write_slice(path: Path, z: float) -> None:
    ds = Dataset()
    ds.Modality = "CT"
    ds.SliceThickness = 1.0
    ds.PixelSpacing = [1.0, 1.0]
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


def _make_study(root: Path, name: str, n_slices: int = 50) -> None:
    study = root / name
    study.mkdir()
    for i in range(n_slices):
        _write_slice(study / f"slice_{i:03d}.dcm", float(i))


def _write_config(path: Path) -> None:
    path.write_text(
        "series_selection:\n"
        "  min_slices: 10\n"
        "  max_slice_thickness_mm: 3.0\n"
        "geometry:\n"
        "  target_spacing_mm: [1.0, 1.0, 1.0]\n"
        "intensity:\n"
        "  hu_min: -1000\n"
        "  hu_max: 400\n"
    )


def test_preprocess_one_study(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    _make_study(dicom_root, "study_a")

    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg)

    code = main(
        [
            "--dicom-root", str(dicom_root),
            "--output-dir", str(out),
            "--config", str(cfg),
        ]
    )
    assert code == 0
    assert (out / "study_a.npy").is_file()
    arr = np.load(out / "study_a.npy")
    assert arr.dtype == np.float32
    assert arr.ndim == 3


def test_preprocess_skips_existing(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    _make_study(dicom_root, "study_a")

    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "study_a.npy", np.zeros((1, 1, 1), dtype=np.float32))

    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg)

    code = main(
        [
            "--dicom-root", str(dicom_root),
            "--output-dir", str(out),
            "--config", str(cfg),
        ]
    )
    assert code == 0
    # Existing file was not overwritten.
    arr = np.load(out / "study_a.npy")
    assert arr.shape == (1, 1, 1)


def test_preprocess_failure_returns_nonzero(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    _make_study(dicom_root, "study_a", n_slices=3)  # too few slices

    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg)

    code = main(
        [
            "--dicom-root", str(dicom_root),
            "--output-dir", str(out),
            "--config", str(cfg),
        ]
    )
    assert code == 1
