"""Tests for dicom_loader. No real DICOM needed yet."""

from pathlib import Path

import pytest

from imagex.preprocessing.dicom_loader import load_dicom_file, load_dicom_series


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.dcm"
    with pytest.raises(FileNotFoundError):
        load_dicom_file(missing)


def test_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_dir"
    with pytest.raises(NotADirectoryError):
        load_dicom_series(missing)