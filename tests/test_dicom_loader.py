"""Tests for dicom_loader. No real DICOM needed yet."""

from pathlib import Path

import pytest

from imagex.preprocessing.dicom_loader import load_dicom_file, load_dicom_series


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dicom_file(tmp_path / "does_not_exist.dcm")


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        load_dicom_series(tmp_path / "no_such_dir")


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_dicom_series(tmp_path)