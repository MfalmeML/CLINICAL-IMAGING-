"""Tests for dicom_loader. No real DICOM needed yet."""

from pathlib import Path

import pytest

from imagex.preprocessing.dicom_loader import load_dicom_file


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.dcm"
    with pytest.raises(FileNotFoundError):
        load_dicom_file(missing)