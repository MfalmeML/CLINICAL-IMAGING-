"""Load DICOM files and series from disk."""

from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset


def load_dicom_file(path: str | Path) -> FileDataset:
    """Read a single DICOM file.

    Raises:
        FileNotFoundError: if ``path`` does not exist or is not a file.
        pydicom.errors.InvalidDicomError: if the file is not valid DICOM.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    return pydicom.dcmread(path)


def load_dicom_series(directory: str | Path) -> list[FileDataset]:
    """Read every ``.dcm`` file in a directory, ordered by InstanceNumber.

    Raises:
        NotADirectoryError: if ``directory`` is not an existing directory.
        ValueError: if the directory contains no ``.dcm`` files.
        pydicom.errors.InvalidDicomError: if any ``.dcm`` file is invalid.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {directory}")

    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"
    )
    if not files:
        raise ValueError(f"No .dcm files found in: {directory}")

    datasets = [load_dicom_file(p) for p in files]
    # Slices are not guaranteed to sort correctly by filename.
    datasets.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    return datasets