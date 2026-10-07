"""DICOM loading and validation. Phase 1 foundation."""

from pathlib import Path

import pydicom


def load_dicom_file(path: Path) -> pydicom.Dataset:
    """Load a single DICOM file. Raises on unreadable input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such DICOM file: {path}")
    return pydicom.dcmread(path, force=False)


def load_dicom_series(directory: Path) -> list[pydicom.Dataset]:
    """Load all readable DICOM files in a directory.

    Order is not guaranteed. Unreadable files are skipped silently.
    Raises NotADirectoryError if path is not a directory.
    Raises ValueError if no readable DICOM files are found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    datasets: list[pydicom.Dataset] = []
    for f in sorted(directory.glob("*")):
        try:
            datasets.append(load_dicom_file(f))
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No readable DICOM files in: {directory}")

    return datasets
