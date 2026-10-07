"""DICOM loading and validation. Phase 1 foundation."""

from pathlib import Path
import pydicom


def load_dicom_file(path: Path) -> pydicom.Dataset:
    """Load a single DICOM file. Raises on unreadable input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such DICOM file: {path}")
    return pydicom.dcmread(path, force=False)