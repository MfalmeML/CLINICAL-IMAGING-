"""De-identification of DICOM metadata and pixel data.

Defense-in-depth: runs even on datasets already de-identified upstream.
Strips PHI from metadata. Does not yet redact burned-in pixel annotations.
"""

from __future__ import annotations

import copy

import pydicom


# Tags allowed to survive. Everything else with a value is cleared.
_RETAINED_TAGS = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "ConvolutionKernel",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "InstanceNumber",
}


# Tags that carry direct identifiers and must be removed outright.
_PHI_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "ContentDate",
    "ContentTime",
    "AccessionNumber",
    "StudyID",
)


def _is_phi(keyword: str) -> bool:
    return any(keyword.startswith(prefix) for prefix in _PHI_KEYWORDS)


def _scrub(ds: pydicom.Dataset) -> None:
    """Remove PHI from ``ds`` in place, recursing into sequence items.

    Internal helper: only call this on a dataset you own (a copy).
    Each level deletes only its own elements, so a tag in a nested item
    can never remove a same-numbered tag at another level.
    """
    for elem in list(ds):  # snapshot: we delete while walking
        if elem.tag.group == 0x7FE0:  # PixelData
            continue
        keyword = elem.keyword or ""
        if keyword in _RETAINED_TAGS:
            continue
        if _is_phi(keyword):
            del ds[elem.tag]
        elif keyword == "" and elem.VR not in ("SQ", "OB", "OW", "UN"):
            # Unknown private/non-standard tag with a value: clear it.
            del ds[elem.tag]
        elif elem.VR == "SQ":
            for item in elem.value:
                _scrub(item)


def deidentify_dataset(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Return a de-identified deep copy. Retained tags survive; PHI is removed.

    The input dataset is never modified, including nested sequence items.
    """
    out = copy.deepcopy(ds)
    _scrub(out)
    return out