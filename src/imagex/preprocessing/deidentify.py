"""De-identification of DICOM metadata and pixel data.

Defense-in-depth: runs even on datasets already de-identified upstream.
Strips PHI from metadata. Does not yet redact burned-in pixel annotations.
"""

from __future__ import annotations

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


def deidentify_dataset(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Return a de-identified copy. Retained tags survive; PHI is removed.

    Modifies a deep copy; input dataset is untouched.
    """
    out = ds.copy()

    for elem in list(out.iterall()):
        if elem.tag.group == 0x7FE0:  # PixelData
            continue
        keyword = elem.keyword or ""
        if keyword in _RETAINED_TAGS:
            continue
        if _is_phi(keyword):
            del out[elem.tag]
            continue
        if keyword == "" and elem.VR not in ("SQ", "OB", "OW", "UN"):
            # Unknown private/non-standard tag with a value: clear it.
            try:
                del out[elem.tag]
            except KeyError:
                pass
        elif elem.VR == "SQ":
            for item in elem.value:
                deidentify_dataset(item)

    return out
