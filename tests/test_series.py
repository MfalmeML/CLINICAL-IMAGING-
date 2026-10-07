"""Tests for series selection, ordering, and stacking."""

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from imagex.preprocessing.series import order_slices, select_ct_series, stack_volume


def _slice(z: float, thickness: float = 1.0, modality: str = "CT") -> Dataset:
    ds = Dataset()
    # pydicom 3 refuses to decode pixel_array without a transfer syntax.
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Modality = modality
    ds.SliceThickness = thickness
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = (np.ones((2, 2), dtype=np.uint16) * int(z * 10)).tobytes()
    return ds


def test_select_filters_non_ct() -> None:
    mixed = [_slice(0), _slice(1, modality="MR"), _slice(2)]
    out = select_ct_series(mixed, min_slices=2, max_slice_thickness_mm=3.0)
    assert len(out) == 2


def test_select_rejects_thick_slices() -> None:
    thick = [_slice(i, thickness=5.0) for i in range(5)]
    with pytest.raises(ValueError):
        select_ct_series(thick, min_slices=2, max_slice_thickness_mm=3.0)


def test_select_rejects_too_few() -> None:
    few = [_slice(0), _slice(1)]
    with pytest.raises(ValueError):
        select_ct_series(few, min_slices=40, max_slice_thickness_mm=3.0)


def test_select_rejects_empty() -> None:
    with pytest.raises(ValueError):
        select_ct_series([], min_slices=1, max_slice_thickness_mm=3.0)


def test_order_slices_ascending_z() -> None:
    shuffled = [_slice(3), _slice(1), _slice(2), _slice(0)]
    ordered = order_slices(shuffled)
    zs = [float(s.ImagePositionPatient[2]) for s in ordered]
    assert zs == [0.0, 1.0, 2.0, 3.0]


def test_order_slices_missing_position_raises() -> None:
    bad = Dataset()
    bad.Modality = "CT"
    with pytest.raises(ValueError):
        order_slices([bad])


def test_stack_volume_shape() -> None:
    slices = order_slices([_slice(0), _slice(1), _slice(2)])
    vol = stack_volume(slices)
    assert vol.shape == (3, 2, 2)


def test_stack_volume_empty_raises() -> None:
    with pytest.raises(ValueError):
        stack_volume([])