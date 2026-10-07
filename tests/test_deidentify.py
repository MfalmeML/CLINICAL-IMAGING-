"""Tests for DICOM de-identification."""

import pydicom
import pytest
from pydicom.dataset import Dataset

from imagex.preprocessing.deidentify import deidentify_dataset


def _make_dataset() -> Dataset:
    ds = Dataset()
    ds.PatientName = "Doe^John"
    ds.PatientID = "12345"
    ds.StudyDate = "20260301"
    ds.InstitutionName = "Some Hospital"
    ds.Modality = "CT"
    ds.Manufacturer = "GE"
    ds.ConvolutionKernel = "STANDARD"
    ds.SliceThickness = 1.25
    ds.Rows = 512
    ds.Columns = 512
    return ds


def test_phi_removed() -> None:
    out = deidentify_dataset(_make_dataset())
    assert "PatientName" not in out
    assert "PatientID" not in out
    assert "StudyDate" not in out
    assert "InstitutionName" not in out


def test_retained_tags_survive() -> None:
    out = deidentify_dataset(_make_dataset())
    assert out.Modality == "CT"
    assert out.Manufacturer == "GE"
    assert out.ConvolutionKernel == "STANDARD"
    assert out.SliceThickness == 1.25
    assert out.Rows == 512


def test_input_is_not_mutated() -> None:
    ds = _make_dataset()
    _ = deidentify_dataset(ds)
    assert ds.PatientName == "Doe^John"
    assert ds.PatientID == "12345"


def test_idempotent() -> None:
    once = deidentify_dataset(_make_dataset())
    twice = deidentify_dataset(once)
    assert twice.Modality == "CT"
    assert "PatientName" not in twice
