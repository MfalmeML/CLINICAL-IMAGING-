"""Tests for HU conversion and normalization."""

import numpy as np
import pytest

from imagex.preprocessing.intensity import clip_hu, normalize_minmax, to_hounsfield


def test_to_hounsfield_identity() -> None:
    raw = np.array([[0, 100], [200, 300]], dtype=np.int16)
    hu = to_hounsfield(raw, slope=1.0, intercept=0.0)
    assert hu.dtype == np.float32
    assert np.array_equal(hu, raw.astype(np.float32))


def test_to_hounsfield_applies_slope_and_intercept() -> None:
    raw = np.array([0, 1000], dtype=np.int16)
    hu = to_hounsfield(raw, slope=1.0, intercept=-1024.0)
    assert hu[0] == -1024.0
    assert hu[1] == -24.0


def test_clip_hu_bounds_values() -> None:
    hu = np.array([-2000, -500, 200, 5000], dtype=np.float32)
    out = clip_hu(hu, -1000, 400)
    assert out.tolist() == [-1000.0, -500.0, 200.0, 400.0]


def test_clip_hu_rejects_inverted_window() -> None:
    with pytest.raises(ValueError):
        clip_hu(np.array([0.0]), hu_min=400, hu_max=-1000)


def test_normalize_minmax_endpoints() -> None:
    clipped = np.array([-1000.0, 400.0], dtype=np.float32)
    out = normalize_minmax(clipped, -1000, 400)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)


def test_normalize_minmax_rejects_degenerate_window() -> None:
    with pytest.raises(ValueError):
        normalize_minmax(np.array([0.0]), hu_min=1.0, hu_max=1.0)
