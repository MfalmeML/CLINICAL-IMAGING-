"""Tests for the temporal engine orchestrator."""

import json

import numpy as np
import pytest

from imagex.temporal.engine import report_to_dict, run_temporal
from imagex.temporal.matching import Lesion


def _make_blob(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0) -> np.ndarray:
    z, y, x = np.indices(shape)
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    return np.exp(-(dist ** 2) / (2 * radius ** 2)).astype(np.float32)


def _lesion(lid: str, x: float, y: float, z: float, d: float) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def test_run_temporal_returns_report() -> None:
    prior = _make_blob(center=(8, 16, 16))
    current = _make_blob(center=(8, 16, 16))
    prior_lesions = [_lesion("p1", 16.0, 16.0, 8.0, 5.0)]
    current_lesions = [_lesion("c1", 16.0, 16.0, 8.0, 7.0)]

    report = run_temporal(
        prior_volume=prior,
        current_volume=current,
        spacing_xyz=(1.0, 1.0, 1.0),
        prior_lesions=prior_lesions,
        current_lesions=current_lesions,
        interval_days=180.0,
        registration_iterations=20,
    )
    assert report.interval_days == 180.0
    assert len(report.changes) == 1
    assert report.changes[0].change_class == "growing"


def test_run_temporal_new_and_resolved() -> None:
    prior = _make_blob()
    current = _make_blob()
    prior_lesions = [_lesion("p1", 16.0, 16.0, 8.0, 5.0), _lesion("p2", 200.0, 16.0, 8.0, 4.0)]
    current_lesions = [_lesion("c1", 16.0, 16.0, 8.0, 5.0), _lesion("c2", 300.0, 16.0, 8.0, 6.0)]

    report = run_temporal(
        prior_volume=prior,
        current_volume=current,
        spacing_xyz=(1.0, 1.0, 1.0),
        prior_lesions=prior_lesions,
        current_lesions=current_lesions,
        interval_days=90.0,
        registration_iterations=20,
    )
    classes = sorted(c.change_class for c in report.changes)
    assert "new" in classes
    assert "resolved" in classes
    assert "stable" in classes


def test_run_temporal_rejects_shape_mismatch() -> None:
    prior = _make_blob(shape=(16, 32, 32))
    current = _make_blob(shape=(16, 32, 24))
    with pytest.raises(ValueError):
        run_temporal(
            prior_volume=prior,
            current_volume=current,
            spacing_xyz=(1.0, 1.0, 1.0),
            prior_lesions=[],
            current_lesions=[],
            interval_days=180.0,
        )


def test_run_temporal_rejects_non_positive_interval() -> None:
    vol = _make_blob()
    with pytest.raises(ValueError):
        run_temporal(
            prior_volume=vol,
            current_volume=vol,
            spacing_xyz=(1.0, 1.0, 1.0),
            prior_lesions=[],
            current_lesions=[],
            interval_days=0.0,
        )


def test_report_to_dict_is_json_serializable() -> None:
    prior = _make_blob()
    current = _make_blob()
    prior_lesions = [_lesion("p1", 16.0, 16.0, 8.0, 5.0)]
    current_lesions = [_lesion("c1", 16.0, 16.0, 8.0, 7.0)]

    report = run_temporal(
        prior_volume=prior,
        current_volume=current,
        spacing_xyz=(1.0, 1.0, 1.0),
        prior_lesions=prior_lesions,
        current_lesions=current_lesions,
        interval_days=180.0,
        registration_iterations=10,
    )
    d = report_to_dict(report)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["interval_days"] == 180.0
    assert "registration" in parsed
    assert "match" in parsed
    assert "changes" in parsed
