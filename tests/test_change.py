"""Tests for change detection and growth estimation."""

import json

import pytest

from imagex.temporal.change import (
    STABLE_THRESHOLD_MM,
    LesionChange,
    change_to_dict,
    compute_change,
)
from imagex.temporal.matching import Lesion, MatchResult, match_lesions


def _lesion(lid: str, x: float, y: float, z: float, d: float) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def _one_match(
    current_d: float,
    prior_d: float,
    distance: float = 1.0,
) -> MatchResult:
    current = [_lesion("c1", distance, 0.0, 0.0, current_d)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0, prior_d)]
    return match_lesions(current, prior, distance_tolerance_mm=5.0)


def test_stable_when_change_below_threshold() -> None:
    m = _one_match(current_d=5.2, prior_d=5.0)
    changes = compute_change(m, interval_days=180.0)
    assert len(changes) == 1
    assert changes[0].change_class == "stable"


def test_growing_when_above_threshold_positive() -> None:
    m = _one_match(current_d=7.0, prior_d=5.0)
    changes = compute_change(m, interval_days=180.0)
    assert changes[0].change_class == "growing"
    assert changes[0].diameter_change_mm == pytest.approx(2.0)


def test_shrinking_when_above_threshold_negative() -> None:
    m = _one_match(current_d=3.0, prior_d=5.0)
    changes = compute_change(m, interval_days=180.0)
    assert changes[0].change_class == "shrinking"
    assert changes[0].diameter_change_mm == pytest.approx(-2.0)


def test_growth_rate_computed_per_day() -> None:
    m = _one_match(current_d=7.0, prior_d=5.0)
    changes = compute_change(m, interval_days=100.0)
    assert changes[0].growth_rate_mm_per_interval == pytest.approx(2.0 / 100.0)


def test_growth_rate_is_nan_when_interval_missing() -> None:
    m = _one_match(current_d=7.0, prior_d=5.0)
    changes = compute_change(m, interval_days=None)
    assert changes[0].growth_rate_mm_per_interval != changes[0].growth_rate_mm_per_interval  # NaN


def test_volume_change_percent_positive_for_growth() -> None:
    m = _one_match(current_d=7.0, prior_d=5.0)
    changes = compute_change(m, interval_days=180.0)
    assert changes[0].volume_change_percent > 0.0


def test_volume_change_percent_negative_for_shrinkage() -> None:
    m = _one_match(current_d=3.0, prior_d=5.0)
    changes = compute_change(m, interval_days=180.0)
    assert changes[0].volume_change_percent < 0.0


def test_new_lesion_classified() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0, 5.0), _lesion("c2", 100.0, 0.0, 0.0, 4.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0, 5.0)]
    m = match_lesions(current, prior, distance_tolerance_mm=5.0)
    changes = compute_change(m, interval_days=180.0)
    new = [c for c in changes if c.change_class == "new"]
    assert len(new) == 1
    assert new[0].lesion_id == "c2"
    assert new[0].prior_lesion_id is None


def test_resolved_lesion_classified() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0, 5.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0, 5.0), _lesion("p2", 100.0, 0.0, 0.0, 4.0)]
    m = match_lesions(current, prior, distance_tolerance_mm=5.0)
    changes = compute_change(m, interval_days=180.0)
    resolved = [c for c in changes if c.change_class == "resolved"]
    assert len(resolved) == 1
    assert resolved[0].lesion_id == "p2"


def test_rejects_non_positive_interval() -> None:
    m = _one_match(5.0, 5.0)
    with pytest.raises(ValueError):
        compute_change(m, interval_days=0.0)
    with pytest.raises(ValueError):
        compute_change(m, interval_days=-10.0)


def test_change_to_dict_json_serializable_for_all_classes() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0, 7.0), _lesion("c2", 100.0, 0.0, 0.0, 4.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0, 5.0), _lesion("p2", 200.0, 0.0, 0.0, 4.0)]
    m = match_lesions(current, prior, distance_tolerance_mm=5.0)
    changes = compute_change(m, interval_days=180.0)
    for c in changes:
        d = change_to_dict(c)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["change_class"] in ("stable", "growing", "shrinking", "new", "resolved")


def test_nan_converted_to_none_in_dict() -> None:
    change = LesionChange(
        lesion_id="c1",
        prior_lesion_id=None,
        change_class="new",
        diameter_change_mm=float("nan"),
        volume_change_percent=float("nan"),
        growth_rate_mm_per_interval=float("nan"),
        interval_days=None,
    )
    d = change_to_dict(change)
    assert d["diameter_change_mm"] is None
    assert d["volume_change_percent"] is None
    assert d["growth_rate_mm_per_interval"] is None
