"""Tests for lesion matching between studies."""

import json

import pytest

from imagex.temporal.matching import Lesion, match_lesions, match_summary


def _lesion(lid: str, x: float, y: float, z: float, d: float = 5.0) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def test_identical_positions_match() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0, d=6.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0, d=5.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert r.matched[0].current.lesion_id == "c1"
    assert r.matched[0].prior.lesion_id == "p1"
    assert r.matched[0].distance_mm == pytest.approx(0.0)


def test_within_tolerance_matches() -> None:
    current = [_lesion("c1", 2.0, 0.0, 0.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert r.matched[0].distance_mm == pytest.approx(2.0)


def test_outside_tolerance_does_not_match() -> None:
    current = [_lesion("c1", 20.0, 0.0, 0.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 0
    assert len(r.current_only) == 1
    assert len(r.prior_only) == 1


def test_new_lesion_goes_to_current_only() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0), _lesion("c2", 100.0, 0.0, 0.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert len(r.current_only) == 1
    assert r.current_only[0].lesion_id == "c2"
    assert len(r.prior_only) == 0


def test_resolved_lesion_goes_to_prior_only() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0)]
    prior = [_lesion("p1", 0.0, 0.0, 0.0), _lesion("p2", 100.0, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert len(r.prior_only) == 1
    assert r.prior_only[0].lesion_id == "p2"


def test_greedy_nearest_wins() -> None:
    # Current lesion is closer to p2 than to p1; p2 should be chosen.
    current = [_lesion("c1", 1.0, 0.0, 0.0)]
    prior = [_lesion("p1", 4.0, 0.0, 0.0), _lesion("p2", 1.5, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert r.matched[0].prior.lesion_id == "p2"


def test_one_prior_matches_at_most_one_current() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0), _lesion("c2", 1.0, 0.0, 0.0)]
    prior = [_lesion("p1", 0.5, 0.0, 0.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    assert len(r.matched) == 1
    assert len(r.current_only) == 1


def test_empty_inputs() -> None:
    r = match_lesions([], [], distance_tolerance_mm=5.0)
    assert r.matched == []
    assert r.current_only == []
    assert r.prior_only == []


def test_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValueError):
        match_lesions([], [], distance_tolerance_mm=0.0)
    with pytest.raises(ValueError):
        match_lesions([], [], distance_tolerance_mm=-1.0)


def test_match_summary_is_json_serializable() -> None:
    current = [_lesion("c1", 0.0, 0.0, 0.0, d=6.0)]
    prior = [_lesion("p1", 1.0, 0.0, 0.0, d=5.0)]
    r = match_lesions(current, prior, distance_tolerance_mm=5.0)
    d = match_summary(r)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["n_matched"] == 1
    assert parsed["matches"][0]["current_diameter_mm"] == 6.0
    assert parsed["matches"][0]["prior_diameter_mm"] == 5.0
