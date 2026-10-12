"""Tests for second-reader discrepancy detection."""

import json

import pytest

from imagex.second_reader.discrepancy import (
    DiscrepancyFlag,
    DiscrepancyKind,
    compare_reads,
    flag_to_dict,
)
from imagex.temporal.matching import Lesion


def _lesion(lid: str, x: float, y: float, z: float, d: float = 5.0) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def test_agreement_when_both_find_same_lesion() -> None:
    ai = [_lesion("a1", 0.0, 0.0, 0.0)]
    rad = [_lesion("r1", 1.0, 0.0, 0.0)]
    flag = compare_reads(ai, rad)
    assert flag.kind == DiscrepancyKind.AGREEMENT
    assert flag.recommendation == "agreement"
    assert flag.matched_count == 1


def test_agreement_when_both_find_nothing() -> None:
    flag = compare_reads([], [])
    assert flag.kind == DiscrepancyKind.AGREEMENT
    assert flag.recommendation == "agreement"
    assert flag.matched_count == 0


def test_ai_only_lesion_flags_review() -> None:
    ai = [_lesion("a1", 0.0, 0.0, 0.0)]
    rad = []
    flag = compare_reads(ai, rad)
    assert flag.kind == DiscrepancyKind.AI_ONLY
    assert flag.recommendation == "review recommended"
    assert len(flag.ai_only_lesions) == 1
    assert len(flag.radiologist_only_lesions) == 0


def test_radiologist_only_lesion_flags_review() -> None:
    ai = []
    rad = [_lesion("r1", 0.0, 0.0, 0.0)]
    flag = compare_reads(ai, rad)
    assert flag.kind == DiscrepancyKind.RADIOLOGIST_ONLY
    assert flag.recommendation == "review recommended"
    assert len(flag.radiologist_only_lesions) == 1
    assert len(flag.ai_only_lesions) == 0


def test_location_mismatch_flags_review() -> None:
    ai = [_lesion("a1", 0.0, 0.0, 0.0), _lesion("a2", 100.0, 0.0, 0.0)]
    rad = [_lesion("r1", 1.0, 0.0, 0.0), _lesion("r2", 200.0, 0.0, 0.0)]
    flag = compare_reads(ai, rad)
    assert flag.kind == DiscrepancyKind.LOCATION_MISMATCH
    assert flag.recommendation == "review recommended"
    assert len(flag.ai_only_lesions) == 1
    assert len(flag.radiologist_only_lesions) == 1
    assert flag.matched_count == 1


def test_never_asserts_correctness() -> None:
    # The DiscrepancyFlag type must have no field that names a winner.
    ai = [_lesion("a1", 0.0, 0.0, 0.0)]
    rad = []
    flag = compare_reads(ai, rad)
    field_names = set(flag.__dataclass_fields__.keys())
    forbidden = {"ai_correct", "radiologist_correct", "ai_wrong", "radiologist_wrong", "winner"}
    assert field_names.isdisjoint(forbidden)


def test_recommendation_is_only_agreement_or_review() -> None:
    for ai, rad in [
        ([], []),
        ([_lesion("a1", 0.0, 0.0, 0.0)], []),
        ([], [_lesion("r1", 0.0, 0.0, 0.0)]),
        ([_lesion("a1", 0.0, 0.0, 0.0)], [_lesion("r1", 0.0, 0.0, 0.0)]),
    ]:
        flag = compare_reads(ai, rad)
        assert flag.recommendation in ("agreement", "review recommended")


def test_flag_to_dict_is_json_serializable() -> None:
    ai = [_lesion("a1", 0.0, 0.0, 0.0)]
    rad = [_lesion("r1", 100.0, 0.0, 0.0)]
    flag = compare_reads(ai, rad)
    d = flag_to_dict(flag)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert "kind" in parsed
    assert "recommendation" in parsed
    assert isinstance(parsed["ai_only_lesion_ids"], list)
    assert isinstance(parsed["radiologist_only_lesion_ids"], list)


def test_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValueError):
        compare_reads([], [], distance_tolerance_mm=0.0)
