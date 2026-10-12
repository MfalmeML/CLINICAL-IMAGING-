"""Tests for rule-based priority assignment."""

import json

import pytest

from imagex.triage.priority import (
    ABNORMALITY_HIGH,
    ABNORMALITY_LOW,
    LARGE_LESION_MM,
    SIGNIFICANT_GROWTH_MM_PER_DAY,
    Priority,
    TriageInput,
    assign_priority,
    triage_input_to_dict,
)


def _input(**overrides) -> TriageInput:
    defaults = dict(
        abnormality_score=0.0,
        max_lesion_diameter_mm=0.0,
        has_prior=False,
        growth_detected=False,
        new_lesion_detected=False,
        max_growth_rate_mm_per_day=0.0,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
    )
    defaults.update(overrides)
    return TriageInput(**defaults)


def test_no_signal_is_low() -> None:
    assert assign_priority(_input()) == Priority.LOW


def test_safety_human_review_forces_urgent() -> None:
    inp = _input(safety_recommendation="human-review")
    assert assign_priority(inp) == Priority.URGENT


def test_ood_forces_urgent() -> None:
    inp = _input(is_ood=True)
    assert assign_priority(inp) == Priority.URGENT


def test_safety_override_beats_other_high_signal() -> None:
    inp = _input(
        safety_recommendation="human-review",
        abnormality_score=0.99,
        max_lesion_diameter_mm=30.0,
        has_prior=True,
        growth_detected=True,
        max_growth_rate_mm_per_day=1.0,
    )
    assert assign_priority(inp) == Priority.URGENT


def test_new_lesion_with_prior_is_urgent() -> None:
    inp = _input(has_prior=True, new_lesion_detected=True)
    assert assign_priority(inp) == Priority.URGENT


def test_new_lesion_without_prior_is_not_urgent() -> None:
    inp = _input(has_prior=False, new_lesion_detected=True)
    # Without a prior, "new" is meaningless. Should not escalate to URGENT.
    assert assign_priority(inp) != Priority.URGENT


def test_significant_growth_is_high() -> None:
    inp = _input(
        has_prior=True,
        growth_detected=True,
        max_growth_rate_mm_per_day=SIGNIFICANT_GROWTH_MM_PER_DAY + 0.001,
    )
    assert assign_priority(inp) == Priority.HIGH


def test_minor_growth_is_medium() -> None:
    inp = _input(
        has_prior=True,
        growth_detected=True,
        max_growth_rate_mm_per_day=SIGNIFICANT_GROWTH_MM_PER_DAY / 2.0,
    )
    assert assign_priority(inp) == Priority.MEDIUM


def test_high_abnormality_large_lesion_is_high() -> None:
    inp = _input(
        abnormality_score=ABNORMALITY_HIGH + 0.1,
        max_lesion_diameter_mm=LARGE_LESION_MM + 2.0,
    )
    assert assign_priority(inp) == Priority.HIGH


def test_high_abnormality_small_lesion_is_medium() -> None:
    inp = _input(
        abnormality_score=ABNORMALITY_HIGH + 0.1,
        max_lesion_diameter_mm=LARGE_LESION_MM - 2.0,
    )
    assert assign_priority(inp) == Priority.MEDIUM


def test_moderate_abnormality_medium_lesion_is_medium() -> None:
    inp = _input(
        abnormality_score=ABNORMALITY_LOW + 0.05,
        max_lesion_diameter_mm=8.0,
    )
    assert assign_priority(inp) == Priority.MEDIUM


def test_low_abnormality_small_lesion_is_low() -> None:
    inp = _input(
        abnormality_score=ABNORMALITY_LOW - 0.05,
        max_lesion_diameter_mm=8.0,
    )
    assert assign_priority(inp) == Priority.LOW


def test_priority_is_deterministic() -> None:
    inp = _input(
        abnormality_score=0.5,
        max_lesion_diameter_mm=7.0,
        has_prior=True,
        growth_detected=True,
        max_growth_rate_mm_per_day=0.01,
    )
    results = {assign_priority(inp) for _ in range(10)}
    assert len(results) == 1


def test_triage_input_to_dict_is_json_serializable() -> None:
    d = triage_input_to_dict(_input(abnormality_score=0.5))
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["abnormality_score"] == 0.5
    assert parsed["safety_recommendation"] == "safe-to-predict"


def test_priority_enum_values() -> None:
    assert Priority.LOW.value == "LOW"
    assert Priority.URGENT.value == "URGENT"
