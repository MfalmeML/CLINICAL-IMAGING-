"""Tests for the full pipeline orchestrator."""

import json

import pytest

from imagex.pipeline import (
    PerceptionOutput,
    pipeline_result_to_dict,
    run_pipeline,
)
from imagex.temporal.change import LesionChange
from imagex.temporal.matching import Lesion
from imagex.triage.priority import Priority


def _lesion(lid: str, x: float, y: float, z: float, d: float = 5.0) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def _growth(
    lid: str, class_: str, rate: float = 0.0
) -> LesionChange:
    return LesionChange(
        lesion_id=lid,
        prior_lesion_id=lid,
        change_class=class_,
        diameter_change_mm=1.0 if class_ == "growing" else 0.0,
        volume_change_percent=10.0 if class_ == "growing" else 0.0,
        growth_rate_mm_per_interval=rate,
        interval_days=180.0,
    )


def test_pipeline_no_prior_no_read() -> None:
    perc = PerceptionOutput(abnormality_score=0.1, lesions=[])
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
    )
    assert result.temporal_changes is None
    assert result.priority == Priority.LOW
    assert result.discrepancy is None


def test_pipeline_no_prior_does_not_fabricate_temporal_signal() -> None:
    perc = PerceptionOutput(abnormality_score=0.1, lesions=[])
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
    )
    assert result.triage_input.has_prior is False
    assert result.triage_input.growth_detected is False
    assert result.triage_input.new_lesion_detected is False
    assert result.triage_input.max_growth_rate_mm_per_day == 0.0


def test_pipeline_with_growth_escalates() -> None:
    perc = PerceptionOutput(
        abnormality_score=0.4,
        lesions=[_lesion("a1", 0.0, 0.0, 0.0, d=8.0)],
    )
    changes = [_growth("a1", "growing", rate=0.05)]
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
        temporal_changes=changes,
    )
    assert result.triage_input.has_prior is True
    assert result.triage_input.growth_detected is True
    assert result.priority == Priority.HIGH


def test_pipeline_new_lesion_escalates_to_urgent() -> None:
    perc = PerceptionOutput(
        abnormality_score=0.5,
        lesions=[_lesion("a1", 0.0, 0.0, 0.0, d=6.0)],
    )
    changes = [_growth("a1", "new")]
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
        temporal_changes=changes,
    )
    assert result.priority == Priority.URGENT


def test_safety_override_beats_strong_perception_signal() -> None:
    perc = PerceptionOutput(
        abnormality_score=0.99,
        lesions=[_lesion("a1", 0.0, 0.0, 0.0, d=30.0)],
    )
    result = run_pipeline(
        perception=perc,
        safety_recommendation="human-review",
        uncertainty=5.0,
        is_ood=False,
    )
    assert result.priority == Priority.URGENT


def test_ood_forces_urgent_even_with_weak_perception() -> None:
    perc = PerceptionOutput(abnormality_score=0.0, lesions=[])
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=True,
    )
    assert result.priority == Priority.URGENT


def test_radiologist_read_produces_discrepancy_flag() -> None:
    perc = PerceptionOutput(
        abnormality_score=0.5,
        lesions=[_lesion("a1", 0.0, 0.0, 0.0)],
    )
    rad = [_lesion("r1", 1.0, 0.0, 0.0)]
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
        radiologist_lesions=rad,
    )
    assert result.discrepancy is not None
    assert result.discrepancy.recommendation == "agreement"


def test_no_radiologist_read_leaves_discrepancy_none() -> None:
    perc = PerceptionOutput(abnormality_score=0.5, lesions=[_lesion("a1", 0.0, 0.0, 0.0)])
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
    )
    assert result.discrepancy is None


def test_rejects_unknown_safety_recommendation() -> None:
    perc = PerceptionOutput(abnormality_score=0.5, lesions=[])
    with pytest.raises(ValueError):
        run_pipeline(
            perception=perc,
            safety_recommendation="maybe",
            uncertainty=0.0,
            is_ood=False,
        )


def test_pipeline_result_to_dict_is_json_serializable_all_cases() -> None:
    perc = PerceptionOutput(
        abnormality_score=0.5,
        lesions=[_lesion("a1", 0.0, 0.0, 0.0, d=8.0)],
    )
    changes = [_growth("a1", "growing", rate=0.05)]
    rad = [_lesion("r1", 1.0, 0.0, 0.0), _lesion("r2", 100.0, 0.0, 0.0)]
    result = run_pipeline(
        perception=perc,
        safety_recommendation="safe-to-predict",
        uncertainty=0.0,
        is_ood=False,
        temporal_changes=changes,
        radiologist_lesions=rad,
    )
    d = pipeline_result_to_dict(result)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["priority"] in ("LOW", "MEDIUM", "HIGH", "URGENT")
    assert parsed["temporal_changes"] is not None
    assert parsed["discrepancy"] is not None
