"""Full inference pipeline orchestrator.

Composes the engines in the order the build plan specifies:

    quality gate -> perception -> temporal (if prior) -> safety -> triage -> second-reader

The orchestrator enforces two invariants:

1. No prior means temporal output is None, not a fabricated default.
2. Safety recommendation has override precedence over triage output.

Every engine in this pipeline takes precomputed inputs from its
predecessors. This module does not run perception models itself; it
takes a `PerceptionOutput` as input. That keeps the pipeline testable
without a trained model and keeps the composition logic separate from
model execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from imagex.second_reader.discrepancy import (
    DiscrepancyFlag,
    compare_reads,
)
from imagex.temporal.change import LesionChange
from imagex.temporal.matching import Lesion
from imagex.triage.priority import Priority, TriageInput, assign_priority


@dataclass(frozen=True)
class PerceptionOutput:
    """Perception engine result for one study. Caller-provided."""
    abnormality_score: float
    lesions: list[Lesion]


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end result for one case."""
    perception: PerceptionOutput
    temporal_changes: list[LesionChange] | None  # None if no prior
    triage_input: TriageInput
    priority: Priority
    discrepancy: DiscrepancyFlag | None  # None if no radiologist read


def _summarize_temporal(
    changes: list[LesionChange],
) -> tuple[bool, bool, float]:
    """Return (growth_detected, new_lesion_detected, max_growth_rate)."""
    growth = any(c.change_class == "growing" for c in changes)
    new = any(c.change_class == "new" for c in changes)
    rates = [
        c.growth_rate_mm_per_interval
        for c in changes
        if c.change_class == "growing" and c.growth_rate_mm_per_interval == c.growth_rate_mm_per_interval
    ]
    max_rate = max(rates) if rates else 0.0
    return growth, new, max_rate


def _max_lesion_diameter(lesions: list[Lesion]) -> float:
    if not lesions:
        return 0.0
    return max(l.diameter_mm for l in lesions)


def run_pipeline(
    perception: PerceptionOutput,
    safety_recommendation: str,
    uncertainty: float,
    is_ood: bool,
    temporal_changes: list[LesionChange] | None = None,
    radiologist_lesions: list[Lesion] | None = None,
) -> PipelineResult:
    """Compose engines into one end-to-end result.

    temporal_changes: output of the temporal engine, or None if no prior.
    radiologist_lesions: available radiologist findings, or None if no
        radiologist read is available.

    Raises ValueError if safety_recommendation is not a recognized value.
    """
    if safety_recommendation not in ("safe-to-predict", "human-review"):
        raise ValueError(
            f"unrecognized safety_recommendation: {safety_recommendation!r}"
        )

    has_prior = temporal_changes is not None

    if has_prior:
        growth, new, max_rate = _summarize_temporal(temporal_changes)  # type: ignore[arg-type]
    else:
        growth, new, max_rate = False, False, 0.0

    triage_in = TriageInput(
        abnormality_score=perception.abnormality_score,
        max_lesion_diameter_mm=_max_lesion_diameter(perception.lesions),
        has_prior=has_prior,
        growth_detected=growth,
        new_lesion_detected=new,
        max_growth_rate_mm_per_day=max_rate,
        safety_recommendation=safety_recommendation,
        uncertainty=uncertainty,
        is_ood=is_ood,
    )

    priority = assign_priority(triage_in)

    discrepancy: DiscrepancyFlag | None = None
    if radiologist_lesions is not None:
        discrepancy = compare_reads(perception.lesions, radiologist_lesions)

    return PipelineResult(
        perception=perception,
        temporal_changes=temporal_changes,
        triage_input=triage_in,
        priority=priority,
        discrepancy=discrepancy,
    )


def pipeline_result_to_dict(result: PipelineResult) -> dict:
    """JSON-serializable view for logging and dashboard rendering."""
    from imagex.second_reader.discrepancy import flag_to_dict
    from imagex.temporal.change import change_to_dict
    from imagex.triage.priority import triage_input_to_dict

    return {
        "perception": {
            "abnormality_score": result.perception.abnormality_score,
            "lesion_ids": [l.lesion_id for l in result.perception.lesions],
        },
        "temporal_changes": (
            None
            if result.temporal_changes is None
            else [change_to_dict(c) for c in result.temporal_changes]
        ),
        "triage_input": triage_input_to_dict(result.triage_input),
        "priority": result.priority.value,
        "discrepancy": (
            None if result.discrepancy is None else flag_to_dict(result.discrepancy)
        ),
    }
