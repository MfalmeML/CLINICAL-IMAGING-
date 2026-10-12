"""Temporal engine orchestrator.

Composes registration, lesion matching, and change detection into a
single call. Input is a prior study, a current study, lesion candidates
from each, and the interval between them. Output is a structured change
report.

The orchestrator does not fabricate a change signal when no prior exists.
The build plan is explicit about this: without a prior, the temporal
output is null, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imagex.temporal.change import LesionChange, compute_change
from imagex.temporal.matching import Lesion, MatchResult, match_lesions
from imagex.temporal.registration import (
    RegistrationResult,
    register_rigid,
    resample_to_fixed,
)


@dataclass(frozen=True)
class TemporalReport:
    """Structured output of the temporal engine for one study pair."""
    registration: RegistrationResult
    match: MatchResult
    changes: list[LesionChange]
    interval_days: float


def run_temporal(
    prior_volume: np.ndarray,
    current_volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    prior_lesions: list[Lesion],
    current_lesions: list[Lesion],
    interval_days: float,
    distance_tolerance_mm: float = 10.0,
    registration_iterations: int = 100,
) -> TemporalReport:
    """Run the full temporal engine on one prior/current pair.

    prior_volume, current_volume: (z, y, x) float arrays.
    spacing_xyz: shared voxel spacing of both volumes.
    prior_lesions, current_lesions: lesion candidates in each study,
        already expressed in millimeters in each study's own physical
        coordinate frame.
    interval_days: days between the two studies. Must be > 0.

    Lesion coordinates are expected to be in the same physical frame
    after registration. This function does not transform lesion
    centroids itself; it assumes the caller has already done so, or
    that both studies share an origin. Transforming centroids through
    the registration transform is a caller responsibility because it
    depends on how lesions were detected in each study.

    Raises ValueError if volumes differ in shape, or interval is not positive.
    """
    if prior_volume.shape != current_volume.shape:
        raise ValueError(
            f"prior {prior_volume.shape} and current {current_volume.shape} "
            "must have the same shape"
        )
    if interval_days <= 0.0:
        raise ValueError(f"interval_days must be > 0, got {interval_days}")

    registration = register_rigid(
        fixed=current_volume,
        moving=prior_volume,
        spacing_xyz=spacing_xyz,
        number_of_iterations=registration_iterations,
    )

    match = match_lesions(
        current=current_lesions,
        prior=prior_lesions,
        distance_tolerance_mm=distance_tolerance_mm,
    )

    changes = compute_change(match, interval_days=interval_days)

    return TemporalReport(
        registration=registration,
        match=match,
        changes=changes,
        interval_days=interval_days,
    )


def report_to_dict(report: TemporalReport) -> dict:
    """JSON-serializable view of a TemporalReport."""
    from imagex.temporal.change import change_to_dict
    from imagex.temporal.matching import match_summary

    return {
        "registration": {
            "transform_type": report.registration.transform_type,
            "final_metric_value": report.registration.final_metric_value,
            "iterations": report.registration.iterations,
            "converged": report.registration.converged,
        },
        "match": match_summary(report.match),
        "changes": [change_to_dict(c) for c in report.changes],
        "interval_days": report.interval_days,
    }
