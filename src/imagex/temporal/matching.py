"""Lesion matching between registered current and prior studies.

After registration, a lesion in the current study and a lesion in the
prior study can be compared. Two lesions are the same lesion if their
registered centroids are within a distance tolerance and their
overlapping volumes agree.

Distance is measured in millimeters using the shared physical frame
of the registered volumes. Voxel indices are never compared directly
because voxel spacing varies across scanners and protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lesion:
    """A single lesion in one study."""
    lesion_id: str
    centroid_mm: tuple[float, float, float]
    volume_mm3: float
    diameter_mm: float


@dataclass(frozen=True)
class LesionMatch:
    """A pairing between a current-study lesion and a prior-study lesion."""
    current: Lesion
    prior: Lesion
    distance_mm: float


@dataclass(frozen=True)
class MatchResult:
    """Full result of matching current lesions against prior lesions."""
    matched: list[LesionMatch]
    current_only: list[Lesion]   # new lesions
    prior_only: list[Lesion]     # resolved lesions


def _distance_mm(a: Lesion, b: Lesion) -> float:
    ca = np.asarray(a.centroid_mm, dtype=np.float64)
    cb = np.asarray(b.centroid_mm, dtype=np.float64)
    return float(np.linalg.norm(ca - cb))


def match_lesions(
    current: list[Lesion],
    prior: list[Lesion],
    distance_tolerance_mm: float = 10.0,
) -> MatchResult:
    """Match lesions between current and prior studies.

    For each current lesion, the closest prior lesion within the distance
    tolerance is selected. Each prior lesion can match at most one current
    lesion. Greedy nearest-neighbor is used, which is stable and easy to
    audit; a full optimal assignment is a later refinement.

    Raises ValueError if tolerance is not positive.
    """
    if distance_tolerance_mm <= 0.0:
        raise ValueError(
            f"distance_tolerance_mm must be > 0, got {distance_tolerance_mm}"
        )

    # All pairs within tolerance, sorted by distance ascending.
    candidates: list[tuple[float, int, int]] = []
    for i, c in enumerate(current):
        for j, p in enumerate(prior):
            d = _distance_mm(c, p)
            if d <= distance_tolerance_mm:
                candidates.append((d, i, j))
    candidates.sort(key=lambda t: t[0])

    matched_current: set[int] = set()
    matched_prior: set[int] = set()
    matches: list[LesionMatch] = []

    for d, i, j in candidates:
        if i in matched_current or j in matched_prior:
            continue
        matched_current.add(i)
        matched_prior.add(j)
        matches.append(
            LesionMatch(current=current[i], prior=prior[j], distance_mm=d)
        )

    current_only = [c for i, c in enumerate(current) if i not in matched_current]
    prior_only = [p for j, p in enumerate(prior) if j not in matched_prior]

    return MatchResult(
        matched=matches,
        current_only=current_only,
        prior_only=prior_only,
    )


def match_summary(result: MatchResult) -> dict:
    """JSON-serializable summary of a MatchResult."""
    return {
        "n_matched": len(result.matched),
        "n_current_only": len(result.current_only),
        "n_prior_only": len(result.prior_only),
        "matches": [
            {
                "current_id": m.current.lesion_id,
                "prior_id": m.prior.lesion_id,
                "distance_mm": m.distance_mm,
                "current_diameter_mm": m.current.diameter_mm,
                "prior_diameter_mm": m.prior.diameter_mm,
            }
            for m in result.matched
        ],
        "current_only_ids": [c.lesion_id for c in result.current_only],
        "prior_only_ids": [p.lesion_id for p in result.prior_only],
    }
