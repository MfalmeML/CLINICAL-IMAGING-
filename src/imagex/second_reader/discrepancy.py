"""Second-reader discrepancy detection.

Compares an independent AI read against an available radiologist read.
Produces a discrepancy flag, never a verdict.

The framing matters as much as the mechanism. On disagreement, the
system does not assert the radiologist is wrong. It reports a discrepancy
and recommends review. The output type makes that distinction structural:
there is no "ai_correct" field and no "radiologist_wrong" field. There is
only `DiscrepancyFlag`, which can be `agreement` or `potential_discrepancy`.

Three discrepancy kinds are distinguished:

- ai_found_radiologist_did_not: the AI detected a lesion not in the
  radiologist's findings. This is the "potential miss" case.
- radiologist_found_ai_did_not: the radiologist found a lesion the AI
  did not detect. This is the AI's failure case, reported for audit.
- location_disagreement: both found lesions, but not at matching locations.

Both directions of disagreement are surfaced. Neither is asserted as
correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from imagex.temporal.matching import Lesion, match_lesions


class DiscrepancyKind(str, Enum):
    AGREEMENT = "agreement"
    AI_ONLY = "ai_found_radiologist_did_not"
    RADIOLOGIST_ONLY = "radiologist_found_ai_did_not"
    LOCATION_MISMATCH = "location_disagreement"


@dataclass(frozen=True)
class DiscrepancyFlag:
    """Result of comparing an AI read to a radiologist read."""
    kind: DiscrepancyKind
    recommendation: str             # "agreement" or "review recommended"
    ai_only_lesions: list[Lesion]
    radiologist_only_lesions: list[Lesion]
    matched_count: int


def compare_reads(
    ai_lesions: list[Lesion],
    radiologist_lesions: list[Lesion],
    distance_tolerance_mm: float = 10.0,
) -> DiscrepancyFlag:
    """Compare AI and radiologist lesion sets.

    distance_tolerance_mm: two lesions within this distance are considered
    to describe the same finding. Matches the temporal matching tolerance
    by default; callers may set it independently if the comparison task
    requires a different threshold.

    Returns a DiscrepancyFlag. Never returns a verdict on which read is
    correct, because that judgment is not this system's to make.
    """
    match = match_lesions(
        current=ai_lesions,
        prior=radiologist_lesions,
        distance_tolerance_mm=distance_tolerance_mm,
    )

    ai_only = match.current_only
    rad_only = match.prior_only
    n_matched = len(match.matched)

    if not ai_only and not rad_only:
        kind = DiscrepancyKind.AGREEMENT
        recommendation = "agreement"
    elif ai_only and not rad_only:
        kind = DiscrepancyKind.AI_ONLY
        recommendation = "review recommended"
    elif rad_only and not ai_only:
        kind = DiscrepancyKind.RADIOLOGIST_ONLY
        recommendation = "review recommended"
    else:
        kind = DiscrepancyKind.LOCATION_MISMATCH
        recommendation = "review recommended"

    return DiscrepancyFlag(
        kind=kind,
        recommendation=recommendation,
        ai_only_lesions=list(ai_only),
        radiologist_only_lesions=list(rad_only),
        matched_count=n_matched,
    )


def flag_to_dict(flag: DiscrepancyFlag) -> dict:
    """JSON-serializable view for logging and dashboard rendering."""
    return {
        "kind": flag.kind.value,
        "recommendation": flag.recommendation,
        "ai_only_lesion_ids": [l.lesion_id for l in flag.ai_only_lesions],
        "radiologist_only_lesion_ids": [
            l.lesion_id for l in flag.radiologist_only_lesions
        ],
        "matched_count": flag.matched_count,
    }
