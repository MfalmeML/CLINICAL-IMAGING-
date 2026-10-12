"""Change detection and growth estimation between matched lesions.

Produces a measurement, not a verdict. Whether a growth rate is clinically
significant is a triage decision, made downstream. This module reports:

- change_class: stable, growing, shrinking, new, or resolved
- growth_rate_mm_per_interval: signed change in diameter per day
- volume_change_percent: signed percentage change in volume

Classification thresholds here are deliberately permissive. They separate
"noise-level change" from "measurable change", not "benign" from
"malignant". The clinical interpretation is not this module's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from imagex.temporal.matching import LesionMatch, MatchResult


# Below this absolute change in diameter, a matched pair is called stable.
# 1 mm is roughly the inter-observer variability for manual nodule
# measurement on CT. Changes smaller than this are not reliably distinguishable
# from measurement noise.
STABLE_THRESHOLD_MM: float = 1.0


@dataclass(frozen=True)
class LesionChange:
    """Change measurement for one matched lesion or one unmatched lesion."""
    lesion_id: str                      # id in the current study (or prior if resolved)
    prior_lesion_id: str | None         # None if new
    change_class: str                   # stable, growing, shrinking, new, resolved
    diameter_change_mm: float           # signed
    volume_change_percent: float        # signed
    growth_rate_mm_per_interval: float  # signed, per day
    interval_days: float | None         # None if no prior or unresolved interval


def _classify_matched(
    diameter_change_mm: float,
    stable_threshold_mm: float,
) -> str:
    if abs(diameter_change_mm) < stable_threshold_mm:
        return "stable"
    if diameter_change_mm > 0.0:
        return "growing"
    return "shrinking"


def _volume_change_percent(prior_volume: float, current_volume: float) -> float:
    if prior_volume <= 0.0:
        return float("nan")
    return 100.0 * (current_volume - prior_volume) / prior_volume


def compute_change(
    match: MatchResult,
    interval_days: float | None,
) -> list[LesionChange]:
    """Compute change measurements for every lesion in a match result.

    interval_days: days between the two studies. Required for matched pairs
    to compute a rate. May be None if unknown; then the rate is NaN.
    """
    if interval_days is not None and interval_days <= 0.0:
        raise ValueError(f"interval_days must be > 0, got {interval_days}")

    changes: list[LesionChange] = []

    for m in match.matched:
        diameter_delta = m.current.diameter_mm - m.prior.diameter_mm
        volume_pct = _volume_change_percent(m.prior.volume_mm3, m.current.volume_mm3)

        if interval_days is not None:
            rate = diameter_delta / interval_days
        else:
            rate = float("nan")

        changes.append(
            LesionChange(
                lesion_id=m.current.lesion_id,
                prior_lesion_id=m.prior.lesion_id,
                change_class=_classify_matched(diameter_delta, STABLE_THRESHOLD_MM),
                diameter_change_mm=diameter_delta,
                volume_change_percent=volume_pct,
                growth_rate_mm_per_interval=rate,
                interval_days=interval_days,
            )
        )

    for c in match.current_only:
        changes.append(
            LesionChange(
                lesion_id=c.lesion_id,
                prior_lesion_id=None,
                change_class="new",
                diameter_change_mm=float("nan"),
                volume_change_percent=float("nan"),
                growth_rate_mm_per_interval=float("nan"),
                interval_days=None,
            )
        )

    for p in match.prior_only:
        changes.append(
            LesionChange(
                lesion_id=p.lesion_id,
                prior_lesion_id=p.lesion_id,
                change_class="resolved",
                diameter_change_mm=float("nan"),
                volume_change_percent=float("nan"),
                growth_rate_mm_per_interval=float("nan"),
                interval_days=None,
            )
        )

    return changes


def change_to_dict(change: LesionChange) -> dict:
    """JSON-serializable view for logging and dashboard rendering.

    NaN is replaced with None because NaN is not valid JSON.
    """
    def _clean(x: float) -> float | None:
        return None if x != x else float(x)  # x != x is True only for NaN

    return {
        "lesion_id": change.lesion_id,
        "prior_lesion_id": change.prior_lesion_id,
        "change_class": change.change_class,
        "diameter_change_mm": _clean(change.diameter_change_mm),
        "volume_change_percent": _clean(change.volume_change_percent),
        "growth_rate_mm_per_interval": _clean(change.growth_rate_mm_per_interval),
        "interval_days": change.interval_days,
    }
