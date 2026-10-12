"""Rule-based priority assignment.

Interpretable, auditable, and deliberately not learned. The build plan
requires this ordering: a rule ships first, a learned priority model is
a later extension justified by ablation.

Inputs are the outputs of the upstream engines:

- perception: abnormality score and lesion severity
- temporal: change class per lesion, if a prior exists
- safety: recommendation, uncertainty, OOD flag

The safety recommendation has override precedence. A case flagged for
human review by the safety engine is URGENT, regardless of what the
perception or temporal signals would otherwise produce. This mirrors
the fraud platform's ring-score override pattern: the safety layer has
the final say.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


@dataclass(frozen=True)
class TriageInput:
    """Inputs from upstream engines, already computed."""
    # Perception
    abnormality_score: float         # in [0, 1], probability of any abnormal finding
    max_lesion_diameter_mm: float    # 0.0 if no lesion detected

    # Temporal, if a prior exists. None if no prior is available.
    has_prior: bool
    growth_detected: bool            # any matched lesion classified as growing
    new_lesion_detected: bool        # any new lesion relative to prior
    max_growth_rate_mm_per_day: float  # 0.0 if no growth or no prior

    # Safety
    safety_recommendation: str       # "safe-to-predict" or "human-review"
    uncertainty: float               # mutual information
    is_ood: bool


# Rule thresholds. Named, documented, and adjustable. Not tuned.
# The build plan requires the rule to be interpretable before it is
# optimized; changing these is a deliberate act, not a hyperparameter search.
ABNORMALITY_HIGH: float = 0.7
ABNORMALITY_LOW: float = 0.3
LARGE_LESION_MM: float = 10.0
SMALL_LESION_MM: float = 6.0
SIGNIFICANT_GROWTH_MM_PER_DAY: float = 0.02


def assign_priority(inp: TriageInput) -> Priority:
    """Assign a priority tier. Deterministic. No learned component."""

    # 1. Safety override: highest precedence.
    if inp.safety_recommendation == "human-review" or inp.is_ood:
        return Priority.URGENT

    # 2. New lesion with a prior: categorical escalation.
    if inp.has_prior and inp.new_lesion_detected:
        return Priority.URGENT

    # 3. Growth with a prior: escalates on rate.
    if inp.has_prior and inp.growth_detected:
        if inp.max_growth_rate_mm_per_day >= SIGNIFICANT_GROWTH_MM_PER_DAY:
            return Priority.HIGH
        return Priority.MEDIUM

    # 4. No prior or no growth: rule on perception only.
    if inp.abnormality_score >= ABNORMALITY_HIGH:
        if inp.max_lesion_diameter_mm >= LARGE_LESION_MM:
            return Priority.HIGH
        return Priority.MEDIUM

    if inp.abnormality_score >= ABNORMALITY_LOW:
        if inp.max_lesion_diameter_mm >= SMALL_LESION_MM:
            return Priority.MEDIUM
        return Priority.LOW

    return Priority.LOW


def triage_input_to_dict(inp: TriageInput) -> dict:
    """JSON-serializable view for logging."""
    return {
        "abnormality_score": inp.abnormality_score,
        "max_lesion_diameter_mm": inp.max_lesion_diameter_mm,
        "has_prior": inp.has_prior,
        "growth_detected": inp.growth_detected,
        "new_lesion_detected": inp.new_lesion_detected,
        "max_growth_rate_mm_per_day": inp.max_growth_rate_mm_per_day,
        "safety_recommendation": inp.safety_recommendation,
        "uncertainty": inp.uncertainty,
        "is_ood": inp.is_ood,
    }
