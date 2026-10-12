"""Temperature scaling for post-hoc confidence calibration.

Fits a single scalar T that minimizes NLL of softmax(logits / T).
T > 1 softens overconfident predictions; T < 1 sharpens them.

Public API:
    - softmax_probabilities(logits)          -> probabilities
    - TemperatureScaler(init_temperature)    -> callable scaler
    - apply_temperature(logits, scaler)      -> temperature-scaled LOGITS
    - fit_temperature(logits, labels)        -> TemperatureScaler
    - expected_calibration_error(...)        -> CalibrationReport

Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Softmax helpers
# ---------------------------------------------------------------------------

def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a 2D logits array. Rows sum to 1."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemperatureScaler:
    """Fitted temperature. `apply` returns calibrated probabilities."""
    init_temperature: float

    def __post_init__(self) -> None:
        if self.init_temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {self.init_temperature}"
            )

    @property
    def temperature(self) -> float:
        """Alias kept for callers that read `.temperature` (e.g. logging)."""
        return self.init_temperature

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-scaled SOFTMAX PROBABILITIES.

        Used by engine.py for calibrated confidence. If you want scaled
        logits instead, use the module-level `apply_temperature`.
        """
        scaled_logits = apply_temperature(logits, self)
        return softmax_probabilities(scaled_logits)


def apply_temperature(
    logits: np.ndarray,
    scaler: TemperatureScaler,
) -> np.ndarray:
    """Return temperature-scaled LOGITS: `logits / scaler.init_temperature`.

    Note: this returns logits, not probabilities. The test file
    (test_calibration.py) asserts `apply_temperature(logits, T=1) == logits`,
    which is only true if we return scaled logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    return logits / scaler.init_temperature


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    probs = softmax_probabilities(logits / T)
    n = len(labels)
    eps = 1e-12
    return -np.log(probs[np.arange(n), labels] + eps).mean()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
) -> TemperatureScaler:
    """Fit a scalar temperature on held-out logits.

    Uses a coarse-to-fine 1D search (grid then golden-style refinement)
    to stay dependency-free — no scipy.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must have the same first dimension; "
            f"got {logits.shape[0]} vs {labels.shape[0]}"
        )
    if logits.shape[0] == 0:
        raise ValueError("logits must be non-empty")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    # Coarse log-spaced grid.
    grid = np.exp(np.linspace(np.log(0.1), np.log(10.0), num=max(20, max_iter // 2)))
    losses = np.array([_nll(logits, labels, float(T)) for T in grid])
    best_T = float(grid[int(np.argmin(losses))])

    # Local refinement.
    lo, hi = max(1e-3, best_T / 2.0), best_T * 2.0
    for _ in range(3):
        local = np.linspace(lo, hi, num=max(10, max_iter // 4 or 10))
        local_losses = np.array([_nll(logits, labels, float(T)) for T in local])
        best_T = float(local[int(np.argmin(local_losses))])
        span = (hi - lo) / 4.0
        lo, hi = max(1e-3, best_T - span), best_T + span

    return TemperatureScaler(init_temperature=best_T)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationReport:
    """Result of expected_calibration_error."""
    ece: float
    bin_counts: tuple[int, ...] = field(default_factory=tuple)
    bin_confidences: tuple[float, ...] = field(default_factory=tuple)
    bin_accuracies: tuple[float, ...] = field(default_factory=tuple)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Standard ECE over equal-width confidence bins.

    confidences: per-sample max softmax prob, shape (N,), values in [0, 1].
    correct:     per-sample 0/1 indicator of whether argmax was right.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    if confidences.ndim != 1 or correct.ndim != 1:
        raise ValueError("confidences and correct must both be 1D")
    if confidences.shape[0] != correct.shape[0]:
        raise ValueError(
            f"length mismatch: {confidences.shape[0]} vs {correct.shape[0]}"
        )
    if confidences.size == 0:
        raise ValueError("confidences must be non-empty")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidences must be in [0, 1]")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, edges[1:-1]), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    conf_sums = np.zeros(n_bins, dtype=np.float64)
    acc_sums = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            conf_sums[b] = confidences[mask].mean()
            acc_sums[b] = correct[mask].mean()

    total = confidences.shape[0]
    ece = 0.0
    for b in range(n_bins):
        if counts[b] > 0:
            ece += (counts[b] / total) * abs(acc_sums[b] - conf_sums[b])

    return CalibrationReport(
        ece=float(ece),
        bin_counts=tuple(int(c) for c in counts),
        bin_confidences=tuple(float(c) for c in conf_sums),
        bin_accuracies=tuple(float(a) for a in acc_sums),
    )