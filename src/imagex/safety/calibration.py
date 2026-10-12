"""Temperature scaling for post-hoc confidence calibration.

Fits a single scalar T that minimizes NLL of softmax(logits / T).
T > 1 softens overconfident predictions; T < 1 sharpens them.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureScaler:
    """Fitted temperature. Divide logits by `temperature` before softmax."""
    temperature: float

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-scaled softmax probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D, got {logits.shape}")
        z = logits / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    probs = _softmax(logits / T)
    n = len(labels)
    eps = 1e-12
    return -np.log(probs[np.arange(n), labels] + eps).mean()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
) -> TemperatureScaler:
    """Fit a scalar temperature on held-out logits via grid + golden search.

    A coarse-to-fine 1D search is robust and dependency-free. We avoid
    scipy.optimize to keep the module import-light.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must have same N: {logits.shape[0]} vs {labels.shape[0]}"
        )
    if logits.shape[0] == 0:
        raise ValueError("empty logits")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    # Coarse log-spaced grid, then refine around the best point.
    grid = np.exp(np.linspace(np.log(0.1), np.log(10.0), num=max(10, max_iter)))
    losses = np.array([_nll(logits, labels, float(T)) for T in grid])
    best_T = float(grid[int(np.argmin(losses))])

    # Refine with a local finer grid around best_T.
    lo, hi = best_T / 2.0, best_T * 2.0
    for _ in range(3):
        local = np.linspace(lo, hi, num=max(10, max_iter // 2 or 10))
        local_losses = np.array([_nll(logits, labels, float(T)) for T in local])
        best_T = float(local[int(np.argmin(local_losses))])
        span = (hi - lo) / 4.0
        lo, hi = max(1e-3, best_T - span), best_T + span

    return TemperatureScaler(temperature=best_T)