"""Calibration: does predicted confidence match observed correctness.

Implements Expected Calibration Error (ECE) and temperature scaling.

ECE is a diagnostic. Temperature scaling is the correction. The
correction is fit on validation data and applied to test data without
re-fitting. Fitting on test data would invalidate the calibration claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ECEResult:
    ece: float
    bin_counts: list[int]
    bin_confidences: list[float]
    bin_accuracies: list[float]
    n_bins: int


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float64))


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> ECEResult:
    """Compute ECE over equal-width confidence bins.

    confidences: predicted probability of the chosen class, shape (N,).
    correct: 1 if the prediction was correct, 0 otherwise, shape (N,).
    n_bins: number of confidence bins in (0, 1].
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.int32)
    if conf.shape != corr.shape:
        raise ValueError(f"shape mismatch: conf {conf.shape} corr {corr.shape}")
    if conf.size == 0:
        raise ValueError("empty input")
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        raise ValueError("confidences must be in [0, 1]")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts: list[int] = []
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    ece = 0.0
    n = conf.size

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf > lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(corr[mask].mean())
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        ece += (count / n) * abs(avg_acc - avg_conf)

    return ECEResult(
        ece=ece,
        bin_counts=bin_counts,
        bin_confidences=bin_conf,
        bin_accuracies=bin_acc,
        n_bins=n_bins,
    )


class TemperatureScaler(nn.Module):
    """Single-scalar temperature on logits. Fitted on validation data."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        if init_temperature <= 0.0:
            raise ValueError("initial temperature must be > 0")
        # Parameterize log(T) so T stays positive without clamping.
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(init_temperature)), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fit a scalar temperature by minimizing NLL on validation logits.

    logits: shape (N, C), raw model output before softmax.
    labels: shape (N,), integer class indices.
    """
    logits_t = _to_tensor(logits).float()
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    if logits_t.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits_t.shape}")
    if logits_t.size(0) != labels_t.size(0):
        raise ValueError("logits and labels must have the same length")

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(
        [scaler.log_temperature], lr=lr, max_iter=max_iter
    )
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    scaler.eval()
    return scaler


def apply_temperature(
    logits: np.ndarray,
    scaler: TemperatureScaler,
) -> np.ndarray:
    """Apply a fitted temperature to logits. Returns scaled logits."""
    with torch.no_grad():
        scaled = scaler(_to_tensor(logits).float())
    return scaled.cpu().numpy()


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a 2D logits array."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
