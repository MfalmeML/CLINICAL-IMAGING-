"""Out-of-distribution detection.

Two methods behind a common interface:

- Energy score on logits. Free, no fitting required, works out of the box.
- Mahalanobis distance on penultimate features. Requires fitting a
  feature distribution on training data; more powerful when fitted.

The correctness requirement is empirical: OOD scores must be higher on
out-of-distribution inputs than on in-distribution inputs. Both methods
are tested for that property on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MahalanobisReference:
    """Feature statistics fitted on training data."""
    mean: np.ndarray          # shape (D,)
    precision: np.ndarray     # shape (D, D), inverse of covariance
    feature_dim: int


@dataclass(frozen=True)
class OODResult:
    scores: np.ndarray        # shape (N,); higher means more OOD
    threshold: float
    is_ood: np.ndarray        # shape (N,), boolean


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Free-energy score of logits. Higher energy means more in-distribution.

    The standard formulation returns negative log-sum-exp, so lower values
    indicate OOD. This function returns the negative of that, so higher
    values indicate OOD, matching the convention of the rest of the module.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits.shape}")
    # logsumexp along classes
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    # Higher for in-distribution → negate to match OOD convention.
    return -lse


def _penultimate_features(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """Extract the input to the final linear layer.

    Assumes the final classification head is a Linear module named
    `classifier` or is the last Linear encountered. Raises if neither
    is available.
    """
    model.eval()
    feats: list[np.ndarray] = []

    def _hook(_module, inp, _out):
        feats.append(inp[0].detach().cpu().numpy())

    target: nn.Module | None = None
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Linear):
        target = getattr(model, "classifier")
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        target = getattr(model, "head")
    if target is None:
        raise AttributeError(
            "model has no `classifier` or `head` Linear layer; cannot extract features"
        )

    handle = target.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()

    if not feats:
        raise RuntimeError("feature hook did not fire")
    return feats[0]


def fit_mahalanobis(
    model: nn.Module,
    inputs: torch.Tensor,
    regularization: float = 1e-3,
) -> MahalanobisReference:
    """Fit feature mean and precision on in-distribution training inputs.

    regularization: added to the diagonal of the covariance before inverting.
    Prevents singular matrices when features are collinear.
    """
    if regularization <= 0.0:
        raise ValueError("regularization must be > 0")

    feats = _penultimate_features(model, inputs)
    if feats.ndim != 2:
        raise ValueError(f"features must be 2D, got {feats.shape}")
    if feats.shape[0] < 2:
        raise ValueError("need at least 2 samples to fit covariance")

    mean = feats.mean(axis=0)
    centered = feats - mean
    cov = (centered.T @ centered) / (feats.shape[0] - 1)
    cov_reg = cov + regularization * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov_reg)
    return MahalanobisReference(
        mean=mean,
        precision=precision,
        feature_dim=feats.shape[1],
    )


def mahalanobis_score(
    model: nn.Module,
    inputs: torch.Tensor,
    reference: MahalanobisReference,
) -> np.ndarray:
    """Per-sample Mahalanobis distance from the fitted feature distribution."""
    feats = _penultimate_features(model, inputs)
    if feats.shape[1] != reference.feature_dim:
        raise ValueError(
            f"feature dim mismatch: {feats.shape[1]} vs {reference.feature_dim}"
        )
    diff = feats - reference.mean
    # Vectorized quadratic form: row-wise (x - mu)^T P (x - mu)
    scores = np.einsum("ij,jk,ik->i", diff, reference.precision, diff)
    return np.sqrt(np.maximum(scores, 0.0))


def threshold_at_percentile(
    in_distribution_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Set an OOD threshold at a percentile of in-distribution scores.

    A 95th-percentile threshold flags roughly 5 percent of in-distribution
    inputs as OOD. That is the acceptable false-flag rate. Adjust per the
    clinical tolerance for false alerts.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    scores = np.asarray(in_distribution_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("empty scores")
    return float(np.percentile(scores, percentile))


def apply_ood_threshold(
    scores: np.ndarray,
    threshold: float,
) -> OODResult:
    """Flag samples whose OOD score exceeds the threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    return OODResult(
        scores=scores,
        threshold=float(threshold),
        is_ood=scores > threshold,
    )


def ood_detection_auroc(
    in_distribution_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """AUROC of an OOD score. 1.0 is perfect, 0.5 is chance.

    Positive class is OOD. Scores are assumed higher-is-more-OOD.
    """
    in_scores = np.asarray(in_distribution_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    if in_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("both score sets must be non-empty")

    # Mann-Whitney U with average ranks, same logic as metrics.auroc.
    all_scores = np.concatenate([in_scores, ood_scores])
    labels = np.concatenate(
        [np.zeros(in_scores.size, dtype=np.int32), np.ones(ood_scores.size, dtype=np.int32)]
    )
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=np.float64)

    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    pos_mask = labels == 1
    rank_sum_pos = float(ranks[pos_mask].sum())
    n_pos = int(pos_mask.sum())
    n_neg = len(labels) - n_pos
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
