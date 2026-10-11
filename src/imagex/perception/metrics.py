"""Evaluation metrics for the perception baseline ladder.

Accuracy is deliberately absent. It is misleading on imbalanced detection
data and the build plan names FROC, AUROC, AUPRC, sensitivity, and
specificity as the required metrics. Only those are implemented here.

No sklearn dependency. The metrics needed are small and writing them
directly keeps the numeric behavior auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    sensitivity: float  # TP / (TP + FN)
    specificity: float  # TN / (TN + FP)
    auroc: float
    auprc: float
    n_pos: int
    n_neg: int
    threshold: float


def _binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int32)


def sensitivity_specificity(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Return (sensitivity, specificity) at a fixed threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} labels {labels.shape}")

    pred = _binarize(scores, threshold)

    tp = int(np.sum((pred == 1) & (labels == 1)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under ROC. Ranks-based; ties handled by average ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.shape != labels.shape:
        raise ValueError("shape mismatch")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Mann-Whitney U with average ranks.
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=np.float64)

    # Average ranks for ties.
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    rank_sum_pos = float(np.sum(ranks[: len(pos)]))
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under precision-recall curve via trapezoid over thresholds."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.shape != labels.shape:
        raise ValueError("shape mismatch")
    if int(np.sum(labels == 1)) == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(labels == 1)), 1)

    # Prepend (recall=0, precision=1) for the trapezoid anchor.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    sens, spec = sensitivity_specificity(scores, labels, threshold)
    return BinaryMetrics(
        sensitivity=sens,
        specificity=spec,
        auroc=auroc(scores, labels),
        auprc=auprc(scores, labels),
        n_pos=int(np.sum(np.asarray(labels) == 1)),
        n_neg=int(np.sum(np.asarray(labels) == 0)),
        threshold=threshold,
    )


def froc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    fp_counts: list[int],
) -> dict[int, float]:
    """Sensitivity at given false-positive counts per scan.

    For per-candidate scores aggregated per scan, this reduces to the
    standard FROC evaluation. Here, scores are treated as one pool;
    the `fp_counts` list specifies target operating points.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.shape != labels.shape:
        raise ValueError("shape mismatch")

    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return {fp: float("nan") for fp in fp_counts}

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]

    tp_cum = np.cumsum(sorted_labels == 1)
    fp_cum = np.cumsum(sorted_labels == 0)

    result: dict[int, float] = {}
    for target_fp in fp_counts:
        idx = np.searchsorted(fp_cum, target_fp, side="right") - 1
        if idx < 0:
            result[target_fp] = 0.0
        else:
            result[target_fp] = float(tp_cum[idx]) / n_pos
    return result
