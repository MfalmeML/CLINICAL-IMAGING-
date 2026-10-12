"""Tests for OOD detection methods."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from imagex.safety.ood import (
    apply_ood_threshold,
    energy_score,
    fit_mahalanobis,
    mahalanobis_score,
    ood_detection_auroc,
    threshold_at_percentile,
)


class SimpleModel(nn.Module):
    """Model with a `classifier` Linear head for feature extraction."""

    def __init__(self, in_dim: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.classifier(h)


def test_energy_score_shape_and_direction() -> None:
    logits = np.array([[1.0, 2.0], [10.0, 11.0]])
    e = energy_score(logits)
    # Row 2 has larger logits → more confident in-distribution
    # → smaller OOD score under higher-is-more-OOD convention.
    assert e.shape == (2,)
    assert e[1] < e[0]


def test_energy_score_rejects_1d() -> None:
    with pytest.raises(ValueError):
        energy_score(np.array([1.0, 2.0]))


def test_fit_mahalanobis_shapes() -> None:
    model = SimpleModel()
    x = torch.randn(50, 4)
    ref = fit_mahalanobis(model, x)
    assert ref.feature_dim == 8
    assert ref.mean.shape == (8,)
    assert ref.precision.shape == (8, 8)


def test_fit_mahalanobis_rejects_few_samples() -> None:
    model = SimpleModel()
    with pytest.raises(ValueError):
        fit_mahalanobis(model, torch.randn(1, 4))


def test_fit_mahalanobis_rejects_bad_regularization() -> None:
    model = SimpleModel()
    with pytest.raises(ValueError):
        fit_mahalanobis(model, torch.randn(20, 4), regularization=0.0)


def test_mahalanobis_score_shape() -> None:
    model = SimpleModel()
    train = torch.randn(100, 4)
    ref = fit_mahalanobis(model, train)
    scores = mahalanobis_score(model, torch.randn(10, 4), ref)
    assert scores.shape == (10,)


def test_mahalanobis_score_dim_mismatch_raises() -> None:
    """Mismatch in penultimate feature dim (controlled by `hidden`) raises."""
    model = SimpleModel(in_dim=4)
    ref = fit_mahalanobis(model, torch.randn(50, 4))
    other = SimpleModel(in_dim=4, hidden=16)   # different penultimate width
    with pytest.raises(ValueError):
        mahalanobis_score(other, torch.randn(5, 4), ref)


def test_threshold_at_percentile_monotonic() -> None:
    scores = np.arange(100, dtype=np.float64)
    t90 = threshold_at_percentile(scores, percentile=90.0)
    t50 = threshold_at_percentile(scores, percentile=50.0)
    assert t90 > t50


def test_threshold_at_percentile_rejects_bad() -> None:
    with pytest.raises(ValueError):
        threshold_at_percentile(np.array([1.0]), percentile=0.0)
    with pytest.raises(ValueError):
        threshold_at_percentile(np.array([1.0]), percentile=100.0)


def test_apply_ood_threshold_flags_above() -> None:
    scores = np.array([0.1, 0.5, 0.9, 1.5])
    r = apply_ood_threshold(scores, threshold=0.8)
    assert r.is_ood.tolist() == [False, False, True, True]
    assert r.threshold == 0.8


def test_ood_auroc_perfect_separation() -> None:
    in_scores = np.array([0.1, 0.2, 0.3])
    ood_scores = np.array([0.8, 0.9, 1.0])
    assert ood_detection_auroc(in_scores, ood_scores) == pytest.approx(1.0)


def test_ood_auroc_inverted_is_zero() -> None:
    in_scores = np.array([0.8, 0.9, 1.0])
    ood_scores = np.array([0.1, 0.2, 0.3])
    assert ood_detection_auroc(in_scores, ood_scores) == pytest.approx(0.0)


def test_ood_auroc_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ood_detection_auroc(np.array([]), np.array([0.5]))
    with pytest.raises(ValueError):
        ood_detection_auroc(np.array([0.5]), np.array([]))


def test_mahalanobis_separates_in_vs_ood_synthetic() -> None:
    # Train a model to map in-distribution inputs to a tight region.
    torch.manual_seed(0)
    model = SimpleModel(in_dim=4, hidden=8)
    train = torch.randn(200, 4) * 0.5
    ref = fit_mahalanobis(model, train)

    in_test = torch.randn(50, 4) * 0.5
    ood_test = torch.randn(50, 4) * 5.0 + 3.0

    in_scores = mahalanobis_score(model, in_test, ref)
    ood_scores = mahalanobis_score(model, ood_test, ref)
    # OOD scores should be higher on average.
    assert ood_scores.mean() > in_scores.mean()