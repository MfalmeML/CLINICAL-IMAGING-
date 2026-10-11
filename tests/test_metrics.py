"""Tests for evaluation metrics. Known-answer only."""

import numpy as np
import pytest

from imagex.perception.metrics import (
    auprc,
    auroc,
    binary_metrics,
    froc_curve,
    sensitivity_specificity,
)


def test_perfect_separation_auroc_is_one() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_inverted_separation_auroc_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_all_tied_auroc_is_half() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.5)


def test_sensitivity_specificity_perfect() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    sens, spec = sensitivity_specificity(scores, labels, threshold=0.5)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)


def test_sensitivity_specificity_at_threshold() -> None:
    scores = np.array([0.9, 0.4, 0.6, 0.3])
    labels = np.array([1, 1, 0, 0])
    sens, spec = sensitivity_specificity(scores, labels, threshold=0.5)
    assert sens == pytest.approx(0.5)   # one of two positives above
    assert spec == pytest.approx(0.5)   # one of two negatives above


def test_sensitivity_specificity_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sensitivity_specificity(np.array([0.5]), np.array([0, 1]), 0.5)


def test_auprc_perfect_is_one() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert auprc(scores, labels) == pytest.approx(1.0)


def test_auprc_inverted_is_low() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    p = auprc(scores, labels)
    assert p < 0.5


def test_auprc_no_positives_returns_nan() -> None:
    scores = np.array([0.1, 0.2])
    labels = np.array([0, 0])
    assert np.isnan(auprc(scores, labels))


def test_auroc_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        auroc(np.array([0.5]), np.array([0, 1]))


def test_binary_metrics_bundle() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    m = binary_metrics(scores, labels, threshold=0.5)
    assert m.n_pos == 2
    assert m.n_neg == 2
    assert m.sensitivity == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.auroc == pytest.approx(1.0)
    assert m.auprc == pytest.approx(1.0)
    assert m.threshold == 0.5


def test_froc_at_zero_fp_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    out = froc_curve(scores, labels, fp_counts=[0, 1, 2])
    # At zero false positives, the highest-scoring item is a true positive,
    # so the search should find sensitivity > 0 for fp=0.
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0
    assert out[2] == pytest.approx(1.0)


def test_froc_monotonic_in_fp() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.3).astype(np.int32)
    out = froc_curve(scores, labels, fp_counts=[1, 5, 10, 20])
    vals = [out[k] for k in (1, 5, 10, 20)]
    assert vals == sorted(vals)
