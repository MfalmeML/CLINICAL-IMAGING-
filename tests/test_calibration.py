"""Tests for calibration and ECE. Known-answer only."""

import numpy as np
import pytest

from imagex.safety.calibration import (
    TemperatureScaler,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    softmax_probabilities,
)


def test_ece_perfectly_calibrated_is_zero() -> None:
    conf = np.array([0.5] * 100)
    correct = np.array([1, 0] * 50)
    r = expected_calibration_error(conf, correct, n_bins=10)
    assert r.ece == pytest.approx(0.0, abs=1e-6)


def test_ece_overconfident_is_positive() -> None:
    conf = np.array([0.9] * 100)
    correct = np.array([1, 0] * 50)
    r = expected_calibration_error(conf, correct, n_bins=10)
    assert r.ece == pytest.approx(0.4, abs=0.05)


def test_ece_rejects_empty() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([]), np.array([]))


def test_ece_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([1, 0]))


def test_ece_rejects_out_of_range_confidences() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.5]), np.array([1]))


def test_ece_bin_counts_sum_to_n() -> None:
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    correct = (rng.random(200) < conf).astype(np.int32)
    r = expected_calibration_error(conf, correct, n_bins=15)
    assert sum(r.bin_counts) == 200
    assert len(r.bin_confidences) == 15
    assert len(r.bin_accuracies) == 15


def test_temperature_scaler_identity_at_one() -> None:
    scaler = TemperatureScaler(init_temperature=1.0)
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_temperature(logits, scaler)
    assert np.allclose(out, logits, atol=1e-6)


def test_temperature_scaler_rejects_bad_init() -> None:
    with pytest.raises(ValueError):
        TemperatureScaler(init_temperature=0.0)


def test_fit_temperature_reduces_nll_on_overconfident_data() -> None:
    rng = np.random.default_rng(0)
    n = 500
    logits = rng.normal(scale=5.0, size=(n, 2)).astype(np.float32)
    labels = rng.integers(0, 2, size=n).astype(np.int64)

    scaler = fit_temperature(logits, labels, max_iter=50)

    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    raw_loss = float(criterion(torch.from_numpy(logits), torch.from_numpy(labels)).item())
    scaled = apply_temperature(logits, scaler)
    scaled_loss = float(criterion(torch.from_numpy(scaled), torch.from_numpy(labels)).item())
    assert scaled_loss <= raw_loss


def test_fit_temperature_rejects_1d() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.array([1.0, 2.0]), np.array([0, 1]))


def test_fit_temperature_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((4, 2)), np.array([0, 1]))


def test_softmax_probabilities_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
    probs = softmax_probabilities(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_stable_on_large_logits() -> None:
    logits = np.array([[1000.0, 1001.0]])
    probs = softmax_probabilities(logits)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_rejects_1d() -> None:
    with pytest.raises(ValueError):
        softmax_probabilities(np.array([1.0, 2.0]))
