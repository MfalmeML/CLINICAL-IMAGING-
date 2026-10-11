"""Tests for MC dropout uncertainty estimation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from imagex.safety.uncertainty import (
    mc_dropout_predict,
    uncertainty_error_correlation,
)


class DropoutModel(nn.Module):
    """Model with dropout so MC sampling produces variance."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)


def test_mc_dropout_returns_expected_shapes() -> None:
    model = DropoutModel()
    x = torch.randn(8, 4)
    r = mc_dropout_predict(model, x, n_samples=10)
    assert r.mean_probability.shape == (8,)
    assert r.std_probability.shape == (8,)
    assert r.predictive_entropy.shape == (8,)
    assert r.mutual_information.shape == (8,)
    assert r.n_samples == 10


def test_mc_dropout_probabilities_in_range() -> None:
    model = DropoutModel()
    x = torch.randn(8, 4)
    r = mc_dropout_predict(model, x, n_samples=5)
    assert np.all(r.mean_probability >= 0.0)
    assert np.all(r.mean_probability <= 1.0)
    assert np.all(r.std_probability >= 0.0)


def test_mc_dropout_with_zero_dropout_has_zero_std() -> None:
    class NoDropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = NoDropout()
    x = torch.randn(4, 4)
    r = mc_dropout_predict(model, x, n_samples=5)
    assert np.allclose(r.std_probability, 0.0, atol=1e-6)


def test_mc_dropout_rejects_few_samples() -> None:
    model = DropoutModel()
    with pytest.raises(ValueError):
        mc_dropout_predict(model, torch.randn(2, 4), n_samples=1)


def test_mc_dropout_rejects_1d_input() -> None:
    model = DropoutModel()
    with pytest.raises(ValueError):
        mc_dropout_predict(model, torch.randn(4), n_samples=5)


def test_uncertainty_error_correlation_negative_when_informative() -> None:
    # Construct uncertainty that tracks error: high uncertainty on wrong.
    uncertainty = np.array([0.1, 0.2, 0.9, 1.0, 0.15, 0.95])
    correct = np.array([1, 1, 0, 0, 1, 0])
    rho = uncertainty_error_correlation(uncertainty, correct)
    assert rho < 0.0


def test_uncertainty_error_correlation_zero_when_constant() -> None:
    uncertainty = np.array([0.5] * 10)
    correct = np.array([1, 0] * 5)
    rho = uncertainty_error_correlation(uncertainty, correct)
    assert rho == 0.0


def test_uncertainty_error_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        uncertainty_error_correlation(np.array([0.5]), np.array([1, 0]))


def test_uncertainty_error_correlation_rejects_too_few() -> None:
    with pytest.raises(ValueError):
        uncertainty_error_correlation(np.array([0.5]), np.array([1]))
