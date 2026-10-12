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


class Dropout3dModel(nn.Module):
    """Volumetric model whose only dropout is Dropout3d."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.drop = nn.Dropout3d(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.drop(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


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


def test_mc_dropout_activates_dropout3d() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 1, 4, 4, 4)
    r = mc_dropout_predict(Dropout3dModel(), x, n_samples=20)
    assert r.std_probability.max() > 0.0


def test_mc_dropout_restores_training_mode() -> None:
    model = DropoutModel()
    x = torch.randn(4, 4)
    model.train()
    mc_dropout_predict(model, x, n_samples=3)
    assert model.training and model.drop.training
    model.eval()
    mc_dropout_predict(model, x, n_samples=3)
    assert not model.training and not model.drop.training


def test_mc_dropout_rejects_non_binary_output() -> None:
    class ThreeClass(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    with pytest.raises(ValueError):
        mc_dropout_predict(ThreeClass(), torch.randn(2, 4), n_samples=3)


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


def test_uncertainty_error_correlation_known_value() -> None:
    # Ranks of uncertainty are 1..4; the two wrong answers are the most uncertain.
    uncertainty = np.array([1.0, 2.0, 3.0, 4.0])
    correct = np.array([1, 1, 0, 0])
    rho = uncertainty_error_correlation(uncertainty, correct)
    assert rho == pytest.approx(-2 / np.sqrt(5))


def test_uncertainty_error_correlation_zero_when_constant() -> None:
    uncertainty = np.array([0.5] * 10)
    correct = np.array([1, 0] * 5)
    rho = uncertainty_error_correlation(uncertainty, correct)
    assert rho == 0.0


def test_uncertainty_error_correlation_ties_do_not_depend_on_order() -> None:
    uncertainty = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 3], dtype=float)
    correct = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0])
    perm = np.array([9, 3, 5, 0, 7, 1, 8, 2, 6, 4])
    a = uncertainty_error_correlation(uncertainty, correct)
    b = uncertainty_error_correlation(uncertainty[perm], correct[perm])
    assert a == pytest.approx(b)


def test_uncertainty_error_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        uncertainty_error_correlation(np.array([0.5]), np.array([1, 0]))


def test_uncertainty_error_correlation_rejects_too_few() -> None:
    with pytest.raises(ValueError):
        uncertainty_error_correlation(np.array([0.5]), np.array([1]))


def test_uncertainty_error_correlation_rejects_nan() -> None:
    with pytest.raises(ValueError):
        uncertainty_error_correlation(
            np.array([0.1, np.nan, 0.3]), np.array([1, 0, 1])
        )