"""Uncertainty estimation via Monte Carlo dropout.

Runs inference N times with dropout layers active and measures the
dispersion of predictions. High dispersion indicates the model is
sensitive to its own parameters for this input, which is a proxy for
epistemic uncertainty.

This is a practical approximation, not a formal Bayesian treatment.
The correctness requirement is empirical: uncertainty estimates must
correlate with error rate. That property is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class UncertaintyResult:
    mean_probability: np.ndarray  # shape (N,)
    std_probability: np.ndarray   # shape (N,)
    predictive_entropy: np.ndarray  # shape (N,)
    mutual_information: np.ndarray  # shape (N,)
    n_samples: int


def _enable_dropout(model: nn.Module) -> None:
    """Put model in eval mode but re-enable dropout layers."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy of a probability array. Numerically safe."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    n_samples: int = 30,
) -> UncertaintyResult:
    """Run MC dropout inference. Returns per-sample uncertainty statistics.

    inputs: (N, ...) tensor. A batch of N inputs.
    n_samples: number of stochastic forward passes per input.

    mean_probability: average softmax probability of class 1 across passes.
    std_probability: standard deviation of that probability across passes.
    predictive_entropy: entropy of the mean probability.
    mutual_information: predictive entropy minus mean of per-pass entropies.
        This is the epistemic component: how much does the model's own
        uncertainty about its parameters contribute to the prediction.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if inputs.ndim < 2:
        raise ValueError(f"inputs must have at least 2 dims, got {inputs.shape}")

    _enable_dropout(model)

    all_probs: list[np.ndarray] = []
    for _ in range(n_samples):
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())

    stacked = np.stack(all_probs, axis=0)  # (n_samples, N)
    mean_p = stacked.mean(axis=0)
    std_p = stacked.std(axis=0)

    pred_entropy = _entropy(mean_p)
    per_pass_entropy = _entropy(stacked).mean(axis=0)
    mutual_info = pred_entropy - per_pass_entropy
    # Numerical safety: MI should be non-negative in expectation but can dip.
    mutual_info = np.maximum(mutual_info, 0.0)

    return UncertaintyResult(
        mean_probability=mean_p,
        std_probability=std_p,
        predictive_entropy=pred_entropy,
        mutual_information=mutual_info,
        n_samples=n_samples,
    )


def uncertainty_error_correlation(
    uncertainty: np.ndarray,
    correct: np.ndarray,
) -> float:
    """Spearman-style rank correlation between uncertainty and error.

    Here, `correct` is 1 if the prediction was correct. A model whose
    uncertainty tracks its errors will have uncertainty negatively
    correlated with correctness: high uncertainty on wrong answers.

    Returns a value in [-1, 1]. Negative is expected and desired.
    Pearson correlation of ranks is used to avoid requiring scipy.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if uncertainty.shape != correct.shape:
        raise ValueError("shape mismatch")
    if uncertainty.size < 2:
        raise ValueError("need at least 2 samples")

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks

    ru = _rank(uncertainty)
    rc = _rank(correct)
    ru = ru - ru.mean()
    rc = rc - rc.mean()
    denom = np.sqrt((ru * ru).sum() * (rc * rc).sum())
    if denom == 0.0:
        return 0.0
    return float((ru * rc).sum() / denom)
