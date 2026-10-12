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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# These are sibling classes, not subclasses of nn.Dropout, so each is listed.
_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@dataclass(frozen=True)
class UncertaintyResult:
    mean_probability: np.ndarray  # shape (N,)
    std_probability: np.ndarray   # shape (N,)
    predictive_entropy: np.ndarray  # shape (N,)
    mutual_information: np.ndarray  # shape (N,)
    n_samples: int


@contextmanager
def _dropout_active(model: nn.Module) -> Iterator[None]:
    """Eval mode with only the dropout layers switched on.

    Every module's original train/eval flag is restored on exit, so the model
    is never left in a different mode than it was found in.
    """
    saved = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, _DROPOUT_TYPES):
                module.train()
        yield
    finally:
        for module, was_training in saved:
            module.training = was_training


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

    The model must output logits of shape (N, 2); class 1 is the positive class.
    The model's train/eval mode is the same after the call as before it.

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

    all_probs: list[np.ndarray] = []
    with _dropout_active(model):
        for _ in range(n_samples):
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(
                    f"expected logits of shape (N, 2), got {tuple(logits.shape)}"
                )
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


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks starting at 1. Tied values share the mean of their ranks."""
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)  # rank of the last member of each tie group
    return (last - (counts - 1) / 2.0)[inverse]


def uncertainty_error_correlation(
    uncertainty: np.ndarray,
    correct: np.ndarray,
) -> float:
    """Spearman rank correlation between uncertainty and correctness.

    Here, `correct` is 1 if the prediction was correct. A model whose
    uncertainty tracks its errors will have uncertainty negatively
    correlated with correctness: high uncertainty on wrong answers.

    Returns a value in [-1, 1]. Negative is expected and desired.
    Tied values get average ranks, so the result never depends on the order
    of the samples. If either input is constant the correlation is undefined
    and 0.0 is returned. Pearson correlation of ranks is used to avoid
    requiring scipy.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if uncertainty.ndim != 1 or uncertainty.shape != correct.shape:
        raise ValueError(
            f"expected matching 1-D arrays, got {uncertainty.shape} and {correct.shape}"
        )
    if uncertainty.size < 2:
        raise ValueError("need at least 2 samples")
    if not (np.isfinite(uncertainty).all() and np.isfinite(correct).all()):
        raise ValueError("inputs must be finite")

    ru = _average_ranks(uncertainty)
    rc = _average_ranks(correct)
    ru = ru - ru.mean()
    rc = rc - rc.mean()
    denom = np.sqrt((ru * ru).sum() * (rc * rc).sum())
    if denom == 0.0:
        return 0.0
    return float((ru * rc).sum() / denom)