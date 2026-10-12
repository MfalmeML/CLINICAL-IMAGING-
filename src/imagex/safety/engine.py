"""Safety engine: orchestrates calibration, MC-dropout uncertainty, and OOD.

The engine answers one question per input: should this prediction be
trusted, or routed to human review?

Decision rule (per sample):
    human-review if  (uncertainty > uncertainty_threshold)  OR  is_ood
    otherwise        safe-to-predict
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from imagex.safety.calibration import TemperatureScaler
from imagex.safety.ood import MahalanobisReference, mahalanobis_score

Recommendation = Literal["safe-to-predict", "human-review"]


@dataclass(frozen=True)
class SafetyOutput:
    """Per-sample safety decision."""
    prediction: int
    confidence: float
    uncertainty: float
    is_ood: bool
    ood_score: float
    recommendation: Recommendation


def _enable_dropout(model: nn.Module) -> None:
    """Put model in eval mode but re-enable dropout layers for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _mc_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `mc_samples` stochastic forwards. Returns (mean_probs, std_probs).

    Shapes: (N, C) each.
    """
    _enable_dropout(model)
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    stacked = np.stack(probs_list, axis=0)          # (S, N, C)
    return stacked.mean(axis=0), stacked.std(axis=0)


def evaluate_safety(
    model: nn.Module,
    inputs: torch.Tensor,
    temperature_scaler: TemperatureScaler,
    ood_reference: MahalanobisReference,
    ood_threshold: float,
    uncertainty_threshold: float,
    mc_samples: int = 10,
) -> list[SafetyOutput]:
    """Run calibrated MC-dropout inference + OOD check on each input.

    Raises:
        ValueError: if inputs are not 2D, or mc_samples < 2.
    """
    if not isinstance(inputs, torch.Tensor):
        raise ValueError(f"inputs must be a torch.Tensor, got {type(inputs)}")
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2D, got shape {tuple(inputs.shape)}")
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2, got {mc_samples}")
    if inputs.shape[0] == 0:
        return []

    # 1. MC-dropout: mean and std of softmax probabilities.
    mean_probs, std_probs = _mc_forward(model, inputs, mc_samples)

    # 2. Uncertainty: mean per-sample std across classes.
    #    (Total predictive uncertainty — a scalar per sample.)
    uncertainty = std_probs.mean(axis=1)            # (N,)

    # 3. Calibrated confidence: apply temperature to raw logits, then take
    #    the max class probability. We reuse the temperature on the mean
    #    probabilities' log — but cleaner: re-run one deterministic forward
    #    to get logits, then scale.
    model.eval()
    with torch.no_grad():
        raw_logits = model(inputs).cpu().numpy()
    calibrated_probs = temperature_scaler.apply(raw_logits)   # (N, C)
    prediction = calibrated_probs.argmax(axis=1).astype(np.int64)
    confidence = calibrated_probs.max(axis=1)

    # 4. OOD scores via the fitted Mahalanobis reference.
    ood_scores = mahalanobis_score(model, inputs, ood_reference)
    is_ood = ood_scores > ood_threshold

    # 5. Assemble outputs.
    outputs: list[SafetyOutput] = []
    for i in range(inputs.shape[0]):
        review = bool(is_ood[i]) or bool(uncertainty[i] > uncertainty_threshold)
        outputs.append(
            SafetyOutput(
                prediction=int(prediction[i]),
                confidence=float(confidence[i]),
                uncertainty=float(uncertainty[i]),
                is_ood=bool(is_ood[i]),
                ood_score=float(ood_scores[i]),
                recommendation="human-review" if review else "safe-to-predict",
            )
        )
    return outputs


def safety_output_to_dict(output: SafetyOutput) -> dict:
    """Return a JSON-serializable dict representation."""
    return asdict(output)