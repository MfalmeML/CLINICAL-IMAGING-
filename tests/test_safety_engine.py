"""Tests for the safety engine orchestrator."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from imagex.safety.calibration import fit_temperature
from imagex.safety.engine import (
    SafetyOutput,
    evaluate_safety,
    safety_output_to_dict,
)
from imagex.safety.ood import fit_mahalanobis, threshold_at_percentile


class SafeModel(nn.Module):
    """Model with dropout and a classifier head."""

    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.drop = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        return self.classifier(h)


def _setup(tmp_path_unused=None):
    torch.manual_seed(0)
    model = SafeModel()

    # Fitting data for temperature and OOD.
    fit_x = torch.randn(100, 4) * 0.5
    with torch.no_grad():
        fit_logits = model(fit_x).numpy()
    fit_labels = (fit_logits[:, 1] > 0).astype(np.int64)
    scaler = fit_temperature(fit_logits, fit_labels, max_iter=20)

    ood_ref = fit_mahalanobis(model, fit_x)
    in_scores = np.array([0.0])  # placeholder replaced below
    from imagex.safety.ood import mahalanobis_score
    in_scores = mahalanobis_score(model, fit_x, ood_ref)
    threshold = threshold_at_percentile(in_scores, percentile=95.0)

    return model, scaler, ood_ref, threshold


def test_evaluate_safety_returns_one_per_input() -> None:
    model, scaler, ood_ref, threshold = _setup()
    x = torch.randn(5, 4)
    outputs = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=threshold,
        uncertainty_threshold=1.0,  # high enough that nothing flags on uncertainty
        mc_samples=10,
    )
    assert len(outputs) == 5
    for o in outputs:
        assert isinstance(o, SafetyOutput)
        assert o.prediction in (0, 1)
        assert 0.0 <= o.confidence <= 1.0
        assert o.uncertainty >= 0.0
        assert o.recommendation in ("safe-to-predict", "human-review")


def test_high_uncertainty_forces_review() -> None:
    model, scaler, ood_ref, threshold = _setup()
    x = torch.randn(4, 4)
    outputs = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=threshold,
        uncertainty_threshold=-1.0,  # any uncertainty above -1 → review
        mc_samples=10,
    )
    assert all(o.recommendation == "human-review" for o in outputs)


def test_low_uncertainty_and_in_distribution_no_review() -> None:
    model, scaler, ood_ref, threshold = _setup()
    # Draw inputs from the same distribution as the fitting data.
    x = torch.randn(4, 4) * 0.5
    outputs = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=threshold,
        uncertainty_threshold=1e6,  # nothing flags on uncertainty
        mc_samples=10,
    )
    # All should be safe-to-predict since uncertainty threshold is very high
    # and inputs come from in-distribution.
    assert all(o.recommendation == "safe-to-predict" for o in outputs)


def test_ood_forces_review() -> None:
    model, scaler, ood_ref, threshold = _setup()
    # Draw inputs from far outside the fitting distribution.
    x = torch.randn(4, 4) * 10.0 + 5.0
    outputs = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=threshold,
        uncertainty_threshold=1e6,
        mc_samples=10,
    )
    # With inputs this far off, at least one should be flagged as OOD.
    assert any(o.is_ood for o in outputs)
    assert any(o.recommendation == "human-review" for o in outputs)


def test_safety_output_to_dict_is_json_serializable() -> None:
    import json
    model, scaler, ood_ref, threshold = _setup()
    x = torch.randn(3, 4)
    outputs = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=threshold,
        uncertainty_threshold=1.0,
        mc_samples=5,
    )
    d = safety_output_to_dict(outputs[0])
    s = json.dumps(d)
    parsed = json.loads(s)
    assert "recommendation" in parsed
    assert parsed["recommendation"] in ("safe-to-predict", "human-review")


def test_evaluate_safety_rejects_few_mc_samples() -> None:
    model, scaler, ood_ref, threshold = _setup()
    with pytest.raises(ValueError):
        evaluate_safety(
            model=model,
            inputs=torch.randn(2, 4),
            temperature_scaler=scaler,
            ood_reference=ood_ref,
            ood_threshold=threshold,
            uncertainty_threshold=1.0,
            mc_samples=1,
        )


def test_evaluate_safety_rejects_1d_input() -> None:
    model, scaler, ood_ref, threshold = _setup()
    with pytest.raises(ValueError):
        evaluate_safety(
            model=model,
            inputs=torch.randn(4),
            temperature_scaler=scaler,
            ood_reference=ood_ref,
            ood_threshold=threshold,
            uncertainty_threshold=1.0,
            mc_samples=5,
        )
