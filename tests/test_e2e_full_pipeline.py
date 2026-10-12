"""End-to-end test across every engine in the system.

Exercises the full chain with synthetic data. Catches interface drift
between modules: if a module's output shape or field names change,
this test fails and the mismatch is localized to a known integration
point.
"""

import json

import numpy as np
import torch
import torch.nn as nn

from imagex.pipeline import (
    PerceptionOutput,
    pipeline_result_to_dict,
    run_pipeline,
)
from imagex.safety.calibration import fit_temperature
from imagex.safety.engine import evaluate_safety, safety_output_to_dict
from imagex.safety.ood import fit_mahalanobis, mahalanobis_score, threshold_at_percentile
from imagex.temporal.change import compute_change
from imagex.temporal.matching import Lesion, match_lesions
from imagex.temporal.registration import register_rigid
from imagex.triage.priority import Priority


class TinySegModel(nn.Module):
    """A model with dropout and a classifier head, sufficient to exercise
    calibration, uncertainty, and OOD detection."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.drop = nn.Dropout(p=0.2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.drop(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


def _make_blob(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0) -> np.ndarray:
    z, y, x = np.indices(shape)
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    return np.exp(-(dist ** 2) / (2 * radius ** 2)).astype(np.float32)


def _lesion(lid: str, x: float, y: float, z: float, d: float) -> Lesion:
    return Lesion(
        lesion_id=lid,
        centroid_mm=(x, y, z),
        volume_mm3=(4.0 / 3.0) * 3.14159 * (d / 2) ** 3,
        diameter_mm=d,
    )


def test_full_pipeline_with_all_engines(tmp_path) -> None:
    torch.manual_seed(0)
    model = TinySegModel()

    # Fit safety layer.
    fit_inputs = torch.randn(60, 1, 16, 32, 32) * 0.1
    with torch.no_grad():
        fit_logits = model(fit_inputs).numpy()
    fit_labels = (fit_logits[:, 1] > 0).astype(np.int64)
    scaler = fit_temperature(fit_logits, fit_labels, max_iter=10)

    ood_ref = fit_mahalanobis(model, fit_inputs)
    in_scores = mahalanobis_score(model, fit_inputs, ood_ref)
    ood_threshold = threshold_at_percentile(in_scores, percentile=95.0)

    # Registration between prior and current.
    prior_vol = _make_blob(center=(8, 16, 16))
    current_vol = _make_blob(center=(8, 18, 16))
    reg = register_rigid(
        fixed=current_vol,
        moving=prior_vol,
        spacing_xyz=(1.0, 1.0, 1.0),
        number_of_iterations=10,
    )
    assert reg.transform_type == "rigid"

    # Lesion matching and change.
    prior_lesions = [_lesion("p1", 16.0, 16.0, 8.0, d=5.0)]
    current_lesions = [_lesion("c1", 16.0, 16.0, 8.0, d=7.0)]
    match = match_lesions(current_lesions, prior_lesions, distance_tolerance_mm=10.0)
    changes = compute_change(match, interval_days=180.0)
    assert len(changes) == 1
    assert changes[0].change_class == "growing"

    # Safety evaluation on the current volume.
    current_input = torch.from_numpy(current_vol).unsqueeze(0).unsqueeze(0)
    safety_out = evaluate_safety(
        model=model,
        inputs=current_input,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=ood_threshold,
        uncertainty_threshold=1e6,  # nothing flags on uncertainty for this test
        mc_samples=5,
    )
    assert len(safety_out) == 1

    # Pipeline composition.
    perception = PerceptionOutput(
        abnormality_score=0.6,
        lesions=current_lesions,
    )
    radiologist_lesions = [_lesion("r1", 17.0, 16.0, 8.0, d=6.0)]

    result = run_pipeline(
        perception=perception,
        safety_recommendation=safety_out[0].recommendation,
        uncertainty=safety_out[0].uncertainty,
        is_ood=safety_out[0].is_ood,
        temporal_changes=changes,
        radiologist_lesions=radiologist_lesions,
    )

    # The growing lesion with prior should not be LOW.
    assert result.priority in (Priority.MEDIUM, Priority.HIGH, Priority.URGENT)
    assert result.discrepancy is not None
    assert result.discrepancy.recommendation in ("agreement", "review recommended")

    # Result must be JSON-serializable end to end.
    d = pipeline_result_to_dict(result)
    s = json.dumps(d)
    assert len(s) > 0

    # Safety output must also be serializable.
    sd = safety_output_to_dict(safety_out[0])
    json.dumps(sd)


def test_full_pipeline_without_prior() -> None:
    torch.manual_seed(0)
    model = TinySegModel()

    fit_inputs = torch.randn(30, 1, 16, 32, 32) * 0.1
    with torch.no_grad():
        fit_logits = model(fit_inputs).numpy()
    fit_labels = (fit_logits[:, 1] > 0).astype(np.int64)
    scaler = fit_temperature(fit_logits, fit_labels, max_iter=5)
    ood_ref = fit_mahalanobis(model, fit_inputs)
    in_scores = mahalanobis_score(model, fit_inputs, ood_ref)
    ood_threshold = threshold_at_percentile(in_scores, percentile=95.0)

    vol = _make_blob()
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    safety_out = evaluate_safety(
        model=model,
        inputs=x,
        temperature_scaler=scaler,
        ood_reference=ood_ref,
        ood_threshold=ood_threshold,
        uncertainty_threshold=1e6,
        mc_samples=5,
    )

    perception = PerceptionOutput(abnormality_score=0.2, lesions=[])
    result = run_pipeline(
        perception=perception,
        safety_recommendation=safety_out[0].recommendation,
        uncertainty=safety_out[0].uncertainty,
        is_ood=safety_out[0].is_ood,
        temporal_changes=None,  # no prior
        radiologist_lesions=None,  # no radiologist read
    )

    assert result.temporal_changes is None
    assert result.discrepancy is None
    assert result.triage_input.has_prior is False
    json.dumps(pipeline_result_to_dict(result))
