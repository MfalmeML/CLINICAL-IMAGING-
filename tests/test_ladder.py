"""Tests for the baseline ladder runner. Synthetic volumes only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from imagex.perception.ladder import run_baseline, run_to_dict


class TinyVolumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def _make_model() -> TinyVolumeModel:
    """Seeded construction, so two calls give identical initial weights."""
    torch.manual_seed(0)
    return TinyVolumeModel()


def _write_volumes(tmp_path: Path, n: int, shape=(2, 8, 8)) -> tuple[dict, dict]:
    paths: dict = {}
    labels: dict = {}
    for i in range(n):
        s = f"series_{i:03d}"
        p = tmp_path / f"{s}.npy"
        np.save(p, np.full(shape, float(i % 2), dtype=np.float32))
        paths[s] = p
        labels[s] = i % 2
    return paths, labels


def test_run_baseline_completes(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    run = run_baseline(
        name="tiny",
        model=_make_model(),
        volume_paths=paths,
        labels=labels,
        epochs=2,
        batch_size=4,
        seed=42,
    )
    assert run.n_train + run.n_val + run.n_test == 30
    assert run.test_binary.n_pos + run.test_binary.n_neg == run.n_test
    assert run.test_binary.n_pos > 0 and run.test_binary.n_neg > 0
    assert 0.0 <= run.test_accuracy <= 1.0
    assert 0.0 <= run.test_binary.sensitivity <= 1.0
    assert 0.0 <= run.test_binary.specificity <= 1.0
    assert 0.0 <= run.test_binary.auroc <= 1.0
    assert 0.0 <= run.test_binary.auprc <= 1.0


def test_run_baseline_is_deterministic(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    a = run_baseline("tiny", _make_model(), paths, labels, epochs=2, seed=42)
    b = run_baseline("tiny", _make_model(), paths, labels, epochs=2, seed=42)
    # Both classes must be present, otherwise AUROC is NaN and compares as unequal.
    assert a.test_binary.n_pos > 0 and a.test_binary.n_neg > 0
    assert a.test_loss == pytest.approx(b.test_loss)
    assert a.test_accuracy == pytest.approx(b.test_accuracy)
    assert a.test_binary.auroc == pytest.approx(b.test_binary.auroc)
    assert a.test_binary.auprc == pytest.approx(b.test_binary.auprc)


def test_rare_positives_still_reach_the_test_split(tmp_path: Path) -> None:
    paths, _ = _write_volumes(tmp_path, n=40)
    labels = {s: int(i % 7 == 0) for i, s in enumerate(sorted(paths))}  # 6 positives
    run = run_baseline("tiny", _make_model(), paths, labels, epochs=1, seed=42)
    assert run.test_binary.n_pos >= 1
    assert run.test_binary.n_neg >= 1


def test_run_baseline_rejects_single_class(tmp_path: Path) -> None:
    paths, _ = _write_volumes(tmp_path, n=30)
    labels = {s: 0 for s in paths}
    with pytest.raises(ValueError, match="both classes"):
        run_baseline("tiny", _make_model(), paths, labels)


def test_run_baseline_rejects_too_few(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=2)
    with pytest.raises(ValueError):
        run_baseline("tiny", _make_model(), paths, labels)


def test_run_to_dict_is_json_serializable(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    run = run_baseline("tiny", _make_model(), paths, labels, epochs=1, seed=42)
    d = run_to_dict(run)
    s = json.dumps(d, allow_nan=False)  # strict JSON: NaN would raise here
    parsed = json.loads(s)
    assert parsed["name"] == "tiny"
    for key in ("auroc", "auprc", "sensitivity", "specificity", "threshold"):
        assert key in parsed["test"]
