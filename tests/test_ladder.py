"""Tests for the baseline ladder runner. Synthetic volumes only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from imagex.perception.ladder import run_baseline, run_to_dict


class TinyVolumeModel(nn.Module):
    """Small model over a 3D volume. Fast enough for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def _write_volumes(tmp_path: Path, n: int, shape=(2, 8, 8)) -> tuple[dict, dict]:
    paths: dict[str, Path] = {}
    labels: dict[str, int] = {}
    for i in range(n):
        s = f"series_{i:03d}"
        p = tmp_path / f"{s}.npy"
        label = i % 2
        arr = np.full(shape, float(label), dtype=np.float32)
        np.save(p, arr)
        paths[s] = p
        labels[s] = label
    return paths, labels


def test_run_baseline_completes(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    run = run_baseline(
        name="tiny",
        model=TinyVolumeModel(),
        volume_paths=paths,
        labels=labels,
        epochs=2,
        batch_size=4,
        seed=42,
    )
    assert run.n_train + run.n_val + run.n_test == 30
    assert 0.0 <= run.test.accuracy <= 1.0
    assert run.test.n_samples == run.n_test


def test_run_baseline_is_deterministic(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    a = run_baseline("tiny", TinyVolumeModel(), paths, labels, epochs=2, seed=42)
    b = run_baseline("tiny", TinyVolumeModel(), paths, labels, epochs=2, seed=42)
    assert a.test.accuracy == pytest.approx(b.test.accuracy)
    assert a.final_train.loss == pytest.approx(b.final_train.loss)


def test_run_baseline_rejects_too_few(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=2)
    with pytest.raises(ValueError):
        run_baseline("tiny", TinyVolumeModel(), paths, labels)


def test_run_to_dict_is_json_serializable(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    run = run_baseline("tiny", TinyVolumeModel(), paths, labels, epochs=1, seed=42)
    d = run_to_dict(run)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["name"] == "tiny"
    assert "test" in parsed and "accuracy" in parsed["test"]
