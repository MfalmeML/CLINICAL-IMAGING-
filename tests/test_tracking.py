"""Tests for MLflow tracking wrapper. Uses a temp SQLite DB."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from imagex.perception.ladder import run_baseline
from imagex.perception.tracking import log_ladder_run, sqlite_tracking_uri


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


def _write_volumes(tmp_path: Path, n: int) -> tuple[dict, dict]:
    paths: dict = {}
    labels: dict = {}
    for i in range(n):
        s = f"series_{i:03d}"
        p = tmp_path / f"{s}.npy"
        np.save(p, np.full((2, 8, 8), float(i % 2), dtype=np.float32))
        paths[s] = p
        labels[s] = i % 2
    return paths, labels


def test_log_ladder_run_returns_id(tmp_path: Path) -> None:
    paths, labels = _write_volumes(tmp_path, n=30)
    run = run_baseline("baseline-test", TinyVolumeModel(), paths, labels, epochs=1, seed=42)

    uri = sqlite_tracking_uri(tmp_path / "mlflow.db")
    run_id = log_ladder_run(
        run,
        params={"architecture": "tiny", "epochs": 1, "seed": 42},
        tracking_uri=uri,
    )
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_sqlite_uri_creates_parent(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "mlflow.db"
    uri = sqlite_tracking_uri(db)
    assert uri.startswith("sqlite:///")
    assert db.parent.is_dir()
