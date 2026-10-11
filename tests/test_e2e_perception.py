"""End-to-end perception pipeline test. Fully synthetic.

Exercises: dataset construction, splits, training loop, metrics,
MLflow logging, and the ladder run in one chain.
"""

from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn

from imagex.perception.ladder import run_baseline
from imagex.perception.tracking import log_ladder_run, sqlite_tracking_uri


class E2EModel(nn.Module):
    """Minimal 3D model sufficient for the chain to execute."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def _make_dataset(tmp_path: Path, n: int = 40) -> tuple[dict, dict]:
    paths: dict = {}
    labels: dict = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        s = f"series_{i:04d}"
        label = i % 2
        # Signal the label into the volume so a shallow model can learn it.
        arr = rng.normal(loc=float(label), scale=0.1, size=(4, 8, 8)).astype(np.float32)
        p = tmp_path / f"{s}.npy"
        np.save(p, arr)
        paths[s] = p
        labels[s] = label
    return paths, labels


def test_e2e_ladder_to_mlflow(tmp_path: Path) -> None:
    paths, labels = _make_dataset(tmp_path, n=40)

    run = run_baseline(
        name="e2e-tiny",
        model=E2EModel(),
        volume_paths=paths,
        labels=labels,
        epochs=3,
        batch_size=8,
        lr=1e-2,
        seed=42,
    )

    assert run.n_train + run.n_val + run.n_test == 40
    assert 0.0 <= run.test_accuracy <= 1.0
    # The synthetic signal is separable, so AUROC should be well above chance.
    assert run.test_binary.auroc > 0.7, f"auroc too low: {run.test_binary.auroc}"

    uri = sqlite_tracking_uri(tmp_path / "mlflow.db")
    run_id = log_ladder_run(
        run,
        params={
            "architecture": "e2e-tiny",
            "seed": 42,
            "epochs": 3,
            "batch_size": 8,
            "lr": 1e-2,
            "dataset": "synthetic",
            "split_seed": 42,
            "preprocessing_config_hash": "synthetic",
        },
        tracking_uri=uri,
    )
    assert run_id

    # Verify the run is actually retrievable from the tracking store.
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    fetched = client.get_run(run_id)
    assert fetched.data.params["architecture"] == "e2e-tiny"
    assert "test_auroc" in fetched.data.metrics
