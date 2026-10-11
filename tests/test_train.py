"""Tests for the training loop. Synthetic data only."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imagex.perception.baseline1_2d import Baseline1_2DCNN
from imagex.perception.dataset import VolumeDataset
from imagex.perception.train import evaluate, train_one_epoch


class TinySliceModel(nn.Module):
    """Minimal model taking a single 2D slice for fast tests."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, Z, Y, X) -> (N, 1)
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)


def _write_volume(path: Path, label: int, shape: tuple[int, int, int] = (2, 4, 4)) -> None:
    # Signal the label into the volume so the model can learn it.
    arr = np.full(shape, float(label), dtype=np.float32)
    np.save(path, arr)


def _make_loader(tmp_path: Path, n: int = 8) -> DataLoader:
    paths = []
    labels = []
    for i in range(n):
        label = i % 2
        p = tmp_path / f"vol_{i}.npy"
        _write_volume(p, label)
        paths.append(p)
        labels.append(label)
    ds = VolumeDataset(paths, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


def test_train_one_epoch_returns_metrics(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path)
    model = TinySliceModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    m = train_one_epoch(model, loader, opt, crit, torch.device("cpu"))
    assert m.n_samples == 8
    assert 0.0 <= m.accuracy <= 1.0
    assert m.loss >= 0.0


def test_evaluate_returns_metrics(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path)
    model = TinySliceModel()
    crit = nn.CrossEntropyLoss()
    m = evaluate(model, loader, crit, torch.device("cpu"))
    assert m.n_samples == 8
    assert 0.0 <= m.accuracy <= 1.0


def test_train_reduces_loss_on_separable_data(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, n=16)
    model = TinySliceModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    crit = nn.CrossEntropyLoss()

    first = train_one_epoch(model, loader, opt, crit, torch.device("cpu"))
    for _ in range(20):
        last = train_one_epoch(model, loader, opt, crit, torch.device("cpu"))

    assert last.loss <= first.loss
