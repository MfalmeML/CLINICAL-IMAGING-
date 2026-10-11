"""Baseline ladder runner. Synthetic-data compatible.

Trains and evaluates a baseline model on a set of preprocessed volumes.
Records metrics to a structured dict so baselines are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imagex.perception.dataset import VolumeDataset
from imagex.perception.splits import deterministic_holdout
from imagex.perception.train import EpochMetrics, evaluate, train_one_epoch


@dataclass
class LadderRun:
    name: str
    n_train: int
    n_val: int
    n_test: int
    final_train: EpochMetrics
    final_val: EpochMetrics
    test: EpochMetrics


def _make_loader(
    paths: list[Path],
    labels: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = VolumeDataset(paths, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_baseline(
    name: str,
    model: nn.Module,
    volume_paths: dict[str, Path],
    labels: dict[str, int],
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> LadderRun:
    """Train and evaluate one baseline. Deterministic given seed.

    `volume_paths` and `labels` are keyed by seriesuid. Seriesuids missing
    from either are dropped before splitting.
    """
    common = sorted(set(volume_paths) & set(labels))
    if len(common) < 3:
        raise ValueError(f"need at least 3 series with both volume and label, got {len(common)}")

    split = deterministic_holdout(common, seed=seed)

    train_paths = [volume_paths[s] for s in split.train]
    train_labels = [labels[s] for s in split.train]
    val_paths = [volume_paths[s] for s in split.val]
    val_labels = [labels[s] for s in split.val]
    test_paths = [volume_paths[s] for s in split.test]
    test_labels = [labels[s] for s in split.test]

    if not train_paths or not val_paths or not test_paths:
        raise ValueError(
            f"split produced empty partition: train={len(train_paths)} "
            f"val={len(val_paths)} test={len(test_paths)}"
        )

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)

    train_loader = _make_loader(train_paths, train_labels, batch_size, shuffle=True)
    val_loader = _make_loader(val_paths, val_labels, batch_size, shuffle=False)
    test_loader = _make_loader(test_paths, test_labels, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    final_train: EpochMetrics | None = None
    final_val: EpochMetrics | None = None

    for _ in range(epochs):
        final_train = train_one_epoch(model, train_loader, optimizer, criterion, dev)
        final_val = evaluate(model, val_loader, criterion, dev)

    assert final_train is not None and final_val is not None

    test_metrics = evaluate(model, test_loader, criterion, dev)

    return LadderRun(
        name=name,
        n_train=len(train_paths),
        n_val=len(val_paths),
        n_test=len(test_paths),
        final_train=final_train,
        final_val=final_val,
        test=test_metrics,
    )


def run_to_dict(run: LadderRun) -> dict[str, Any]:
    """Convert a LadderRun to a JSON-serializable dict for experiment tracking."""
    return {
        "name": run.name,
        "n_train": run.n_train,
        "n_val": run.n_val,
        "n_test": run.n_test,
        "final_train": {
            "loss": run.final_train.loss,
            "accuracy": run.final_train.accuracy,
            "n_samples": run.final_train.n_samples,
        },
        "final_val": {
            "loss": run.final_val.loss,
            "accuracy": run.final_val.accuracy,
            "n_samples": run.final_val.n_samples,
        },
        "test": {
            "loss": run.test.loss,
            "accuracy": run.test.accuracy,
            "n_samples": run.test.n_samples,
        },
    }
