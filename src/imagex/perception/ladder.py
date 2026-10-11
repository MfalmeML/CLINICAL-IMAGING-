"""Baseline ladder runner. Produces metrics required by the build plan.

Training loop is delegated to train.py. This module handles splitting,
orchestration, and evaluation-metric aggregation. Accuracy is reported
for continuity but is not the decision metric; AUROC, AUPRC, sensitivity,
and specificity at a fixed threshold are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imagex.perception.dataset import VolumeDataset
from imagex.perception.metrics import BinaryMetrics, binary_metrics
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
    test_loss: float
    test_accuracy: float
    test_binary: BinaryMetrics


def _make_loader(
    paths: list[Path],
    labels: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = VolumeDataset(paths, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run model over loader. Returns (scores, labels, mean_loss, accuracy).

    scores are softmax probabilities of the positive class.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)

        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += bs

        all_scores.append(probs[:, 1].detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    mean_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return scores, labels, mean_loss, accuracy


def run_baseline(
    name: str,
    model: nn.Module,
    volume_paths: dict[str, Path],
    labels: dict[str, int],
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    threshold: float = 0.5,
    device: str = "cpu",
) -> LadderRun:
    """Train and evaluate one baseline. Deterministic given seed."""
    common = sorted(set(volume_paths) & set(labels))
    if len(common) < 3:
        raise ValueError(f"need at least 3 series, got {len(common)}")

    split = deterministic_holdout(common, seed=seed)

    if not split.train or not split.val or not split.test:
        raise ValueError(
            f"empty partition: train={len(split.train)} "
            f"val={len(split.val)} test={len(split.test)}"
        )

    train_paths = [volume_paths[s] for s in split.train]
    train_labels = [labels[s] for s in split.train]
    val_paths = [volume_paths[s] for s in split.val]
    val_labels = [labels[s] for s in split.val]
    test_paths = [volume_paths[s] for s in split.test]
    test_labels = [labels[s] for s in split.test]

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

    scores, test_labels_arr, test_loss, test_accuracy = _collect_scores(
        model, test_loader, dev
    )
    test_binary = binary_metrics(scores, test_labels_arr, threshold=threshold)

    return LadderRun(
        name=name,
        n_train=len(train_paths),
        n_val=len(val_paths),
        n_test=len(test_paths),
        final_train=final_train,
        final_val=final_val,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        test_binary=test_binary,
    )


def run_to_dict(run: LadderRun) -> dict[str, Any]:
    """JSON-serializable view of a LadderRun for tracking."""
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
            "loss": run.test_loss,
            "accuracy": run.test_accuracy,
            "sensitivity": run.test_binary.sensitivity,
            "specificity": run.test_binary.specificity,
            "auroc": run.test_binary.auroc,
            "auprc": run.test_binary.auprc,
            "n_pos": run.test_binary.n_pos,
            "n_neg": run.test_binary.n_neg,
            "threshold": run.test_binary.threshold,
        },
    }
