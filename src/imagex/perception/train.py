"""Training loop for the perception baseline ladder.

Deliberately minimal: no callbacks, no schedulers, no mixed precision.
Each baseline's training is a direct call to `train_one_epoch` plus
`evaluate`, with no hidden state. This makes baseline-vs-baseline
comparison honest: same loop, same protocol, only the model differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    n_samples: int


@dataclass
class TrainHistory:
    train: list[EpochMetrics] = field(default_factory=list)
    val: list[EpochMetrics] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Single training epoch. Returns loss, accuracy, sample count."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += bs

    return EpochMetrics(
        loss=total_loss / max(1, total_samples),
        accuracy=total_correct / max(1, total_samples),
        n_samples=total_samples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluation. No gradient, no dropout, no augmentation."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += bs

    return EpochMetrics(
        loss=total_loss / max(1, total_samples),
        accuracy=total_correct / max(1, total_samples),
        n_samples=total_samples,
    )
