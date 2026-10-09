"""Baseline 1: 2D CNN on representative axial slices. No volumetric context.

This is the floor of the baseline ladder. It exists to be beaten, not to win.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Baseline1_2DCNN(nn.Module):
    """Simple 2D CNN classifier.

    Input: (N, 1, H, W) single-channel axial slice.
    Output: (N, 2) logits over {normal, abnormal}.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input (N, C, H, W), got {x.shape}")
        feats = self.features(x)
        feats = feats.flatten(1)
        return self.classifier(feats)
