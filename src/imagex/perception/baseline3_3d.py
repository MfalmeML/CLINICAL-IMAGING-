"""Baseline 3: 3D CNN. Full volumetric context.

Input is the entire preprocessed volume as a single-channel 3D tensor.
This is the first rung of the ladder that sees the whole study, not a
slice or a slice stack. Its purpose is to isolate the value of volumetric
context against Baselines 1 and 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Baseline3_3DCNN(nn.Module):
    """3D CNN classifier over a full volume.

    Input: (N, 1, Z, Y, X).
    Output: (N, 2) logits over {normal, abnormal}.

    Deliberately shallow. The point of the baseline ladder is to compare
    the effect of input representation, not to chase capacity.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5D input (N, C, Z, Y, X), got {x.shape}")
        feats = self.features(x)
        feats = feats.flatten(1)
        return self.classifier(feats)
