"""Baseline 2: 2.5D CNN. Stacked adjacent axial slices as channels.

Each input is a fixed-size stack of K adjacent slices, treated as K channels.
Volumetric context is limited to the K-slice window. This sits between the
slice-only Baseline 1 and the fully volumetric Baseline 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Baseline2_2p5DCNN(nn.Module):
    """2.5D CNN classifier.

    Input: (N, K, H, W) where K is the number of stacked slices.
    Output: (N, 2) logits over {normal, abnormal}.
    """

    def __init__(self, n_slices: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.n_slices = n_slices

        self.features = nn.Sequential(
            nn.Conv2d(n_slices, 32, kernel_size=3, padding=1),
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
            raise ValueError(f"expected 4D input (N, K, H, W), got {x.shape}")
        if x.size(1) != self.n_slices:
            raise ValueError(
                f"expected {self.n_slices} channels, got {x.size(1)}"
            )
        feats = self.features(x)
        feats = feats.flatten(1)
        return self.classifier(feats)
