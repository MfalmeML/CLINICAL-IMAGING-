"""Dataset wrapper for preprocessed CT volumes and lesion labels.

Consumes the output contract of preprocess_study: a float32 (z, y, x) volume
in [0, 1]. Labels are supplied separately for now; LIDC-IDRI parsing is a
later step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    """Maps volume paths to (tensor, label) pairs.

    Each item is a .npy file holding a float32 array of shape (z, y, x).
    The optional transform operates on the numpy array before tensorization.
    """

    def __init__(
        self,
        volume_paths: list[Path],
        labels: list[int],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if len(volume_paths) != len(labels):
            raise ValueError(
                f"volume_paths ({len(volume_paths)}) and labels ({len(labels)}) "
                "must be the same length"
            )
        if not volume_paths:
            raise ValueError("empty dataset")

        self.volume_paths = [Path(p) for p in volume_paths]
        self.labels = [int(v) for v in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.volume_paths[idx]
        if not path.is_file():
            raise FileNotFoundError(f"volume not found: {path}")

        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"expected 3D volume at {path}, got shape {arr.shape}")

        if self.transform is not None:
            arr = self.transform(arr)

        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        # Add channel dimension: (z, y, x) -> (1, z, y, x)
        tensor = tensor.unsqueeze(0)
        return tensor, self.labels[idx]
