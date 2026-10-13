"""Dataset wrapper for preprocessed CT volumes and lesion labels.

Consumes the output contract of preprocess_study: a float32 (z, y, x) volume
in [0, 1]. Labels are supplied separately for now; LIDC-IDRI parsing is a
later step.

Depth normalization:
    Raw CT volumes have variable z depth (LUNA16 ranges from ~100 to ~400
    slices). 3D CNNs need a fixed depth to batch, so each volume is
    center-cropped (if too deep) or symmetrically zero-padded (if too
    shallow) to TARGET_DEPTH before tensorization. Cropping/padding is
    applied only along the depth axis; in-plane (y, x) is left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed depth every volume is resampled to. Chosen to fit within LUNA16's
# minimum native depth so cropping (rather than padding) dominates.
TARGET_DEPTH: int = 128


def fix_depth(
    arr: np.ndarray,
    target: int = TARGET_DEPTH,
) -> np.ndarray:
    """Center-crop or symmetric zero-pad the depth (axis 0) to `target`.

    arr must be 3D (z, y, x). Returns a new array of shape (target, y, x).
    """
    if arr.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {arr.shape}")
    if target < 1:
        raise ValueError(f"target depth must be >= 1, got {target}")

    z = arr.shape[0]
    if z == target:
        return arr

    if z > target:
        start = (z - target) // 2
        return arr[start : start + target]

    # z < target → symmetric zero-pad along depth.
    pad_total = target - z
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_width = [(pad_top, pad_bottom), (0, 0), (0, 0)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


class VolumeDataset(Dataset):
    """Maps volume paths to (tensor, label) pairs.

    Each item is a .npy file holding a float32 array of shape (z, y, x).
    The optional transform operates on the numpy array before tensorization.
    Depth is normalized to TARGET_DEPTH after the transform.
    """

    def __init__(
        self,
        volume_paths: list[Path],
        labels: list[int],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        target_depth: int = TARGET_DEPTH,
    ) -> None:
        if len(volume_paths) != len(labels):
            raise ValueError(
                f"volume_paths ({len(volume_paths)}) and labels ({len(labels)}) "
                "must be the same length"
            )
        if not volume_paths:
            raise ValueError("empty dataset")
        if target_depth < 1:
            raise ValueError(f"target_depth must be >= 1, got {target_depth}")

        self.volume_paths = [Path(p) for p in volume_paths]
        self.labels = [int(v) for v in labels]
        self.transform = transform
        self.target_depth = target_depth

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

        # Normalize depth so every sample in a batch stacks cleanly.
        arr = fix_depth(arr, target=self.target_depth)

        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        # Add channel dimension: (z, y, x) -> (1, z, y, x)
        tensor = tensor.unsqueeze(0)
        return tensor, self.labels[idx]