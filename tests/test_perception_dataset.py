"""Tests for the perception dataset wrapper."""

from pathlib import Path

import numpy as np
import pytest
import torch

from imagex.perception.dataset import VolumeDataset


def _write_volume(path: Path, shape: tuple[int, int, int] = (4, 8, 8)) -> None:
    np.save(path, np.random.rand(*shape).astype(np.float32))


def test_length_matches_inputs(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"vol_{i}.npy"
        _write_volume(p)
        paths.append(p)
    ds = VolumeDataset(paths, [0, 1, 0])
    assert len(ds) == 3


def test_item_shape_and_dtype(tmp_path: Path) -> None:
    p = tmp_path / "vol.npy"
    _write_volume(p, shape=(4, 8, 8))
    ds = VolumeDataset([p], [1])
    tensor, label = ds[0]
    assert tensor.shape == (1, 4, 8, 8)
    assert tensor.dtype == torch.float32
    assert label == 1


def test_mismatched_lengths_raise(tmp_path: Path) -> None:
    p = tmp_path / "vol.npy"
    _write_volume(p)
    with pytest.raises(ValueError):
        VolumeDataset([p], [0, 1])


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError):
        VolumeDataset([], [])


def test_missing_file_raises(tmp_path: Path) -> None:
    ds = VolumeDataset([tmp_path / "missing.npy"], [0])
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_non_3d_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((4, 4), dtype=np.float32))
    ds = VolumeDataset([p], [0])
    with pytest.raises(ValueError):
        _ = ds[0]


def test_transform_applied(tmp_path: Path) -> None:
    p = tmp_path / "vol.npy"
    _write_volume(p, shape=(4, 8, 8))

    def halve(arr: np.ndarray) -> np.ndarray:
        return arr * 0.5

    ds = VolumeDataset([p], [0], transform=halve)
    tensor, _ = ds[0]
    assert float(tensor.max()) <= 0.5
