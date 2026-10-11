"""Tests for LUNA16 series index construction."""

from pathlib import Path

import pytest

from imagex.perception.series_index import (
    build_subset_index,
    list_seriesuids,
    mhd_path_for,
)


def _make_fake_luna16(root: Path, n_per_subset: int = 2) -> None:
    for k in range(10):
        sub = root / f"subset{k}"
        sub.mkdir()
        for i in range(n_per_subset):
            (sub / f"1.3.6.1.subset{k}.series{i}.mhd").touch()


def test_build_index_counts(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=2)
    idx = build_subset_index(tmp_path)
    assert len(idx) == 20
    assert all(0 <= v <= 9 for v in idx.values())


def test_build_index_maps_correct_subset(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=1)
    idx = build_subset_index(tmp_path)
    assert idx["1.3.6.1.subset3.series0"] == 3


def test_build_index_missing_subset_raises(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path)
    (tmp_path / "subset5").rmdir()
    with pytest.raises(FileNotFoundError):
        build_subset_index(tmp_path)


def test_build_index_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        build_subset_index(tmp_path / "nope")


def test_list_seriesuids_full(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=3)
    idx = build_subset_index(tmp_path)
    all_ids = list_seriesuids(idx)
    assert len(all_ids) == 30


def test_list_seriesuids_subset(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=3)
    idx = build_subset_index(tmp_path)
    subset2 = list_seriesuids(idx, subset=2)
    assert len(subset2) == 3
    assert all(idx[s] == 2 for s in subset2)


def test_list_seriesuids_rejects_bad_subset(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=1)
    idx = build_subset_index(tmp_path)
    with pytest.raises(ValueError):
        list_seriesuids(idx, subset=10)


def test_mhd_path_for(tmp_path: Path) -> None:
    p = mhd_path_for(tmp_path, "abc", 4)
    assert p == tmp_path / "subset4" / "abc.mhd"


def test_build_index_rejects_duplicate_across_subsets(tmp_path: Path) -> None:
    _make_fake_luna16(tmp_path, n_per_subset=1)
    # Same seriesuid planted in two subsets
    (tmp_path / "subset9" / "dup.mhd").touch()
    (tmp_path / "subset8" / "dup.mhd").touch()
    with pytest.raises(ValueError):
        build_subset_index(tmp_path)
