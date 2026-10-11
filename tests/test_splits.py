"""Tests for deterministic split logic."""

import pytest

from imagex.perception.splits import deterministic_holdout, luna16_fold


def test_holdout_is_deterministic() -> None:
    ids = [f"series_{i}" for i in range(1000)]
    a = deterministic_holdout(ids, seed=42)
    b = deterministic_holdout(ids, seed=42)
    assert a == b


def test_holdout_partitions_all_ids() -> None:
    ids = [f"series_{i}" for i in range(1000)]
    s = deterministic_holdout(ids, seed=42)
    combined = set(s.train) | set(s.val) | set(s.test)
    assert combined == set(ids)
    assert len(s.train) + len(s.val) + len(s.test) == len(ids)


def test_holdout_fractions_approximate() -> None:
    ids = [f"series_{i}" for i in range(10000)]
    s = deterministic_holdout(ids, val_fraction=0.15, test_fraction=0.15, seed=42)
    assert 0.13 <= len(s.val) / 10000 <= 0.17
    assert 0.13 <= len(s.test) / 10000 <= 0.17
    assert 0.66 <= len(s.train) / 10000 <= 0.74


def test_holdout_different_seeds_give_different_splits() -> None:
    ids = [f"series_{i}" for i in range(1000)]
    a = deterministic_holdout(ids, seed=42)
    b = deterministic_holdout(ids, seed=43)
    assert a.test != b.test


def test_holdout_rejects_bad_fractions() -> None:
    ids = ["a", "b"]
    with pytest.raises(ValueError):
        deterministic_holdout(ids, val_fraction=0.0)
    with pytest.raises(ValueError):
        deterministic_holdout(ids, val_fraction=0.6, test_fraction=0.6)


def test_holdout_rejects_empty() -> None:
    with pytest.raises(ValueError):
        deterministic_holdout([])


def test_luna16_fold_partitions() -> None:
    ids = [f"s{i}" for i in range(20)]
    subset_of = {s: i % 10 for i, s in enumerate(ids)}
    fold = luna16_fold(ids, held_out_subset=3, subset_of=subset_of)
    assert set(fold.train) | set(fold.val) | set(fold.test) == set(ids)
    assert len(fold.test) == 2
    assert len(fold.val) == 2
    assert len(fold.train) == 16


def test_luna16_fold_rejects_bad_subset() -> None:
    with pytest.raises(ValueError):
        luna16_fold(["a"], held_out_subset=10, subset_of={"a": 0})


def test_luna16_fold_missing_subset_raises() -> None:
    with pytest.raises(ValueError):
        luna16_fold(["a", "b"], held_out_subset=0, subset_of={"a": 0})
