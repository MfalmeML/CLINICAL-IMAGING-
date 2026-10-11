"""Tests for LUNA16 subset-holdout split."""

import pytest

from imagex.perception.splits import luna16_fold


def test_luna16_fold_partitions() -> None:
    ids = [f"s{i}" for i in range(20)]
    subset_of = {s: i % 10 for i, s in enumerate(ids)}
    fold = luna16_fold(ids, held_out_subset=3, subset_of=subset_of)
    assert set(fold.train) | set(fold.val) | set(fold.test) == set(ids)
    assert len(fold.test) == 2
    assert len(fold.val) == 2
    assert len(fold.train) == 16


def test_luna16_fold_test_is_correct_subset() -> None:
    ids = [f"s{i}" for i in range(20)]
    subset_of = {s: i % 10 for i, s in enumerate(ids)}
    fold = luna16_fold(ids, held_out_subset=3, subset_of=subset_of)
    assert all(subset_of[s] == 3 for s in fold.test)
    assert all(subset_of[s] == 4 for s in fold.val)


def test_luna16_fold_wraps_val_to_zero() -> None:
    ids = [f"s{i}" for i in range(20)]
    subset_of = {s: i % 10 for i, s in enumerate(ids)}
    fold = luna16_fold(ids, held_out_subset=9, subset_of=subset_of)
    assert all(subset_of[s] == 9 for s in fold.test)
    assert all(subset_of[s] == 0 for s in fold.val)


def test_luna16_fold_rejects_bad_subset() -> None:
    with pytest.raises(ValueError):
        luna16_fold(["a"], held_out_subset=10, subset_of={"a": 0})


def test_luna16_fold_missing_subset_raises() -> None:
    with pytest.raises(ValueError):
        luna16_fold(["a", "b"], held_out_subset=0, subset_of={"a": 0})


def test_luna16_fold_is_deterministic() -> None:
    ids = [f"s{i}" for i in range(20)]
    subset_of = {s: i % 10 for i, s in enumerate(ids)}
    a = luna16_fold(ids, held_out_subset=3, subset_of=subset_of)
    b = luna16_fold(ids, held_out_subset=3, subset_of=subset_of)
    assert a == b
