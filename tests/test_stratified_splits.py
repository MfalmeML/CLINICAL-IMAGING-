"""Tests for the stratified holdout split."""

import pytest

from imagex.perception.stratified_splits import stratified_holdout


def _ids_and_labels(n_pos: int, n_neg: int) -> tuple[list[str], dict[str, int]]:
    labels = {f"p{i:03d}": 1 for i in range(n_pos)}
    labels.update({f"n{i:03d}": 0 for i in range(n_neg)})
    return list(labels), labels


def test_partitions_all_ids_without_overlap() -> None:
    ids, labels = _ids_and_labels(20, 80)
    s = stratified_holdout(ids, labels)
    combined = s.train + s.val + s.test
    assert sorted(combined) == sorted(ids)
    assert len(set(combined)) == len(ids)


def test_every_split_has_both_classes() -> None:
    ids, labels = _ids_and_labels(20, 80)
    s = stratified_holdout(ids, labels)
    for part in (s.train, s.val, s.test):
        assert {labels[i] for i in part} == {0, 1}


def test_rare_class_still_reaches_val_and_test() -> None:
    ids, labels = _ids_and_labels(5, 300)
    s = stratified_holdout(ids, labels)
    assert any(labels[i] == 1 for i in s.val)
    assert any(labels[i] == 1 for i in s.test)


def test_class_proportions_are_preserved() -> None:
    ids, labels = _ids_and_labels(20, 80)
    s = stratified_holdout(ids, labels, val_fraction=0.2, test_fraction=0.2)
    # 20 positives -> 4 per held-out split; 80 negatives -> 16 per held-out split.
    assert len(s.test) == 20
    assert sum(labels[i] for i in s.test) == 4
    assert len(s.val) == 20
    assert sum(labels[i] for i in s.val) == 4


def test_is_deterministic() -> None:
    ids, labels = _ids_and_labels(20, 80)
    assert stratified_holdout(ids, labels, seed=7) == stratified_holdout(ids, labels, seed=7)


def test_input_order_does_not_matter() -> None:
    ids, labels = _ids_and_labels(20, 80)
    assert stratified_holdout(ids, labels) == stratified_holdout(ids[::-1], labels)


def test_different_seeds_give_different_splits() -> None:
    ids, labels = _ids_and_labels(50, 50)
    a = stratified_holdout(ids, labels, seed=42)
    b = stratified_holdout(ids, labels, seed=43)
    assert a.test != b.test


def test_rejects_class_too_small_for_three_splits() -> None:
    ids, labels = _ids_and_labels(2, 50)
    with pytest.raises(ValueError, match="too few"):
        stratified_holdout(ids, labels)


def test_rejects_id_without_label() -> None:
    ids, labels = _ids_and_labels(10, 10)
    with pytest.raises(ValueError, match="no label"):
        stratified_holdout(ids + ["unlabelled"], labels)


def test_rejects_empty_ids() -> None:
    with pytest.raises(ValueError):
        stratified_holdout([], {})


def test_rejects_bad_fractions() -> None:
    ids, labels = _ids_and_labels(10, 10)
    with pytest.raises(ValueError):
        stratified_holdout(ids, labels, val_fraction=0.0)
    with pytest.raises(ValueError):
        stratified_holdout(ids, labels, val_fraction=0.6, test_fraction=0.6)


def test_rejects_duplicate_ids() -> None:
    ids, labels = _ids_and_labels(10, 10)
    with pytest.raises(ValueError, match="unique"):
        stratified_holdout(ids + [ids[0]], labels)
