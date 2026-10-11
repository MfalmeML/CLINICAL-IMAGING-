"""LUNA16 cross-validation split by subset holdout.

Complements stratified_splits.py. Both are canonical, but for different
purposes:

- stratified_splits.stratified_holdout: per-class random holdout. Used for
  training and internal validation, where every partition must contain
  every class.
- splits.luna16_fold: hold out an entire LUNA16 subset. Used for external
  validation, where the model must be evaluated on data from a different
  source distribution than it was trained on.

Do not use luna16_fold for training splits. Do not use stratified_holdout
for external validation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def luna16_fold(
    seriesuids: list[str],
    held_out_subset: int,
    subset_of: dict[str, int],
) -> FoldSplit:
    """LUNA16 cross-validation split: one subset is test, one is val, rest train.

    `subset_of` maps each seriesuid to its subset index (0..9).
    The subset immediately after `held_out_subset` (mod 10) is used for val.

    Raises ValueError if held_out_subset is not in 0..9, or if any seriesuid
    is missing from subset_of.
    """
    if held_out_subset not in range(10):
        raise ValueError(f"held_out_subset must be 0..9, got {held_out_subset}")

    val_subset = (held_out_subset + 1) % 10

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for s in seriesuids:
        if s not in subset_of:
            raise ValueError(f"seriesuid missing from subset_of map: {s}")
        k = subset_of[s]
        if k == held_out_subset:
            test.append(s)
        elif k == val_subset:
            val.append(s)
        else:
            train.append(s)

    train.sort()
    val.sort()
    test.sort()
    return FoldSplit(train=train, val=val, test=test)
