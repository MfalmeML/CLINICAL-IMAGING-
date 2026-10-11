"""Train/validation/test split logic.

Determinism is the entire point of this module. Every baseline in the
ladder must see the same held-out set, or the ablation table is invalid.

LUNA16 ships 10 cross-validation subsets (subset0 .. subset9). This module
supports both k-fold style splitting across those subsets and a single
deterministic holdout for quick iteration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FoldSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def _hash_seriesid(seriesuid: str, seed: int) -> int:
    """Stable integer hash of a seriesuid. Python's hash() is not stable."""
    digest = hashlib.sha256(f"{seed}:{seriesuid}".encode()).hexdigest()
    return int(digest[:16], 16)


def deterministic_holdout(
    seriesuids: list[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> FoldSplit:
    """Split seriesuids into train/val/test by stable hash.

    Fractions are approximate. Each seriesid lands in exactly one bucket.
    Re-running with the same inputs and seed yields the same split.
    """
    if not seriesuids:
        raise ValueError("empty seriesuid list")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    # Bucket by hash modulo 10000, compare against thresholds.
    val_cut = int(val_fraction * 10000)
    test_cut = val_cut + int(test_fraction * 10000)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for s in seriesuids:
        b = _hash_seriesid(s, seed) % 10000
        if b < val_cut:
            val.append(s)
        elif b < test_cut:
            test.append(s)
        else:
            train.append(s)

    train.sort()
    val.sort()
    test.sort()
    return FoldSplit(train=train, val=val, test=test)


def luna16_fold(seriesuids: list[str], held_out_subset: int, subset_of: dict[str, int]) -> FoldSplit:
    """LUNA16 cross-validation split: one subset is test, one is val, rest train.

    `subset_of` maps each seriesuid to its subset index (0..9).
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
