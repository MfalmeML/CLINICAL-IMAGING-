"""Stratified, deterministic train/val/test split.

Each class is split on its own, so every partition holds every class (when the
class has enough members) no matter how rare it is. Order within a class is set
by a hash of (seed, id), so the result does not depend on input order, the
Python version or the platform.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StratifiedSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def _hash_key(seed: int, series_id: str) -> str:
    return hashlib.sha256(f"{seed}:{series_id}".encode()).hexdigest()


def _count(fraction: float, n: int) -> int:
    """Number of items for a fraction of n: rounded half up, at least 1."""
    return max(1, math.floor(fraction * n + 0.5))


def stratified_holdout(
    ids: Sequence[str],
    labels: Mapping[str, Hashable],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> StratifiedSplit:
    """Split ids into train/val/test, keeping class proportions in each part.

    Every class contributes at least one item to val and to test. Raises
    ValueError if a class is too small to also leave one item for train.
    """
    ids = list(ids)
    if not ids:
        raise ValueError("ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be below 1")
    missing = [s for s in ids if s not in labels]
    if missing:
        raise ValueError(f"{len(missing)} ids have no label, e.g. {missing[0]!r}")

    by_class: dict[Hashable, list[str]] = {}
    for s in ids:
        by_class.setdefault(labels[s], []).append(s)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for cls in sorted(by_class, key=repr):
        members = sorted(by_class[cls], key=lambda s: _hash_key(seed, s))
        n = len(members)
        n_test = _count(test_fraction, n)
        n_val = _count(val_fraction, n)
        if n_test + n_val >= n:
            raise ValueError(
                f"class {cls!r} has {n} members, too few to fill train, val and test"
            )
        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])

    return StratifiedSplit(train=sorted(train), val=sorted(val), test=sorted(test))
