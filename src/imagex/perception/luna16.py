"""LUNA16 annotation and candidate parsing.

Reads annotations.csv (ground-truth nodules) and candidates.csv
(candidate locations with class labels). Produces per-series label records
the dataset layer and baseline ladder can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NoduleAnnotation:
    seriesuid: str
    coord_x: float
    coord_y: float
    coord_z: float
    diameter_mm: float


@dataclass(frozen=True)
class Candidate:
    seriesuid: str
    coord_x: float
    coord_y: float
    coord_z: float
    label: int  # 1 = nodule, 0 = non-nodule


def load_annotations(path: Path) -> list[NoduleAnnotation]:
    """Parse annotations.csv. Raises on missing columns or missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"annotations file not found: {path}")

    df = pd.read_csv(path)
    required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"annotations missing columns: {sorted(missing)}")

    return [
        NoduleAnnotation(
            seriesuid=str(row.seriesuid),
            coord_x=float(row.coordX),
            coord_y=float(row.coordY),
            coord_z=float(row.coordZ),
            diameter_mm=float(row.diameter_mm),
        )
        for row in df.itertuples(index=False)
    ]


def load_candidates(path: Path) -> list[Candidate]:
    """Parse candidates.csv. Raises on missing columns or missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"candidates file not found: {path}")

    df = pd.read_csv(path)
    required = {"seriesuid", "coordX", "coordY", "coordZ", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")

    labels = df["class"].unique()
    if not set(labels).issubset({0, 1}):
        raise ValueError(f"candidates class must be 0 or 1, got {sorted(labels)}")

    return [
        Candidate(
            seriesuid=str(row.seriesuid),
            coord_x=float(row.coordX),
            coord_y=float(row.coordY),
            coord_z=float(row.coordZ),
            label=int(row._4),  # 'class' is a reserved word; positional access
        )
        for row in df.itertuples(index=False)
    ]


def series_with_nodules(annotations: list[NoduleAnnotation]) -> set[str]:
    """Return the set of seriesuids that contain at least one annotated nodule."""
    return {a.seriesuid for a in annotations}


def series_candidate_counts(candidates: list[Candidate]) -> dict[str, int]:
    """Return per-series candidate count."""
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.seriesuid] = counts.get(c.seriesuid, 0) + 1
    return counts
