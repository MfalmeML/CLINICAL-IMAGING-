"""Tests for LUNA16 CSV parsing."""

from pathlib import Path

import pytest

from imagex.perception.luna16 import (
    load_annotations,
    load_candidates,
    series_candidate_counts,
    series_with_nodules,
)

ANNOT_HEADER = "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
CAND_HEADER = "seriesuid,coordX,coordY,coordZ,class\n"


def test_load_annotations_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "annotations.csv"
    p.write_text(ANNOT_HEADER + "abc,-1.0,-2.0,-3.0,5.5\n" + "abc,-4.0,-5.0,-6.0,7.2\n")
    anns = load_annotations(p)
    assert len(anns) == 2
    assert anns[0].seriesuid == "abc"
    assert anns[0].diameter_mm == 5.5
    assert anns[1].coord_z == -6.0


def test_load_annotations_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path / "nope.csv")


def test_load_annotations_missing_column(tmp_path: Path) -> None:
    p = tmp_path / "annotations.csv"
    p.write_text("seriesuid,coordX\nabc,-1.0\n")
    with pytest.raises(ValueError):
        load_annotations(p)


def test_load_candidates_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "candidates.csv"
    p.write_text(CAND_HEADER + "abc,-1.0,-2.0,-3.0,1\n" + "abc,-4.0,-5.0,-6.0,0\n")
    cands = load_candidates(p)
    assert len(cands) == 2
    assert cands[0].label == 1
    assert cands[1].label == 0


def test_load_candidates_rejects_bad_class(tmp_path: Path) -> None:
    p = tmp_path / "candidates.csv"
    p.write_text(CAND_HEADER + "abc,-1.0,-2.0,-3.0,2\n")
    with pytest.raises(ValueError):
        load_candidates(p)


def test_series_with_nodules(tmp_path: Path) -> None:
    p = tmp_path / "annotations.csv"
    p.write_text(ANNOT_HEADER + "a,0,0,0,1\n" + "a,1,1,1,2\n" + "b,0,0,0,1\n")
    anns = load_annotations(p)
    assert series_with_nodules(anns) == {"a", "b"}


def test_series_candidate_counts(tmp_path: Path) -> None:
    p = tmp_path / "candidates.csv"
    p.write_text(CAND_HEADER + "a,0,0,0,1\n" + "a,1,1,1,0\n" + "b,0,0,0,1\n")
    cands = load_candidates(p)
    counts = series_candidate_counts(cands)
    assert counts == {"a": 2, "b": 1}
