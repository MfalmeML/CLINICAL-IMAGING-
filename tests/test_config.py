"""Tests for config loading."""

from pathlib import Path

import pytest

from imagex.config import load_config


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_valid_config_roundtrips(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(good)
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2


def test_real_preprocessing_config_loads() -> None:
    cfg = load_config(Path("configs/preprocessing.yaml"))
    assert cfg["geometry"]["target_spacing_mm"] == [1.0, 1.0, 1.0]
    assert cfg["intensity"]["hu_min"] == -1000
