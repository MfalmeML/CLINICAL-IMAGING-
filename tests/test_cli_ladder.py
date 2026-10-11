"""Tests for the ladder CLI. Dry-run and argument parsing only."""

from pathlib import Path

import numpy as np
import pytest

from imagex.cli.run_ladder import main


def _write_fixture(tmp_path: Path, n_series: int = 8) -> tuple[Path, Path]:
    volumes_dir = tmp_path / "volumes"
    volumes_dir.mkdir()
    lines = ["seriesuid,coordX,coordY,coordZ,diameter_mm"]

    for i in range(n_series):
        s = f"series_{i:03d}"
        np.save(volumes_dir / f"{s}.npy", np.zeros((2, 8, 8), dtype=np.float32))
        # Half the series have an annotated nodule.
        if i % 2 == 0:
            lines.append(f"{s},0.0,0.0,0.0,5.0")

    annotations = tmp_path / "annotations.csv"
    annotations.write_text("\n".join(lines) + "\n")
    return volumes_dir, annotations


def test_dry_run_succeeds(tmp_path: Path) -> None:
    volumes_dir, annotations = _write_fixture(tmp_path)
    code = main(
        [
            "--baseline", "baseline1-2d-cnn",
            "--volumes-dir", str(volumes_dir),
            "--annotations", str(annotations),
            "--dry-run",
        ]
    )
    assert code == 0


def test_rejects_too_few_positives(tmp_path: Path) -> None:
    volumes_dir = tmp_path / "volumes"
    volumes_dir.mkdir()
    lines = ["seriesuid,coordX,coordY,coordZ,diameter_mm"]
    for i in range(8):
        s = f"series_{i:03d}"
        np.save(volumes_dir / f"{s}.npy", np.zeros((2, 8, 8), dtype=np.float32))
        if i == 0:
            lines.append(f"{s},0.0,0.0,0.0,5.0")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("\n".join(lines) + "\n")

    code = main(
        [
            "--baseline", "baseline1-2d-cnn",
            "--volumes-dir", str(volumes_dir),
            "--annotations", str(annotations),
            "--dry-run",
        ]
    )
    assert code == 2


def test_rejects_unknown_baseline(tmp_path: Path) -> None:
    volumes_dir, annotations = _write_fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--baseline", "nonexistent",
                "--volumes-dir", str(volumes_dir),
                "--annotations", str(annotations),
                "--dry-run",
            ]
        )


def test_rejects_missing_volumes_dir(tmp_path: Path) -> None:
    volumes_dir, annotations = _write_fixture(tmp_path)
    with pytest.raises(NotADirectoryError):
        main(
            [
                "--baseline", "baseline1-2d-cnn",
                "--volumes-dir", str(tmp_path / "does_not_exist"),
                "--annotations", str(annotations),
                "--dry-run",
            ]
        )
