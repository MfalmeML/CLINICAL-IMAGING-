"""Tests for the LUNA16 preparation CLI. Uses synthetic .nii.gz volumes."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from imagex.cli.prepare_luna16 import main


def _write_nii_gz(path: Path, value: float, shape: tuple[int, int, int] = (4, 8, 8)) -> None:
    arr = np.full(shape[::-1], value, dtype=np.int16)  # sitk expects (x, y, z)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))


def _make_source(root: Path) -> None:
    for k in range(2):
        sub = root / f"subset{k}"
        sub.mkdir()
        for i in range(3):
            _write_nii_gz(sub / f"1.3.6.s{k}.r{i}.nii.gz", value=200 + i)


def test_prepare_luna16_success(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    _make_source(src)
    out = tmp_path / "out"

    code = main(["--source", str(src), "--output-dir", str(out)])
    assert code == 0
    npy_files = list(out.glob("*.npy"))
    assert len(npy_files) == 6
    assert (out / "series_index.json").is_file()


def test_prepare_luna16_skips_existing(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    _make_source(src)
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "1.3.6.s0.r0.npy", np.zeros((1, 1, 1), dtype=np.float32))

    code = main(["--source", str(src), "--output-dir", str(out)])
    assert code == 0
    arr = np.load(out / "1.3.6.s0.r0.npy")
    assert arr.shape == (1, 1, 1)


def test_prepare_luna16_rejects_empty_source(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    out = tmp_path / "out"
    try:
        main(["--source", str(src), "--output-dir", str(out)])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
