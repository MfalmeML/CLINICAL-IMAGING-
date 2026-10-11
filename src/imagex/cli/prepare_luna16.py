"""Convert the HF LUNA16 mirror layout into .npy volumes and a series index.

The mirror may store each series as:
  - <subset>/<seriesuid>.mhd + <seriesuid>.raw
  - <subset>/<seriesuid>.nii.gz
  - <seriesuid>.nii.gz  (flat)

This script detects the layout, reads each series, applies the same
intensity pipeline used for DICOM, and writes one .npy per seriesuid
plus a series_index.json mapping seriesuid to subset number.

It is idempotent: existing .npy files are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from imagex.preprocessing.intensity import clip_hu, normalize_minmax


def _read_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a .mhd or .nii.gz volume. Returns (array, spacing_xyz)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (x, y, z)
    return arr.astype(np.float32), (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def _series_files(root: Path) -> dict[str, tuple[Path, int]]:
    """Return {seriesuid: (path, subset_index)} for every readable series."""
    out: dict[str, tuple[Path, int]] = {}
    for k in range(10):
        sub = root / f"subset{k}"
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix == ".mhd":
                out[f.stem] = (f, k)
            elif f.suffix == ".gz" and f.name.endswith(".nii.gz"):
                stem = f.name[: -len(".nii.gz")]
                out[stem] = (f, k)
    # Flat layout fallback: root/*.nii.gz has no subset info. Skip; require subset dirs.
    if not out:
        raise ValueError(
            f"no .mhd or .nii.gz series found under any subset* directory of {root}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagex-prepare-luna16")
    parser.add_argument("--source", type=Path, required=True, help="Root containing subset0..subset9")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    series = _series_files(args.source)
    print(f"series discovered: {len(series)}")

    succeeded = 0
    skipped = 0
    failed = 0
    index: dict[str, int] = {}

    for seriesuid, (path, subset) in sorted(series.items()):
        out_path = args.output_dir / f"{seriesuid}.npy"
        index[seriesuid] = subset

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            arr, _spacing = _read_volume(path)
            clipped = clip_hu(arr, args.hu_min, args.hu_max)
            normalized = normalize_minmax(clipped, args.hu_min, args.hu_max)
            np.save(out_path, normalized.astype(np.float32))
            succeeded += 1
            if succeeded % 50 == 0:
                print(f"  {succeeded} converted...")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {seriesuid}  {type(exc).__name__}: {exc}", file=sys.stderr)

    index_path = args.output_dir / "series_index.json"
    with index_path.open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print()
    print(f"series:    {len(series)}")
    print(f"succeeded: {succeeded}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    print(f"index:     {index_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
