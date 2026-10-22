"""Downsample preprocessed volumes to a training-friendly resolution.

Reads .npy volumes and writes .npy volumes at a smaller target shape.
Uses area interpolation: averaging blocks of voxels. This is not the
correct method for resampling physical spacing, but it is correct for
the specific purpose here: producing a smaller training signal while
preserving the spatial extent and the intensity range.

Physical-spacing-correct resampling is done by the preprocessing pipeline
at 1mm. This script is for training-time approximation only, and the
output directory should be considered a derived training artifact, not
a general-purpose volume representation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def downsample_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Average-pool a volume to target_shape.

    Uses integer block averaging. Requires that each source dimension is
    at least the corresponding target dimension. The output shape is
    exactly target_shape.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got {volume.shape}")
    for s, t in zip(volume.shape, target_shape):
        if s < t:
            raise ValueError(
                f"source dim {s} smaller than target dim {t} in shape {volume.shape}"
            )

    from itertools import product

    dz = volume.shape[0] // target_shape[0]
    dy = volume.shape[1] // target_shape[1]
    dx = volume.shape[2] // target_shape[2]

    # Trim to a multiple, then reshape and average.
    trimmed = volume[
        : dz * target_shape[0],
        : dy * target_shape[1],
        : dx * target_shape[2],
    ]
    reshaped = trimmed.reshape(
        target_shape[0], dz,
        target_shape[1], dy,
        target_shape[2], dx,
    )
    return reshaped.mean(axis=(1, 3, 5)).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagex-downsample")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[64, 128, 128],
                        metavar=("D", "H", "W"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        raise NotADirectoryError(f"source not found: {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = tuple(args.target_shape)
    npy_files = sorted(args.source.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"no .npy files under {args.source}")

    succeeded = 0
    skipped = 0
    failed = 0

    for src_path in npy_files:
        out_path = args.output_dir / src_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            vol = np.load(src_path)
            small = downsample_volume(vol, target)
            np.save(out_path, small)
            succeeded += 1
            if succeeded % 20 == 0:
                print(f"  {succeeded} downsampled...")
        except Exception as exc:
            failed += 1
            print(f"FAIL {src_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print()
    print(f"target shape: {target}")
    print(f"files:        {len(npy_files)}")
    print(f"succeeded:    {succeeded}")
    print(f"skipped:      {skipped}")
    print(f"failed:       {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
