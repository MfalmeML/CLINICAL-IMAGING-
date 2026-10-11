"""CLI entry point: preprocess a directory of DICOM studies to .npy volumes.

Usage:
    python -m imagex.cli.preprocess_dataset \
        --dicom-root data/raw/LIDC-IDRI \
        --output-dir data/processed/lidc \
        --config configs/preprocessing.yaml

Each subdirectory of --dicom-root is treated as one study. Output is one
.npy per study, named <seriesuid>.npy, where seriesuid is the directory
name. Failures are logged and skipped; a summary is printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from imagex.preprocessing.pipeline import preprocess_study


def _discover_studies(root: Path) -> list[Path]:
    """Every immediate subdirectory of root is a candidate study."""
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    studies = sorted(p for p in root.iterdir() if p.is_dir())
    if not studies:
        raise ValueError(f"no subdirectories under {root}")
    return studies


def _seriesuid_for(study_dir: Path) -> str:
    """Identifier used for the output .npy filename.

    The directory name is used. Callers are responsible for ensuring it
    is a stable, unique identifier for the study.
    """
    return study_dir.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagex-preprocess")
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    studies = _discover_studies(args.dicom_root)
    if args.limit > 0:
        studies = studies[: args.limit]

    succeeded = 0
    skipped = 0
    failed = 0

    for study in studies:
        seriesuid = _seriesuid_for(study)
        out_path = args.output_dir / f"{seriesuid}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"skip  {seriesuid}  (exists)")
            continue

        try:
            result = preprocess_study(study, args.config)
            np.save(out_path, result["volume"])
            succeeded += 1
            print(f"ok    {seriesuid}  shape={result['volume'].shape}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {seriesuid}  {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.overwrite:
                # Only show stack traces when explicitly in overwrite mode,
                # to help debugging without flooding normal runs.
                traceback.print_exc()

    print()
    print(f"studies:   {len(studies)}")
    print(f"succeeded: {succeeded}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
