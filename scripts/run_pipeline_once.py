"""Run the preprocessing pipeline on one downloaded CT series.

Usage (from the repo root):
    python scripts/run_pipeline_once.py [series_dir]
With no argument, uses the first series folder under data/raw/LIDC-IDRI.
"""

import sys
from pathlib import Path

from imagex.preprocessing.pipeline import preprocess_study

RAW = Path("data/raw/LIDC-IDRI")
CONFIG = Path("configs/preprocessing.yaml")


def main() -> None:
    if len(sys.argv) > 1:
        series_dir = Path(sys.argv[1])
    else:
        series_dirs = sorted(p for p in RAW.iterdir() if p.is_dir())
        if not series_dirs:
            raise SystemExit(f"No series folders in {RAW}")
        series_dir = series_dirs[0]

    print("series:", series_dir.name)
    out = preprocess_study(series_dir, CONFIG)
    vol = out["volume"]
    print("volume shape:", vol.shape)
    print("dtype:", vol.dtype)
    print("value range:", float(vol.min()), "to", float(vol.max()))
    print("spacing:", out["spacing"])


if __name__ == "__main__":
    main()
