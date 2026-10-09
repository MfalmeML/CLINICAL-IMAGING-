"""Download LIDC-IDRI CT series from TCIA.

Run from the repo root:  python scripts/download_lidc.py --number 2
Drop --number's default (or pass a large value) only after checking disk space.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from tcia_utils import nbia

COLLECTION = "LIDC-IDRI"
INDEX_PATH = Path("data/lidc_ct_series_index.json")
RAW_DIR = Path("data/raw/LIDC-IDRI")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--number", type=int, default=2, help="how many CT series to download"
    )
    args = parser.parse_args()

    all_series = nbia.getSeries(collection=COLLECTION)
    if not all_series:
        raise SystemExit(f"TCIA returned no series for {COLLECTION}")
    by_modality = dict(Counter(s.get("Modality") for s in all_series))
    print(f"Found {len(all_series)} series: {by_modality}")

    # The collection also holds radiographs and derived objects (SEG, SR).
    ct_series = [s for s in all_series if s.get("Modality") == "CT"]
    if not ct_series:
        raise SystemExit("No CT series in the index; check the 'Modality' field.")
    # Stable order, so "the first N" is the same N on every run.
    ct_series.sort(key=lambda s: s["SeriesInstanceUID"])
    print(f"Kept {len(ct_series)} CT series")

    # JSON, not pickle: readable, diffable, and safe to load.
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "collection": COLLECTION,
                "retrieved": date.today().isoformat(),
                "series": ct_series,
            },
            indent=2,
        )
    )

    df = nbia.downloadSeries(
        ct_series, number=args.number, path=str(RAW_DIR), format="df"
    )
    cols = [
        c for c in ("PatientID", "Modality", "SeriesInstanceUID") if c in df.columns
    ]
    print(df[cols])


if __name__ == "__main__":
    main()
