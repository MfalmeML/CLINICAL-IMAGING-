"""CLI entry point: run one rung of the baseline ladder.

Usage:
    python -m imagex.cli.run_ladder --baseline baseline3-3d-cnn \
        --volumes-dir data/processed/luna16 \
        --annotations data/raw/LUNA16/annotations.csv \
        --epochs 5 --batch-size 4 --lr 1e-3 --seed 42

This is a thin wrapper. All logic lives in the perception package.
The CLI only parses arguments, wires modules together, and logs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from imagex.perception.baseline1_2d import Baseline1_2DCNN
from imagex.perception.baseline2_2p5d import Baseline2_2p5DCNN
from imagex.perception.baseline3_3d import Baseline3_3DCNN
from imagex.perception.baseline4_3d_vit import Baseline4_3DViT
from imagex.perception.ladder import run_baseline
from imagex.perception.luna16 import load_annotations, series_with_nodules
from imagex.perception.tracking import log_ladder_run, sqlite_tracking_uri


BASELINES = {
    "baseline1-2d-cnn": lambda: Baseline1_2DCNN(),
    "baseline2-2p5d-cnn": lambda: Baseline2_2p5DCNN(n_slices=3),
    "baseline3-3d-cnn": lambda: Baseline3_3DCNN(),
    "baseline4-3d-vit": lambda: Baseline4_3DViT(embed_dim=32, depth=2, num_heads=4),
}


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _discover_volumes(volumes_dir: Path) -> dict[str, Path]:
    """Map seriesuid -> .npy path for every preprocessed volume on disk."""
    if not volumes_dir.is_dir():
        raise NotADirectoryError(f"volumes directory not found: {volumes_dir}")
    out: dict[str, Path] = {}
    for p in volumes_dir.glob("*.npy"):
        out[p.stem] = p
    if not out:
        raise ValueError(f"no .npy volumes found in {volumes_dir}")
    return out


def _build_labels(
    annotations_path: Path,
    volumes: dict[str, Path],
) -> dict[str, int]:
    """1 if the seriesuid has an annotated nodule, 0 if it does not.

    Restricted to seriesuids with a preprocessed volume on disk.
    """
    annotations = load_annotations(annotations_path)
    positive = series_with_nodules(annotations)
    return {s: (1 if s in positive else 0) for s in volumes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagex-run-ladder")
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--volumes-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--tracking-db", type=Path, default=Path("experiments/mlflow.db"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate inputs, do not train.")
    args = parser.parse_args(argv)

    volumes = _discover_volumes(args.volumes_dir)
    labels = _build_labels(args.annotations, volumes)

    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    print(f"volumes found: {len(volumes)}")
    print(f"positives: {n_pos}  negatives: {n_neg}")

    if n_pos < 2 or n_neg < 2:
        print("error: need at least 2 positives and 2 negatives to run the ladder", file=sys.stderr)
        return 2

    if args.dry_run:
        print("dry run: inputs valid, not training")
        return 0

    model = BASELINES[args.baseline]()

    run = run_baseline(
        name=args.baseline,
        model=model,
        volume_paths=volumes,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        threshold=args.threshold,
        device=args.device,
    )

    params = {
        "architecture": args.baseline,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dataset": "luna16",
        "split_seed": args.seed,
        "preprocessing_config_hash": _config_hash(args.preprocessing_config),
    }

    run_id = log_ladder_run(
        run,
        params=params,
        tracking_uri=sqlite_tracking_uri(args.tracking_db),
    )

    print(f"mlflow run id: {run_id}")
    print(f"test auroc:     {run.test_binary.auroc:.4f}")
    print(f"test auprc:     {run.test_binary.auprc:.4f}")
    print(f"test sens:      {run.test_binary.sensitivity:.4f}")
    print(f"test spec:      {run.test_binary.specificity:.4f}")
    print(f"test accuracy:  {run.test_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
