"""MLflow experiment tracking for the baseline ladder.

Every run of the ladder is logged with the same parameter keys and
the same metric keys, so runs are directly comparable. No ad hoc
logging, no per-baseline metric names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from imagex.perception.ladder import LadderRun, run_to_dict


DEFAULT_EXPERIMENT = "imagex-baseline-ladder"


def log_ladder_run(
    run: LadderRun,
    params: dict[str, Any],
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str:
    """Log a single LadderRun to MLflow. Returns the run id.

    `params` should carry model architecture, training config, and data
    identifiers. Keys must be stable across baselines so runs are comparable.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run.name) as active:
        for key in sorted(params):
            mlflow.log_param(key, params[key])

        d = run_to_dict(run)
        mlflow.log_metric("n_train", d["n_train"])
        mlflow.log_metric("n_val", d["n_val"])
        mlflow.log_metric("n_test", d["n_test"])

        mlflow.log_metric("train_loss", d["final_train"]["loss"])
        mlflow.log_metric("train_accuracy", d["final_train"]["accuracy"])
        mlflow.log_metric("val_loss", d["final_val"]["loss"])
        mlflow.log_metric("val_accuracy", d["final_val"]["accuracy"])
        mlflow.log_metric("test_loss", d["test"]["loss"])
        mlflow.log_metric("test_accuracy", d["test"]["accuracy"])

        return active.info.run_id


def sqlite_tracking_uri(db_path: Path) -> str:
    """Return a local SQLite tracking URI. Keeps MLflow state inside the repo."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"
