"""MLflow experiment tracking for the baseline ladder.

Every run logs the same parameter keys and the same metric keys.
Metric names match the build plan's required evaluation set:
AUROC, AUPRC, sensitivity, specificity, plus training loss/accuracy.
Accuracy is logged but is not the primary metric on this imbalanced task.
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
    """Log a single LadderRun to MLflow. Returns the run id."""
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
        mlflow.log_metric("test_sensitivity", d["test"]["sensitivity"])
        mlflow.log_metric("test_specificity", d["test"]["specificity"])
        mlflow.log_metric("test_auroc", d["test"]["auroc"])
        mlflow.log_metric("test_auprc", d["test"]["auprc"])

        return active.info.run_id


def sqlite_tracking_uri(db_path: Path) -> str:
    """Local SQLite tracking URI. Keeps MLflow state inside the repo."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"
