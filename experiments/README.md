# Experiments

MLflow tracking database and metadata for baseline-ladder runs.

## Tracking database

`experiments/mlflow.db` — SQLite. Created on first run. Gitignored.

To inspect locally:

```sh
mlflow ui --backend-store-uri sqlite:///$(pwd)/experiments/mlflow.db
```

Then open http://127.0.0.1:5000 in a browser.

## Run naming

Each baseline run uses a stable name from this fixed list:

- `baseline1-2d-cnn`
- `baseline2-2p5d-cnn`
- `baseline3-3d-cnn`
- `baseline4-3d-vit`
- `advanced-ssl-3d`

Do not invent names. Do not append dates or ad hoc suffixes. The name
identifies which rung of the ladder the run belongs to. Everything else
that distinguishes runs — seed, data split, hyperparameters — belongs in
MLflow params, not in the run name.

## Required params on every run

- `architecture` — must match the baseline name
- `seed` — integer
- `epochs` — integer
- `batch_size` — integer
- `lr` — float
- `dataset` — `luna16`
- `split_seed` — integer; must match across baselines
- `preprocessing_config_hash` — sha256 of configs/preprocessing.yaml

The `split_seed` and `preprocessing_config_hash` params are what make
baseline-to-baseline comparison honest. If they differ between runs,
the ablation is invalid.

## Required metrics on every run

Logged automatically by `log_ladder_run`:

- `train_loss`, `train_accuracy`
- `val_loss`, `val_accuracy`
- `test_loss`, `test_accuracy`
- `test_sensitivity`, `test_specificity`
- `test_auroc`, `test_auprc`

Accuracy is logged for continuity but is not the decision metric. AUROC
and AUPRC are the decision metrics on this imbalanced task.

## What is not in this directory

- Model weights. Those are large and go to a separate artifact store.
- Raw data. That is under `data/raw/` and gitignored.
- Ad hoc CSV exports. The tracking database is the record.
