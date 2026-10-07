# IMAGEX

Uncertainty-aware longitudinal clinical imaging intelligence platform.

## Scope (V1)

Chest CT. Pulmonary nodules. Detection, localization, segmentation, quantification,
longitudinal change, uncertainty, triage priority, second-reader discrepancy flag.

Explicitly not clinical deployment. Research prototype on licensed, de-identified data.

## Status

Phase 1 — Data foundation. DICOM loading and validation only.

## Layout

- `src/imagex/preprocessing/` — DICOM loading, validation, de-identification, HU conversion, resampling
- `tests/` — unit and integration tests
- `data/raw/` — untracked; raw DICOM
- `data/processed/` — untracked; preprocessed tensors
- `data/cache/` — untracked; versioned intermediate artifacts
- `configs/` — configuration
- `docs/` — specification excerpts and design notes

## Environment

Python venv at `.venv/`. Activate with `source .venv/bin/activate`.

## Tests
pytest tests/ -v

text

## Datasets

- LIDC-IDRI — core labels, segmentation ground truth
- LUNA16 — standardized detection benchmark, 10 CV subsets
- NLST — longitudinal pairs and outcome data; requires data-use agreement
