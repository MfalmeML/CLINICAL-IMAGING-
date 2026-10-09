# LUNA16 metadata

Downloaded: 2026-10-09
Source: https://zenodo.org/records/3723295

## Files

- `annotations.csv` — ground-truth nodule locations and diameters.
  Columns: seriesuid, coordX, coordY, coordZ, diameter_mm.
  Rows: 1186.
- `candidates.csv` — candidate locations for the detection/FP-reduction track.
  Columns: seriesuid, coordX, coordY, coordZ, class.
  Rows: 551065.

Not downloaded: `sampleSubmission.csv`, `candidates_V2.csv`, the 10 CT subset archives.
The CT subsets are required before any model can be trained. This is the next
data-acquisition step, not the next code step.

## Integrity

`SHA256SUMS` in the same directory pins both files.

## Note on filenames

`sample_submission.csv` does not exist. The correct name is `sampleSubmission.csv`.
No standalone `evaluation.py` exists at the Zenodo record; the evaluation script
is bundled with the challenge submission package. Neither is needed for preprocessing
or for the perception baseline ladder.
