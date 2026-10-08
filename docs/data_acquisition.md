# Data acquisition plan

## LIDC-IDRI (core labels, segmentation ground truth)

Source: TCIA (The Cancer Imaging Archive).
Access: public, no DUA required.
Structure: ~1010 patients, 4-radiologist annotations per scan, DICOM.
Retrieval: TCIA NBIA Data Retriever or `tcia_utils` Python package.
Target: `data/raw/LIDC-IDRI/`.

## LUNA16 (standardized detection benchmark)

Source: grand-challenge.org, derived from LIDC-IDRI.
Access: public, no DUA required.
Structure: 888 CT scans, 10 cross-validation subsets, provided candidate locations,
evaluation scripts, FP-reduction scripts.
Retrieval: direct download from grand-challenge or mirror.
Target: `data/raw/LUNA16/`.

## NLST (longitudinal, outcome data)

Source: NCI Cancer Data Access System (CDAS).
Access: requires data-use agreement and approval.
Structure: ~54,000 participants, ~75,000 screening CTs, longitudinal, clinical outcome data.
Lead time: weeks to months. Begin the application now; do not block Phases 2-3 on it.
Target: `data/raw/NLST/` (empty until approval).

## Rules

- Never commit raw data. `.gitignore` already excludes `data/raw/`.
- De-identification runs on every study regardless of upstream status.
- Record dataset version and retrieval date for every dataset pulled.
- Record license and citation requirements alongside each dataset.
