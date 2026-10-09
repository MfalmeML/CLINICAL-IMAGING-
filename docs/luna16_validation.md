# LUNA16 real-data validation

Date: 2026-10-09

## Results

| Quantity | Expected | Actual | Match |
|---|---|---|---|
| annotations parsed | 1186 | 1186 | yes |
| candidates parsed | 551065 | 551065 | yes |
| candidates positive | 1351 | 1351 | yes |
| candidates negative | 549714 | 549714 | yes |
| series with nodules | 601 | 601 | yes |
| series with candidates | 888 | 888 | yes |
| max candidates in one series | - | 1468 | - |
| min candidates in one series | - | 32 | - |

## Notes

- The earlier reference to 1557 positive candidates was wrong. That figure belongs to
  `candidates_V2.csv`, the extended candidate set. The standard `candidates.csv`
  contains 1351 positives.
- All seven quantities that have published reference values match exactly.

## Consequence

The parsers in `src/imagex/perception/luna16.py` are confirmed correct against the
real files. The next code step can consume `Candidate` and `NoduleAnnotation`
without further validation of the CSV layer.
