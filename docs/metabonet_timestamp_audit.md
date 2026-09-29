# MetaboNet Timestamp Audit

**Dataset**: `data/raw/metabonet_public.parquet`

## Observed Anomaly
The dataset contains timestamps extending into the future (up to 2027-07-14) relative to the project context (late 2026).

## Investigation Findings
- **Observed Date Range**: 2011-11-17 to 2027-07-14.
- **Affected Subjects**: Exactly 6 subjects out of 1,291 have timestamps >= 2025:
  - Subject 540: 2018-09-27 to 2027-07-14 (152,927 rows)
  - Subject 544: 2027-05-11 to 2027-07-05 (15,840 rows)
  - Subject 552: 2018-05-10 to 2025-06-07 (137,671 rows)
  - Subject 567: 2026-12-28 to 2027-02-23 (16,515 rows)
  - Subject 584: 2019-01-05 to 2025-07-09 (37,475 rows)
  - Subject 596: 2018-05-12 to 2027-06-06 (105,796 rows)
- **Evidence Found**: There is no documentation within the repository or in the dataset metadata explaining these future dates. 
- **Cause**: UNKNOWN. It is impossible to determine whether this is an intentional de-identification strategy (shifting dates forward), a device clock configuration error by the patient, or a data corruption artifact.

## Recommendations for Modeling
- The timestamps appear locally consistent (sequential) for the affected patients, allowing within-subject temporal feature engineering (e.g. rolling windows) to function computationally.
- However, they must not be aligned to real-world global events (like holidays or global daylight savings) since their true origin is unknown.
- Do not silently shift or drop these rows without further instruction.
