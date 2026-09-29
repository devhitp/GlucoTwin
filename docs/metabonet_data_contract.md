# MetaboNet Dataset Qualification Contract

## Dataset Identity
The data used is the MetaboNet public dataset.

## Local File
`data/raw/metabonet_public.parquet` (must never be committed to source control).

## Row Count
Approximately 154,842,077 rows (divided into 149 Parquet row groups).

## Subject Count
1,291 total unique subject IDs.

## Core Cohort Definition
The core cohort consists of subjects that meet all of the following:
- >= 14 days of recording duration
- >= 1000 CGM valid rows
- Contains any insulin and carbohydrate records

This reduces the effective cohort to 938 subjects.

## Relevant Fields

| Field | Meaning | Datatype | Unit | Unit Confidence | Missingness | Modeling Status |
|-------|---------|----------|------|-----------------|-------------|-----------------|
| `id` | Subject identifier | String | N/A | N/A | None | Required |
| `date` | Observation timestamp | Datetime | N/A | N/A | None | Required |
| `CGM` | Continuous Glucose Monitor reading | Float | UNKNOWN | UNKNOWN | Allowed (gaps exist) | Required |
| `insulin` | Total insulin or insulin proxy | Float | UNKNOWN | UNKNOWN | Sparse | Required (when available) |
| `basal` | Basal insulin rate/dose | Float | UNKNOWN | UNKNOWN | Sparse | Required (when available) |
| `bolus` | Bolus insulin dose | Float | UNKNOWN | UNKNOWN | Sparse | Required (when available) |
| `carbs` | Dietary carbohydrates | Float | UNKNOWN | UNKNOWN | Sparse | Required (when available) |
| `steps` | Activity proxy | Float | UNKNOWN | UNKNOWN | Extremely Sparse | Optional |
| `heartrate` | Heart rate | Float | UNKNOWN | UNKNOWN | Extremely Sparse | Optional |
| `galvanic_skin_response` | EDA/GSR | Float | UNKNOWN | UNKNOWN | Extremely Sparse | Optional |
| `skin_temp` | Skin temperature | Float | UNKNOWN | UNKNOWN | Extremely Sparse | Optional |

## Unit Status
All physiological units (including CGM, insulin, and carbohydrates) are strictly **UNKNOWN**. There is no authoritative schema or metadata available. All current modeling assumes mg/dL for CGM solely based on numerical plausibility. 

## Timestamp Status
**UNKNOWN.** Contains future dates extending into 2025-2027. It is unknown if this is due to de-identification, device clock issues, or data corruption.

## Missingness
Wearable data is extremely sparse. Insulin and carbohydrate events are episodic and missing for long periods. 

## Known Anomalies
- 6 subjects contain timestamps > 2025.
- Wearable signals are missing for the vast majority of the core cohort.

## Modeling Assumptions
- We assume timestamps are monotonic and preserve physiological elapsed time within a subject stream.
- We provisionally assume 70 is a valid engineering threshold for evaluation.

## Prohibited Assumptions
- Do NOT assume CGM is explicitly mg/dL or mmol/L.
- Do NOT convert units using fabricated conversion factors.
- Do NOT silently shift timestamps or align them to calendar events (e.g., holidays) without resolving provenance.

## Clinical Interpretation Restrictions
This dataset is qualified for **engineering and research evaluation only**. The current models and thresholds must not be interpreted as clinically valid, prescriptive, or advisory.
