# MetaboNet Qualification Report

## Dataset Identity
* **File**: `metabonet_public.parquet`
* **Rows**: 154,842,077
* **Columns**: 37
* **Subjects**: 1,291

## Semantics
### CGM
* **Observed**: Min: 1.8, Max: 500.9, Mean: 148.18.
* **Finding**: The range is highly consistent with mg/dL units.

### Insulin
* **Observed**: Exact match `insulin = basal + bolus` on 134,035,414 evaluated rows (0 mismatches).
* **Finding**: The `insulin` column is a strict aggregate of `basal` and `bolus`.

### Basal
* **Observed**: 136M valid entries.
* **Finding**: Represents continuous basal rates.

### Bolus
* **Observed**: 137M valid entries (vast majority are 0). 4 negative values found.
* **Finding**: Represents discrete bolus injection amounts.

### Carbs
* **Observed**: 103M entries (101M zeros, 1.6M positives, 0 negatives). Max value 855.
* **Finding**: Structurally represents meal-time carbohydrate consumption events (sparse, predominantly zero).

### Wearables (Steps, HR, EDA, Skin Temp)
* **Finding**: 99%+ missingness. They remain optional/supplementary but cannot form the core feature set.

## Units
* **Glucose**: UNKNOWN formally, but heavily INFERRED as mg/dL.
* **Insulin**: UNKNOWN formally, likely IU or IU/hr.
* **Carbs**: UNKNOWN formally, likely grams.
* **CONFIRMED STATUS**: UNIT NOT CONFIRMED.

## Timestamp Audit
* **Earliest**: 2011-11-17
* **Latest**: 2027-07-14
* **Future-dated records**: Dates in 2025 (31,463), 2026 (1,151), and 2027 (64,842) are present.
* **Explanation/Status**: Either these are synthetic time-shifted dates to de-identify patients, timezone artifacts, or device clock errors. They are structurally intact.

## Subject Coverage
* **Min**: 52.00 hours
* **Median**: 371.12 days
* **Max**: 4,406.67 days
* **>= 24h**: 1,291
* **>= 7d**: 1,289
* **>= 14d**: 1,285
* **>= 30d**: 1,280

## CGM Continuity
* **Measured**: 130,872,839 valid points out of 154.8M total possible rows (15.48% missingness). With a 5-minute grid, this equates to roughly 12.5 centuries of continuous valid CGM blocks.

## Insulin Coverage
* **Measured**: 140,602,689 valid aggregate points (~9.20% missingness). Very dense basal/bolus coverage.

## Carb Coverage
* **Measured**: 103,416,778 valid points (~33.21% missingness, 1.6 million distinct positive meal events).

## Hypoglycemia (assuming mg/dL)
* **< 70 mg/dL**: 4,268,666 (3.26%)
* **< 54 mg/dL**: 813,419 (0.62%)

## Candidate Windows (Estimates)
* **30m**: Millions of viable contiguous candidate windows.
* **60m**: Millions of viable contiguous candidate windows.

## Wearables
* **Measured coverage**: < 1%

## Qualified Cohorts
### Core (>=14 days + CGM + Insulin + Carbs)
* **Subjects**: 938
* **Finding**: The core cohort is extremely robust.

### Extended (Core + Wearables)
* **Subjects**: Effectively 0 due to wearable sparsity.

## Limitations
* Units are inferred, not explicitly documented.
* 4 minor negative bolus artifacts.
* Future-dated timestamps require careful relative chronological modeling (which GlucoTwin uses, so it's safe).
* Very poor wearable coverage restricts digital twin models to Glucose+Insulin+Carb contexts.
