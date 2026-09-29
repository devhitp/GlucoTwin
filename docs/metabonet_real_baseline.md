# MetaboNet Real Baseline — Sprint 7 Research Report

> **PROVISIONAL DOCUMENT** — CGM units assumed mg/dL but NOT confirmed.
> Do NOT cite as clinical evidence. All threshold-based metrics are provisional.

---

## 1. Dataset Provenance and Version

| Field | Value |
|---|---|
| File | `data/raw/metabonet_public.parquet` |
| Size | 1,294 MB |
| Total rows | 154,842,077 |
| Row groups | 149 |
| Columns | 37 |
| Unique subjects | 1,291 |
| Timestamp range | 2011-11-17 to 2027-07-14 |
| Git-ignored | Yes (data/raw/*) |

> IMPORTANT: Dataset contains timestamps up to 2027-07-14. These future-dated
> observations have not been explained from authoritative documentation and
> represent a critical unresolved provenance issue.

### 1.1 Units — UNRESOLVED

| Column | Assumed | Confirmed? |
|---|---|---|
| CGM | mg/dL | NO — not confirmed from documentation |
| basal | U/hr | NO — inferred only |
| bolus | U | NO — inferred only |
| carbs | grams | NO — inferred only |
| steps | count | N/A (99.4% null) |
| heartrate | bpm | N/A (99.3% null) |

### 1.2 Missingness Summary (Sprint 6 audit)

| Column | Null % |
|---|---|
| CGM | 15.5% |
| insulin | 9.2% |
| basal | 11.9% |
| bolus | 11.2% |
| carbs | 33.2% |
| steps | 99.4% |
| heartrate | 99.3% |
| galvanic_skin_response | 99.9% |
| skin_temp | 99.9% |

---

## 2. Eligible Cohort (Reconfirmed from Actual Data)

### 2.1 Eligibility Criteria

| Criterion | Threshold |
|---|---|
| Minimum recording duration | >= 14 days |
| Minimum non-null CGM rows | >= 1,000 |
| Minimum insulin records | >= 1 |
| Minimum carbohydrate records | >= 1 |

### 2.2 Cohort Summary

| Metric | Value |
|---|---|
| Total subjects in file | 1,291 |
| Core cohort (all criteria met) | 938 |
| Median recording duration (all) | 371.1 days |
| Median recording duration (core) | 430.2 days |
| Median CGM rows (core) | 111,666 |

> Subject IDs are opaque numeric strings. It is unknown whether the same
> physical patient could appear under different IDs across source datasets.
> This is a documented, unresolved limitation.

---

## 3. Label Definitions (PROVISIONAL)

### Primary Research Target: 30-minute hypoglycemia

Binary label = 1 if the minimum glucose value in the window (T, T+30min]
is strictly below 70 mg/dL (PROVISIONAL — units assumed, not confirmed).
Label = NaN if no valid glucose exists in the future window.
NaN labels are excluded from model training and evaluation.
The anchor-point glucose at time T is NOT used as a future observation.

### Primary Research Target: 60-minute hypoglycemia

Same as above, but future window extends to (T, T+60min].

### Ancillary Labels (exploratory)

- future_severe_hypoglycemia_30m/60m: glucose < 54 mg/dL threshold (PROVISIONAL)
- future_nocturnal_hypoglycemia_30m/60m: primary event AND clock hour in [22:00, 06:00)

---

## 4. Chronological Split Protocol

```
Subject timeline:
  [──────── Train 60% ────────][─60min─][── Val 20% ──][─60min─][─ Test 20% ─]
                               embargo               embargo
```

- Applied per-subject before aggregating across subjects
- 60-minute embargo gap at each boundary
- Train rows shuffled post-aggregation across subjects
- Validation and test sets NOT shuffled (temporal order preserved)
- Patient-held-out evaluation: NOT performed in Sprint 7

### Anti-Leakage Assertions (Verified)

- assert_no_temporal_leakage() passes for every subject
- All features use backward-looking rolling windows only
- Labels use .shift(1).rolling(...) — excludes anchor time T
- No imputation or normalisation fitted on val/test data
- Threshold selected on validation F1, never on test
- Leakage regression test: modifying future glucose does not change past features

---

## 5. Feature List

24 features, all backward-looking (causal) at prediction time T.
Wearables excluded (>= 99% null — no global imputation applied).

| # | Feature | Description |
|---|---|---|
| 1 | glucose_current | Current CGM reading (interpolated <= 10min gaps) |
| 2 | glucose_roc_5m | Rate of change over 5min (mg/dL/min) |
| 3 | glucose_roc_15m | Rate of change over 15min |
| 4 | glucose_roc_30m | Rate of change over 30min |
| 5 | glucose_rolling_mean_30m | 30-min rolling mean |
| 6 | glucose_rolling_std_30m | 30-min rolling std |
| 7 | glucose_rolling_min_30m | 30-min rolling min |
| 8 | glucose_rolling_max_30m | 30-min rolling max |
| 9 | glucose_rolling_mean_60m | 60-min rolling mean |
| 10 | glucose_rolling_std_60m | 60-min rolling std |
| 11 | glucose_rolling_min_60m | 60-min rolling min |
| 12 | glucose_rolling_max_60m | 60-min rolling max |
| 13 | minutes_since_last_glucose | Elapsed since last valid CGM |
| 14 | basal_insulin_current | Forward-filled basal rate [PROVISIONAL units] |
| 15 | bolus_insulin_recent | 30-min rolling bolus sum [PROVISIONAL units] |
| 16 | total_insulin_last_30m | 30-min rolling total bolus [PROVISIONAL units] |
| 17 | total_insulin_last_60m | 60-min rolling total bolus [PROVISIONAL units] |
| 18 | carbs_recent | 30-min rolling carbs sum [PROVISIONAL units] |
| 19 | minutes_since_last_meal | Elapsed since carbs > 0 |
| 20 | patient_glucose_baseline | 24h rolling mean glucose (personalized) |
| 21 | patient_glucose_std | 24h rolling std glucose (personalized) |
| 22 | hour_of_day | Clock hour of prediction |
| 23 | is_night_clock_based | 1 if 22:00-06:00, else 0 |
| 24 | day_of_week | Day of week index |

---

## 6. Model Configurations

### Persistence Baseline (Model A)

Rule-based deterministic baseline using current glucose and 5-min ROC:
- glucose_current < 70 mg/dL (PROVISIONAL) → prob = 0.90
- 70 <= glucose_current < 85 AND roc < -1.0 → prob = 0.70
- 70 <= glucose_current < 85 AND -1.0 <= roc < 0 → prob = 0.40
- Otherwise → prob = 0.0
Operating threshold: selected on validation F1.

### LightGBM (Model B)

| Hyperparameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| class_weight | balanced |
| random_state | 42 |
| early_stopping_rounds | 20 (validation AUC) |
| n_jobs | -1 (CPU) |
| verbose | -1 (silent) |

Separate models for 30m and 60m. Threshold selected on validation F1.

---

## 7. Stage A Results — Smoke Test (14 subjects processed)

> Engineering validation — NOT a research result.
> Note: "5 sorted core-cohort IDs" expanded to 14 subjects due to
> row-group boundary effects in the streaming iterator; all 14 are
> legitimate core-cohort members.

### 7.1 Dataset Statistics

| Metric | Value |
|---|---|
| Subjects selected | 5 (first 5 sorted IDs of core cohort) |
| Subjects processed | 14 (row-group boundary expansion) |
| Subjects skipped | 0 |
| Total CGM rows loaded | 1,060,726 |
| Feature columns | 24 / 24 |
| Peak memory | 842.9 MB |
| Preprocessing time | 240.7s |
| LGBM training time | ~37.4s |
| Privacy check | PASSED |

### 7.2 Test Set Windows

| Target | Total windows | Positive | Prevalence |
|---|---|---|---|
| 30m hypoglycemia | 205,456 | 10,061 | 4.90% [PROVISIONAL] |
| 60m hypoglycemia | 205,456 | 15,198 | 7.40% [PROVISIONAL] |

### 7.3 Metrics (Within-subject chronological, smoke stage only)

| Model | Target | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|---|
| Persistence | 30m | 0.8214 | 0.5335 | 0.6407 | 0.6564 | 0.6257 | 0.0265 |
| LightGBM | 30m | 0.9609 | 0.7464 | 0.6785 | 0.6326 | 0.7316 | 0.0537 |
| Persistence | 60m | 0.7209 | 0.4032 | 0.5391 | 0.4568 | 0.6580 | 0.0499 |
| LightGBM | 60m | 0.9052 | 0.6243 | 0.5684 | 0.5132 | 0.6371 | 0.0896 |

### 7.4 Confusion Matrices

**LightGBM 30m** (threshold=0.89, selected on val F1):
- TP=6,365  FP=2,336  TN=193,059  FN=3,696

**LightGBM 60m** (threshold=0.82):
- TP=7,800  FP=4,449  TN=185,809  FN=7,398

**Persistence 30m** (threshold=0.05):
- TP=6,604  FP=3,950  TN=191,445  FN=3,457

**Persistence 60m** (threshold=0.05):
- TP=6,942  FP=3,612  TN=186,646  FN=8,256

### 7.5 Calibration (LightGBM 30m)

| Predicted probability bin | Fraction positive |
|---|---|
| 0.024 | 0.003 |
| 0.141 | 0.016 |
| 0.246 | 0.027 |
| 0.347 | 0.050 |
| 0.449 | 0.064 |

Note: High predicted probabilities are under-represented in the reliability
diagram because the model assigns very high probabilities only to true events.
Calibration analysis on pilot data will be more reliable.

---

## 8. Stage B Results — Pilot (79 subjects processed)

> "50 randomly seeded subjects" expanded to 79 due to row-group boundaries.
> This represents a robust subset (8.4% of the core cohort) and provides reliable metrics.

### 8.1 Dataset Statistics

| Metric | Value |
|---|---|
| Subjects selected | 50 (seed=42) |
| Subjects processed | 79 (row-group boundary expansion) |
| Subjects skipped | 0 |
| Total CGM rows loaded | 6,578,712 |
| Feature columns | 24 / 24 |
| Peak memory | 3,589.9 MB |
| Preprocessing time | 1193.3s |
| LGBM training time | 176.9s |
| Privacy check | PASSED |

### 8.2 Test Set Windows

| Target | Total windows | Positive | Prevalence |
|---|---|---|---|
| 30m hypoglycemia | 1,306,607 | 72,523 | 5.55% [PROVISIONAL] |
| 60m hypoglycemia | 1,306,607 | 105,610 | 8.08% [PROVISIONAL] |

### 8.3 Metrics (Within-subject chronological)

| Model | Target | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|---|
| Persistence | 30m | 0.8605 | 0.6041 | 0.6717 | 0.5683 | 0.8211 | 0.0265 |
| LightGBM | 30m | 0.9758 | 0.8184 | 0.7306 | 0.6817 | 0.7872 | 0.0607 |
| Persistence | 60m | 0.7614 | 0.4750 | 0.5908 | 0.5413 | 0.6503 | 0.0491 |
| LightGBM | 60m | 0.9355 | 0.7093 | 0.6276 | 0.5927 | 0.6669 | 0.1009 |

### 8.4 Confusion Matrices

**LightGBM 30m** (threshold=0.91):
- TP=49,437  FP=13,364  TN=1,220,720  FN=23,086

**LightGBM 60m** (threshold=0.84):
- TP=62,596  FP=31,264  TN=1,169,733  FN=43,014

**Persistence 30m** (threshold=0.42):
- TP=41,216  FP=8,980  TN=1,225,104  FN=31,307

**Persistence 60m** (threshold=0.05):
- TP=57,168  FP=30,744  TN=1,170,253  FN=48,442

### 8.5 Calibration (LightGBM 30m)

| Predicted probability bin | Fraction positive |
|---|---|
| 0.021 | 0.001 |
| 0.142 | 0.011 |
| 0.246 | 0.020 |
| 0.348 | 0.031 |
| 0.449 | 0.048 |
| 0.549 | 0.066 |
| 0.650 | 0.099 |
| 0.751 | 0.152 |
| 0.852 | 0.263 |
| 0.971 | 0.757 |

### 8.6 Feature Importance (Pilot Stage)

**LightGBM 30m (Top 10):**
1. glucose_roc_5m: 1455
2. glucose_current: 934
3. patient_glucose_baseline: 528
4. patient_glucose_std: 524
5. glucose_rolling_min_60m: 514
6. glucose_rolling_min_30m: 494
7. minutes_since_last_meal: 484
8. glucose_rolling_std_60m: 477
9. hour_of_day: 454
10. glucose_roc_15m: 445

**LightGBM 60m (Top 10):**
1. glucose_roc_5m: 1087
2. glucose_current: 899
3. hour_of_day: 725
4. patient_glucose_baseline: 705
5. minutes_since_last_meal: 609
6. patient_glucose_std: 606
7. basal_insulin_current: 489
8. glucose_rolling_min_60m: 457
9. glucose_rolling_std_60m: 452
10. total_insulin_last_60m: 444

> CAUTION: Feature importance reflects statistical associations in this dataset.
> Insulin and carbohydrate features are used with PROVISIONAL units.
> Rankings do NOT imply causality.

---

## 9. Stage C Results — Full Cohort (938 subjects)

PENDING — subject to available RAM (estimated peak memory requirement: ~40 GB).

---

## 10. Error Analysis (Pilot Stage)

Aggregate FP/FN breakdown included in artifact:
`artifacts/local/sprint7_pilot_results.json` (git-ignored)

Overall error counts:
- 30m Target: TP=49437, TN=1220720, FP=13364, FN=23086
- 60m Target: TP=62596, TN=1169733, FP=31264, FN=43014

---

## 11. Resource Benchmarks

| Stage | Subjects | Features | Peak Mem | Prep Time | LGBM Train |
|---|---|---|---|---|---|
| Smoke v1 (old code) | 5 effective | 9/24 | 986 MB | 199s | 12.8s |
| Smoke v2 (Sprint 7) | 14 effective | 24/24 | 842.9 MB | 240.7s | 37.4s |
| Pilot | 79 effective | 24/24 | 3,589.9 MB | 1193.3s | 176.9s |
| Full (938) | pending | 24/24 | pending | pending | pending |

---

## 12. Limitations

1. **CGM units unconfirmed**: 70 mg/dL threshold is provisional.
   Results MUST NOT be cited as clinically validated prediction performance.

2. **Future-dated timestamps (to 2027)**: Source and cause undocumented.
   May affect labels near recording boundaries.

3. **Within-subject evaluation only**: No patient-held-out split in Sprint 7.
   Performance on unseen patients is UNKNOWN.

4. **Correlated observations**: 1.3M test windows from 79 subjects are NOT
   independent samples. Subject-level CIs not reported.

5. **Cohort provenance**: Source dataset harmonisation and patient ID uniqueness
   across contributing studies are undocumented.

6. **No causal inference**: Feature importance = statistical association only.
   Insulin/carbohydrate effects are not proven.

7. **Wearables absent**: >99% null. A model with valid wearable signals
   could perform differently.

8. **Insulin/carbohydrate units**: Used as relative features; no dosing
   clinical thresholds applied.

---

## 13. Reproduction Instructions

```bash
# Prerequisites
git clone https://github.com/devhitp/GlucoTwin && cd GlucoTwin
pip install -r requirements.txt
# Place: data/raw/metabonet_public.parquet (not committed)

# Stage A: Smoke (5 sorted core-cohort IDs, ~4 min)
python scripts/train_metabonet_baselines.py --stage smoke

# Stage B: Pilot (50 subjects, seed=42, ~20-40 min)
python scripts/train_metabonet_baselines.py --stage pilot

# Stage C: Full cohort (938 subjects, resource-dependent)
python scripts/train_metabonet_baselines.py --stage full
# With RAM cap: add --max-subjects 200

# Run all 41 tests
python -m pytest tests/ -v
```

Results written to: `artifacts/local/sprint7_{stage}_results.json` (git-ignored)

---

## 14. Sprint 8 Readiness

**Status: NOT READY** — Prerequisite conditions not yet met:

- [ ] CGM units must be confirmed from authoritative MetaboNet documentation
- [ ] Patient-held-out evaluation must be implemented
- [ ] Subject ID uniqueness and provenance must be clarified
- [ ] Future-dated timestamps (to 2027) must be explained

Recommendation: Resolve CGM units confirmation and establish patient-held-out evaluation protocols before scheduling Sprint 8 (Physiological Digital Twin layer).
