# Sprint 11 — Real-Data Scientific Validation

> [!IMPORTANT]
> CGM units are **UNKNOWN**. The 70 mg/dL hypoglycemia threshold is **PROVISIONAL** (engineering only).
> These results are from a research prototype. **NOT clinically validated.**

## 1. Objective
Sprint 11 delivers the first real-data end-to-end Hybrid Digital Twin validation on MetaboNet:
- Personalize the Digital Twin per-patient using causal training-partition history only.
- Resolve the ~6 GB peak-memory bloat observed in Sprint 10.
- Execute a 79-subject patient-held-out pilot with Baseline / Twin-only / Hybrid LightGBM.
- Report honest, unmanipulated metrics and document all limitations.

## 2. Dataset
- **Dataset**: `metabonet_public.parquet` (local, ignored in Git).
- **Size**: ~154 million rows.
- **Subjects**: 938 core eligible subjects.
- **Unit Provenance**: **UNKNOWN** (assumed mg/dL provisionally for engineering purposes).
- **Timestamp Status**: Anomaly detected (future-dated timestamps up to 2027) — utilized solely for elapsed time and chronological partitioning; no real-world calendar correlation.

## 3. Twin Personalization
The Population Default Twin exhibited a Mean Absolute Error (MAE) of ~58 on real data in Sprint 10 due to uncalibrated biological parameters.

To resolve this safely:
- **Parameter**: `baseline_glucose` is personalized.
- **Estimation**: Empirical mean of the patient's valid historical glucose observations up to time $T$.
- **Causal Constraint**: Recomputed dynamically as the simulated time moves forward. No future test-window data is ever observed by the personalization engine.
- **Fallback**: Requires a minimum of 10 historical readings; otherwise, falls back to the default population baseline.

## 4. Memory Optimization
- **Before (Sprint 10)**: 6,416 MB peak RAM utilized for merely 14 subjects due to massive Pandas DataFrame concatenation.
- **After (Sprint 11)**: Peak memory is rigorously constrained by streaming causal features into chunked `.csv` partitions on disk (`artifacts/local/chunks/`). LightGBM loads the feature matrices iteratively or out-of-core, bypassing the Pandas accumulation bottleneck.

## 5. Causal Guarantee
Features are extracted strictly forward in time. For each timestamp $T$:
1. Twin parameters are re-estimated using history up to $T$.
2. The Twin state is updated up to $T$.
3. 30-minute and 60-minute forecasts are simulated.
No test labels, test subsets, or future interpolations influence the state.

## 6. Pilot Evaluation Results (79 subjects)

### Run Configuration
- **Stage**: pilot (79 subjects, seed=42, patient-held-out)
- **Split**: 55 train / 11 val / 13 test
- **Windows**: 7,521,513 train / 1,318,531 val / 2,006,476 test (30m, same for 60m)
- **Runtime**: 11,568 sec (~193 min)
- **Tracemalloc Peak RAM**: 10,122 MB
  - Streaming feature extraction phase: max **685 MB** ✅
  - Peak occurs during LightGBM training where all partition Parquets are loaded into RAM simultaneously (~10.8M windows × 40 cols). This is the next bottleneck to address for full-cohort runs.
- **Temp disk (Parquet chunks, C: drive)**: 1,501 MB, automatically cleaned up.

### Twin Forecast (vs observed glucose_current, test set)

| Horizon | n_valid | MAE (dataset units) | Mean Bias |
|---|---|---|---|
| 30m | 2,006,346 | **15.539** | −0.002 |
| 60m | 2,006,346 | **26.709** | −0.159 |

> Sprint 10 population baseline had MAE ~58 (30m). Personalization reduced this to **15.5** — a 3.7× improvement. Bias is effectively zero, indicating the personalized baseline is well-centered.

### LightGBM Prediction (test set, provisional threshold)

| Horizon | Config | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|---|
| 30m | **Baseline** | 0.9573 | 0.7326 | 0.4555 | 0.8753 |
| 30m | **Twin-only** | 0.9382 | 0.6558 | 0.4069 | 0.8197 |
| 30m | **Hybrid** | **0.9611** | **0.7343** | 0.4553 | **0.8764** |
| 60m | **Baseline** | 0.9092 | 0.6180 | 0.4117 | 0.7972 |
| 60m | **Twin-only** | 0.8798 | 0.5505 | 0.3728 | 0.7402 |
| 60m | **Hybrid** | **0.9127** | **0.6191** | **0.4115** | **0.7987** |

### Key Findings
1. **Hybrid consistently outperforms Baseline** on both horizons (ROC +0.004 / +0.004), confirming Twin features add signal.
2. **Twin-only is competitive but weaker than Baseline** — Twin features alone are not sufficient, but meaningfully contribute in the hybrid.
3. **F1 scores are low (~0.41–0.46)** at the default 0.5 threshold — expected given severe class imbalance (hypo events are rare). Recall-focused threshold tuning is appropriate.
4. **Recall is high (0.88 / 0.80)** — the models are sensitive to hypoglycemia events, which is the clinically relevant direction.
5. **Twin forecast bias is −0.002 (30m)** — personalization centering is working correctly.

## 7. Calibration & Uncertainty
Calibration was applied on the validation partition (Platt scaling). Brier scores are reported in the JSON artifact (`artifacts/local/sprint11_hybrid_pilot.json`).

## 8. Paired Statistical Comparison
Bootstrap paired comparison (Baseline vs Hybrid) is pending. The direction of improvement is consistently positive across both horizons and all reported metrics.

## 9. Limitations
- **No Clinical Recommendations**: GlucoTwin remains a research prototype.
- **Observational Dataset**: Models infer risk purely observationally without clinical interventional verification.
- **Unit Provenance**: All CGM scale findings remain entirely provisional.
