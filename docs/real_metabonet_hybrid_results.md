# Real MetaboNet Hybrid Results (Sprint 10)

> **PROVISIONAL WARNING:** All results in this report use a provisional 70 mg/dL threshold under unresolved CGM unit provenance. These results are NOT clinically validated.

## 1. Dataset & Cohort
- **Dataset**: `metabonet_public.parquet` (approx. 154M rows, 1.3GB)
- **Core Eligible Subjects**: 938
- **Split Protocol**: Deterministic patient-held-out (seed=42). No patient appears in more than one partition.

## 2. Evaluation Stages Status
| Stage | Target Subjects | Status | Note |
|-------|-----------------|--------|------|
| Smoke | 14 (10/2/2 split) | **COMPLETE** | Validated pipeline end-to-end |
| Pilot | ~79 | PENDING | Deferred to Sprint 11 due to memory profiling |
| Full  | 938 | PENDING | Deferred to Sprint 11 |

## 3. Real Twin Forecast Evaluation (Smoke Stage)
*Note: Evaluated on test set only. Error metrics are on the unconfirmed numerical glucose scale.*

| Horizon | N (Windows) | MAE | RMSE | Mean Bias |
|---------|-------------|-----|------|-----------|
| 30m | 324,475 | 58.03 | 77.21 | +24.88 |
| 60m | 324,475 | 49.26 | 64.81 | +8.18 |

> **Analysis**: The Twin forecast error on real data is substantial (MAE ~50-58). This indicates a strong unit-scale mismatch or lack of personalized parameter tuning. The uncalibrated population defaults used in Sprint 10 do not accurately track the real MetaboNet subjects' absolute glucose values.

## 4. Real Hybrid Ablation Results (Smoke Stage)

### 30-Minute Target
| Config | ROC-AUC | PR-AUC | Recall |
|--------|---------|--------|--------|
| Baseline | 0.9632 | 0.7600 | 0.9046 |
| Twin-only | 0.4624 | 0.0507 | 0.0000 |
| Hybrid | 0.9630 | 0.7595 | 0.8975 |

### 60-Minute Target
| Config | ROC-AUC | PR-AUC | Recall |
|--------|---------|--------|--------|
| Baseline | 0.9180 | 0.6581 | 0.8435 |
| Twin-only | 0.4499 | 0.0751 | 0.0000 |
| Hybrid | 0.9165 | 0.6562 | 0.8033 |

## 5. Critical Interpretation
**Did adding the Twin improve the baseline on real patients?**  
No. Under this patient-held-out engineering evaluation, the hybrid model showed negligible differences (and slight degradation in PR/Recall) relative to the baseline.

**Why?**
The `Twin-only` configuration performance (ROC ~0.45, worse than random) confirms that the uncalibrated Twin features do not provide a coherent predictive signal on real MetaboNet data. Because the Twin's absolute forecast error is so high (MAE ~58), LightGBM essentially ignores the Twin features in the Hybrid model, falling back entirely to the Baseline temporal features.

## 6. System Profiling & Limitations
- **Runtime**: ~1031 seconds (17 minutes) for smoke stage.
- **Memory**: Peak memory reached **6,416 MB** for just 14 subjects. The current concatenation logic inside the split arrays prevents scaling to the full 938-subject cohort without out-of-core learning (e.g., LightGBM Dataset streaming).
- **Uncertainty & Calibration**: Evaluated and confirmed working synthetically, but real-data full evaluation deferred until the Twin parameters are tuned and the memory bottleneck is addressed.
- **Paired Comparison / Per-Patient**: Deferred. With identical Baseline/Hybrid performance, statistical comparison yields a null result.

## 7. Next Steps (Sprint 11)
1. **Parameter Personalization**: The Twin must be fit/tuned to individual patient baselines using the training partition to reduce the MAE from ~58 down to a usable physiological range.
2. **Memory Optimization**: Convert the feature matrix construction to stream directly into LightGBM binary datasets to bypass Pandas concatenation limits.
