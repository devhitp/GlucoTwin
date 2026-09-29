# Hybrid Model — Architecture & Reference

> **DISCLAIMER:** This is a research evaluation framework. Results are NOT clinically validated.
> CGM unit remains UNKNOWN. The 70 threshold remains PROVISIONAL.
> All outputs are described as model-estimated risk probabilities, not clinical diagnoses.

## 1. Architecture

```
Raw observations (MetaboNet)
        |
Preprocessing (Sprint 7.5 pipeline)
        |
Digital Twin (Sprint 8 engine)
        |
  TwinState + TwinTrajectory (at T)
        |
TwinFeatures extractor (hybrid/trajectory_features.py)
        |
Existing ML features ───────────────────┐
                                        |
                               Hybrid LightGBM Model
                                        |
                            30m / 60m risk probability
                                        |
                          Calibration (Platt / Isotonic)
                                        |
                       Bootstrap Ensemble Uncertainty
```

## 2. Three Model Configurations

| Config | Features | Purpose |
|--------|---------|---------|
| **Baseline (A)** | Existing temporal ML features only | Sprint 7/7.5 performance baseline |
| **Twin-only (B)** | Twin-derived features only | Independent twin predictive value |
| **Hybrid (C)** | Baseline + Twin-derived features | Combined architecture |

## 3. Patient-Held-Out Protocol
- Deterministic train/val/test patient split (seed=42).
- No patient appears in more than one partition.
- Calibration is fitted on validation patients only.
- Test patients are only accessed for final evaluation.

## 4. Calibration
- Platt (logistic) calibration for small validation sets (<1000 samples).
- Isotonic regression for large validation sets (≥1000 samples).
- All calibration fitted on validation set only.
- Outputs labelled: "calibrated model probabilities under this evaluation protocol."

## 5. Uncertainty
- Bootstrap ensemble of N LightGBM members (default N=10).
- Each member trained on a bootstrap resample of training patients.
- Returns mean, 5th-percentile, 95th-percentile, and std of probability estimates.
- Status: `bootstrap_ensemble` — NOT a calibrated clinical confidence interval.

## 6. Causality Enforcement
- All Twin features use only observations at T.
- Simulated trajectories (T+30, T+60) are model-generated, never real future data.
- Residuals compare current state only (observed at T vs Twin at T).
- Future actual glucose is used ONLY in the separate Twin trajectory evaluation.

## 7. Limitations
- CGM unit is UNKNOWN — all thresholds and metrics are provisional.
- Twin dynamics are physiological-inspired engineering approximations.
- Wearable signals do not yet influence Twin dynamics.
- Bootstrap uncertainty intervals are not conformal or statistically guaranteed.

## 8. Synthetic Ablation Results (Sprint 9)

> Results below are from 1,000-sample synthetic data. NOT from real patient data.

### 30-Minute Horizon

| Config | ROC-AUC | PR-AUC | Recall | F1 | Brier (raw) | Brier (cal) |
|--------|---------|--------|--------|----|-------------|-------------|
| Baseline | 0.9854 | 0.9426 | 0.8636 | 0.7755 | 0.0377 | 0.0339 |
| Twin-only | 0.9986 | 0.9930 | 1.0000 | 0.8980 | 0.0265 | 0.0235 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 0.9362 | 0.0196 | 0.0191 |

### 60-Minute Horizon

| Config | ROC-AUC | PR-AUC | Recall | F1 | Brier (raw) | Brier (cal) |
|--------|---------|--------|--------|----|-------------|-------------|
| Baseline | 0.9823 | 0.9647 | 0.9024 | 0.9024 | 0.0447 | 0.0464 |
| Twin-only | 0.9991 | 0.9977 | 0.9268 | 0.9500 | 0.0187 | 0.0248 |
| Hybrid | 1.0000 | 1.0000 | 0.9268 | 0.9620 | 0.0140 | 0.0210 |

> **IMPORTANT:** Perfect AUC on synthetic data is expected because labels were generated deterministically from the same glucose features that the Twin and baseline models see. These results validate pipeline correctness, not generalization to real patients.

## 9. Future Extensions (Sprint 10+)
- Run ablation on real MetaboNet cohort (938 patients, patient-held-out).
- Per-patient metric breakdown.
- Statistical paired comparison (bootstrapped CI for metric difference).
- What-If counterfactual simulation engine.
- SHAP explainability for feature attribution.
