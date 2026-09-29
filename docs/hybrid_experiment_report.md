# Sprint 9 Hybrid Experiment Report

> All results in this report are from **synthetic data** unless explicitly marked "real data."
> CGM units are UNKNOWN. The 70 hypoglycemia threshold is PROVISIONAL.
> Results are described as research/engineering evaluation outputs only.

## Experiment Stages

| Stage | Status | Data |
|-------|--------|------|
| Synthetic ablation (1,000 windows) | COMPLETE | Synthetic |
| Smoke test (real data, small cohort) | DEFERRED to Sprint 10 | Real MetaboNet |
| Pilot (real data, 79 subjects) | DEFERRED to Sprint 10 | Real MetaboNet |
| Full cohort (938 subjects) | DEFERRED to Sprint 10 | Real MetaboNet |

## Synthetic Ablation Configuration
- Dataset: 1,000 deterministic synthetic windows (seed=42)
- Train: 700 | Val: 150 | Test: 150
- Models: Baseline (A), Twin-only (B), Hybrid (C)
- Calibration: Platt (fitted on val set)
- Uncertainty: Bootstrap ensemble (10 members, train only)

## 30m Results

| Config | ROC-AUC | PR-AUC | Recall | F1 | Brier(raw) | Brier(cal) | Features |
|--------|---------|--------|--------|----|------------|------------|----------|
| Baseline | 0.9854 | 0.9426 | 0.8636 | 0.7755 | 0.0377 | 0.0339 | 15 |
| Twin-only | 0.9986 | 0.9930 | 1.0000 | 0.8980 | 0.0265 | 0.0235 | 18 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 0.9362 | 0.0196 | 0.0191 | 33 |

## 60m Results

| Config | ROC-AUC | PR-AUC | Recall | F1 | Brier(raw) | Brier(cal) | Features |
|--------|---------|--------|--------|----|------------|------------|----------|
| Baseline | 0.9823 | 0.9647 | 0.9024 | 0.9024 | 0.0447 | 0.0464 | 15 |
| Twin-only | 0.9991 | 0.9977 | 0.9268 | 0.9500 | 0.0187 | 0.0248 | 18 |
| Hybrid | 1.0000 | 1.0000 | 0.9268 | 0.9620 | 0.0140 | 0.0210 | 33 |

## Twin Trajectory Evaluation (Synthetic)

| Horizon | N | MAE | RMSE | Bias |
|---------|---|-----|------|------|
| 30m | 50 | 7.77 | 8.93 | +0.47 |
| 60m | 50 | 9.10 | 10.78 | -3.25 |

Metrics on numerical glucose scale (units UNKNOWN).

## Bootstrap Uncertainty (Hybrid 30m)
- Probability range: [0.003, 0.997]
- Mean 90% width: 0.030
- Status: bootstrap_ensemble — NOT a calibrated clinical CI

## Synthetic Data Interpretation Note
Perfect AUC is expected here: synthetic labels were derived directly from the same features seen by the models. These results confirm pipeline correctness and causality enforcement, not real-patient generalization performance.

## Resource Usage
- Runtime: 0.94s
- Peak memory: ~173.5 MB
- Leakage audit: CLEAN

## Key Engineering Findings
1. Hybrid module is architecturally complete and causally safe.
2. Calibration correctly uses validation data only.
3. Bootstrap uncertainty avoids test contamination.
4. All 97 tests pass (77 pre-existing + 20 Sprint 9 tests).
5. Full real-data evaluation deferred to Sprint 10.
