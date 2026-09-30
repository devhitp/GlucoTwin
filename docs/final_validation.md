# GlucoTwin — Final Validation

## Project
**GlucoTwin** is a research prototype of an AI-powered Digital Twin for Type 1 Diabetes. It combines continuous glucose, insulin, and meal data with personalized physiological modeling to forecast near-term hypoglycemia risk and explore hypothetical future scenarios.

**Target Event:** Hypoglycemia (glucose < 70 mg/dL).
**Prediction Horizons:** 30-minute and 60-minute forecasts.

## Dataset
- **MetaboNet** (observational research dataset).
- **Row count:** ~21 million 5-minute CGM windows.
- **Subject count:** 938 core subjects.
- **Units status:** CGM units are unverified (provisional threshold of 70 used). Insulin and carbohydrate unit provenance is unresolved.
- **Timestamp status:** Known anomaly in time-of-day alignment; physiological response remains consistent.

## Architecture
Data → Preprocessing → Digital Twin (Personalized) → Hybrid ML Prediction → Explainability → What-If Simulation

## Digital Twin
- **State:** Tracks current glucose, insulin action, insulin-on-board, and meal absorption state.
- **Parameters:** Personalized per-patient using causal history (basal glucose estimation).
- **Dynamics:** Deterministic physiological equations. 
- **Missing Data:** Does NOT interpolate missing glucose values longer than 15 minutes. Future data is strictly prohibited during personalization and state updates.

## Prediction
Models evaluated:
- **Baseline:** LightGBM using only standard feature sets.
- **Twin-only:** LightGBM using only physiological variables and projected trajectory.
- **Hybrid:** LightGBM combining Baseline + Twin-only features.

## Evaluation
- **Split:** Patient-held-out (55 train / 11 val / 13 test for the pilot). No patient overlaps between splits.
- **Seed:** 42
- **Metrics:** ROC-AUC, PR-AUC, Precision, Recall, F1, Specificity, Brier Score.
- **Calibration:** Platt scaling fit exclusively on validation data.
- **Uncertainty:** Bootstrap ensembling (documented as research uncertainty, not a medical confidence interval).

## What-If
- **Scenario model:** Allows hypothetical perturbations to meal (carbohydrates) and insulin (bolus) inputs.
- **Causal isolation:** Simulations are run forward from an identical twin state copy. No historical observations are modified. No future observations are leaked.
- **Outputs:** Delta trajectory (Counterfactual minus Baseline).
- **Limitations:** Only for hypothetical research simulation. NOT a dosing recommendation.

## Dashboard
- **Architecture:** Streamlit-based local UI running end-to-end inference over a deterministic research demo fixture. 
- **Demo flow:** Current State -> Digital Twin Forecast -> Risk Estimation -> What-If Scenario.
- **Data:** Uses a synthetic physiological sequence safe for git tracking. Real MetaboNet data is git-ignored.

## Limitations (Critical)
- **Unit provenance:** CGM and insulin/carb units have not been independently confirmed.
- **Timestamp anomaly:** MetaboNet contains an unresolved time-of-day discrepancy.
- **Observational dataset:** The system was developed on retrospective data.
- **Clinical validation:** ZERO clinical validation. The system is a research prototype.
- **Uncertainty calibration:** Not clinically calibrated.
- **Cohort completion status:** The full 938-subject evaluation is restricted by computational time limits (~38 hours runtime on this hardware profile), so the pilot 79-subject results form the current benchmark.
- **Computational constraints:** LightGBM training requires batched memory handling (implemented in Sprint 12) for the full 938-subject dataset.
