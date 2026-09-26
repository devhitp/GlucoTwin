# GlucoTwin Sprint 3 - Hypoglycemia Prediction Baseline

## Overview
This document describes the modeling architecture and baseline implementation for Sprint 3 of the GlucoTwin digital twin prototype. The primary goal is to predict future hypoglycemia events within 30-minute and 60-minute horizons using canonical physiological data.

## Target Definitions
- `future_hypoglycemia_30m`: 1 if glucose drops < 70 mg/dL within the next 30 minutes, 0 otherwise.
- `future_hypoglycemia_60m`: 1 if glucose drops < 70 mg/dL within the next 60 minutes, 0 otherwise.

## Input Features
The pipeline extracts model-safe features excluding identifiers and target variants:
- **CGM Metrics**: `glucose_current`, `glucose_roc_5m`
- **Insulin**: `insulin_bolus_last_30m`
- **Meals**: `carbs_last_30m`
- **Context**: `is_night_clock_based`

## Leakage Prevention
- Target labels strictly evaluate > T.
- Timestamps and identifiers are sanitized and excluded before model input.
- Splitting strategy uses chronological splits (Train, Validation, Test) to ensure time series integrity without boundary leakage.

## Persistence Baseline
A deterministic heuristic that relies only on:
1. Current glucose near threshold.
2. Negative rate of change (ROC).
Provides a reference point before introducing machine learning.

## LightGBM Classifier
- Tabular Machine Learning Baseline using `LGBMClassifier`.
- Handled with `class_weight='balanced'` for imbalanced event classification.
- Robust handling of missing and infinite values.
- Hyperparameters are intentionally conservative to establish a reliable baseline before complex deep sequence modeling.

## Evaluation Metrics
- ROC-AUC
- PR-AUC (Average Precision)
- F1-Score
- Specificity
- Balanced Accuracy
- Brier Score (for Calibration)

## Threshold Analysis
A configurable set of thresholds [0.1, 0.3, 0.5, 0.7, 0.9] is tested on the validation set to evaluate optimal precision and recall tradeoffs, simulating tuning specific alerts for personalized requirements.

## Real Data Limitations
**SYNTHETIC FIXTURE RESULTS ONLY — NOT REAL OHIO T1DM PERFORMANCE**
Because the real OhioT1DM dataset is unavailable, the pipeline relies on synthetic modeling fixtures. The reported evaluation metrics validate pipeline integrity, temporal safety, and artifact correctness, but DO NOT represent clinical or medical validity.
