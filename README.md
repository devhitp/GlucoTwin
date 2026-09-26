# GlucoTwin

**GlucoTwin** is an AI-powered Digital Twin research prototype being developed for the CRP Digital Twin Challenge 2026.

## Research Objective
The goal is to forecast CGM-defined nocturnal hypoglycemia events 30–60 minutes ahead using physiological/kinetic modeling and machine learning based on a personalized patient baseline.

**Disclaimer:** GlucoTwin is a research/hackathon prototype and is NOT a medical device, diagnostic system, or medical advice tool.

## Dataset
This project uses the OhioT1DM dataset from PhysioNet.
The raw dataset contains sensitive research data and must NEVER be committed to this repository.

## Dataset Setup
1. Obtain the OhioT1DM dataset through its legitimate public access process on PhysioNet.
2. Place the dataset files in the `data/raw/` directory, or set the dataset path using the `GLUCOTWIN_DATA_PATH` environment variable.
3. The dataset should contain XML files for each patient (as provided by the official OhioT1DM release).

## Project Structure
- `data/`: Contains raw and processed data (ignored in git except for this readme).
- `src/glucotwin/`: The main application package.
- `tests/`: Automated tests and synthetic test data.
- `scripts/`: Utility scripts, e.g. for inspecting the dataset.
- `docs/`: Architecture documentation.

## Sprint 1
Sprint 1 establishes a clean, reliable foundation for dataset loading, parsing, schema definition, validation, and synchronization. It sets up the core data pipeline before any machine learning or digital twin modeling is implemented.

## Sprint 2 — Data Preprocessing & Feature Engineering
Sprint 2 transforms canonical records into a leakage-safe model-ready dataset. 
- **Causal Features**: Ensures backward-looking boundaries to avoid future data leakage. 
- **Research Labels**: Tracks 30m and 60m future hypoglycemia (<70mg/dL) and severe hypoglycemia (<54mg/dL) events.
- **Pipelines**: Integrates CGM, insulin, meals, wearables, context, and patient baselines into a chronologically aligned data structure.
- **Testing Constraints**: Synthetic data handles validations while preserving the requirement for accurate physiological simulations once the real OhioT1DM dataset is linked.

## Sprint 3 — Hypoglycemia Prediction Baseline
Sprint 3 introduces a leakage-safe modeling pipeline focusing on predicting the `future_hypoglycemia_30m` label.
- **Baselines**: Implements a deterministic persistence rule and a tabular LightGBM model.
- **Evaluation Strategy**: Chronological Train/Validation/Test splits preventing data overlap.
- **Interpretability**: Generates calibration, threshold analyses, feature importances, and context-aware error breakdowns.
- **Disclaimer**: Models are verified purely against a synthetic modeling fixture. DO NOT interpret as clinically validated or real OhioT1DM performance.

## Data Safety
Raw and processed research data are excluded from Git to prevent the accidental exposure of sensitive information. Ensure that any downloaded patient data remains exclusively in `data/raw/`.
