# GlucoTwin

**GlucoTwin** is an AI-powered Digital Twin research prototype being developed for the CRP Digital Twin Challenge 2026.

## Research Objective
The goal is to forecast CGM-defined nocturnal hypoglycemia events 30–60 minutes ahead using physiological/kinetic modeling and machine learning based on a personalized patient baseline.

**Disclaimer:** GlucoTwin is a research/hackathon prototype and is NOT a medical device, diagnostic system, or medical advice tool.

## Dataset
This project uses the MetaboNet public dataset.
The raw dataset contains sensitive research data and must NEVER be committed to this repository.

## Dataset Architecture
- **Core inputs**: CGM, insulin (basal/bolus), carbohydrates, context/time.
- **Optional inputs**: steps, heart rate, EDA, skin temperature (Note: highly sparse in MetaboNet).

## Dataset Setup
1. Obtain the MetaboNet dataset.
2. Place the dataset files in the `data/raw/` directory (e.g. `metabonet_public.parquet`).
3. The pipeline will automatically parse this via `MetaboNetAdapter`.

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
- **Research Labels**: Tracks 30m and 60m future hypoglycemia (<70) and severe hypoglycemia (<54) events (PROVISIONAL threshold).
- **Pipelines**: Integrates CGM, insulin, meals, wearables, context, and patient baselines into a chronologically aligned data structure.
- **Testing Constraints**: Synthetic data handles validations while preserving the requirement for accurate physiological simulations once the real OhioT1DM dataset is linked.

## Sprint 3 — Hypoglycemia Prediction Baseline
Sprint 3 introduces a leakage-safe modeling pipeline focusing on predicting the `future_hypoglycemia_30m` label.
- **Baselines**: Implements a deterministic persistence rule and a tabular LightGBM model.
- **Evaluation Strategy**: Chronological Train/Validation/Test splits preventing data overlap.
- **Interpretability**: Generates calibration, threshold analyses, feature importances, and context-aware error breakdowns.
- **Disclaimer**: Models are verified purely against a synthetic modeling fixture. DO NOT interpret as clinically validated or real OhioT1DM performance.

## Sprint 4 — Real OhioT1DM Integration & Validation
Sprint 4 focuses on dataset integration infrastructure and privacy safeguards.
- **Status**: **REAL OHIO T1DM DATASET NOT AVAILABLE — REAL-DATA VALIDATION PENDING**
- **Infrastructure**: Validated parsing, schema, synchronization, and audit components.
- **Privacy Precautions**: `.gitignore` strictly prohibits committing `data/raw/` and generated artifacts. Audit scripts are restricted to aggregate statistics only.
- **Verification**: If real data is added locally, the pipeline can be safely triggered via `scripts/run_real_pipeline.py`.

## Sprint 5 — First Real OhioT1DM Run
Sprint 5 establishes the readiness criteria for local real-data evaluation and strict privacy protections for research handling.
- **Status**: **BLOCKED: REAL OHIO T1DM DATASET REQUIRES USER-SIDE AUTHORIZED ACCESS/DOWNLOAD**.
- **Acquisition**: A documented protocol (`docs/ohiot1dm_acquisition.md`) mandates downloading the dataset exclusively through authorized PhysioNet channels.
- **Pipeline Constraints**: The repository explicitly blocks fabricated records and refuses to bypass required authentication. 
- **Next Steps**: Once the user acquires the legitimate dataset and places it in `data/raw/`, `scripts/run_real_pipeline.py` will execute the actual preprocessing, labels, temporal splits, and LightGBM model automatically.

## Sprint 6 — Data Qualification & Canonical Adapter
Sprint 6 formally qualifies the local MetaboNet dataset for core GlucoTwin objectives.
- **Adapter**: `MetaboNetAdapter` maps the raw Parquet into the canonical GlucoTwin schema.
- **Qualification**: Over 130M CGM points, 140M insulin events, and 1M+ meal events were memory-safely audited.
- **Outcome**: A "Core Cohort" of 938 subjects (with continuous coverage > 14 days and robust physiological streams) is identified as the primary target for future digital twin experiments.

## Data Safety
Raw and processed research data are excluded from Git to prevent the accidental exposure of sensitive information. Ensure that any downloaded patient data remains exclusively in `data/raw/`.
