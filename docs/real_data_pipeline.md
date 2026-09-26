# GlucoTwin - Real Data Integration

## Objective
Sprint 4 establishes the integration infrastructure for the actual OhioT1DM dataset. Because the dataset contains sensitive research data, it is governed by strict privacy rules and must remain locally hosted.

## Data Provenance
- **Dataset**: OhioT1DM Dataset (PhysioNet)
- **Local Configuration**: Place XML files in `data/raw/`
- **Data Privacy**: The raw and processed directories are intentionally ignored in version control (`.gitignore`). No patient data should ever be committed.

## Infrastructure Ready
The following infrastructure is implemented and validated via synthetic/structural equivalents:
1. **Dataset Discovery**: `DatasetLoader` automatically indexes available XMLs.
2. **Parsing**: `DataParser` translates the OhioT1DM XML schema into canonical Python dataclasses.
3. **Schema Validation**: Validates constraints and handles missingness natively.
4. **Synchronization**: Merges asynchronous signals into 5-minute fixed intervals.
5. **Preprocessing**: Validated in Sprint 2 for causal safety.
6. **Modeling**: Validated in Sprint 3.

## Status
**REAL OHIO T1DM DATASET NOT AVAILABLE — REAL-DATA VALIDATION PENDING**
Currently, only the `559.xml` synthetic fixture exists. Attempting to run the real pipeline script (`scripts/run_real_pipeline.py`) correctly identifies that the real data is missing and halts execution to prevent fabricating real-data evaluation results.

Once the actual dataset is placed in `data/raw/`, the `audit_real_dataset.py` will automatically produce aggregate validation metrics.
