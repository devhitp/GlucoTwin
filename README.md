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

## Data Safety
Raw and processed research data are excluded from Git to prevent the accidental exposure of sensitive information. Ensure that any downloaded patient data remains exclusively in `data/raw/`.
