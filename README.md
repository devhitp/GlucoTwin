# GlucoTwin

An AI-powered Digital Twin for Type 1 Diabetes that combines continuous glucose, insulin and meal data with personalized physiological modeling to forecast near-term hypoglycemia risk and explore hypothetical future scenarios.

## Problem
People with Type 1 Diabetes face constant uncertainty regarding near-term hypoglycemia risk. Real-world physiological data (CGM, insulin, meals) is highly complex, nonlinear, and difficult to project safely without domain knowledge.

## Solution
GlucoTwin solves this by creating a personalized Digital Twin for each subject based on causal physiological dynamics. It embeds deterministic kinetic modeling (Twin parameters, insulin-on-board, meal absorption) into a Hybrid ML engine (LightGBM) to significantly boost predictive capability and allow counterfactual "What-If" exploration.

## Dataset
Developed and evaluated on the **MetaboNet** public dataset (Core cohort: 938 subjects, ~21M windows).
*Note: Due to privacy restrictions, raw patient data is explicitly `.gitignore`d and NEVER included in the repository. The provided demo uses deterministic synthetic data.*

## Architecture
1. **Digital Twin**: Models personalized baseline glucose, insulin action, and carbohydrate decay.
2. **Hybrid ML**: LightGBM model combining historical tabular features with Twin-projected future states.
3. **What-If Simulation**: A causal engine allowing safe perturbation of hypothetical meals or insulin boluses from a read-only current state.

![Architecture Diagram](https://via.placeholder.com/800x400.png?text=Data+->+Digital+Twin+->+Hybrid+Prediction+->+What-If+Simulation)

## Evaluation
The architecture was evaluated via a strict patient-held-out protocol (no leakage, causal streaming feature extraction).
- **Metrics Evaluated**: ROC-AUC, PR-AUC, F1, Recall, Precision, MAE (for trajectory).
- **Personalization**: Reduced Twin forecast MAE from ~58 to ~15.
- **Model**: Hybrid config outperformed Baseline configs safely.

## Demo Instructions
A complete, end-to-end interactive dashboard is provided using a safe, synthetic demo fixture.

**To run the demo:**
```bash
pip install -r requirements.txt
pip install streamlit
streamlit run scripts/run_dashboard.py
```

## Limitations & Disclaimer
⚠️ **RESEARCH PROTOTYPE ONLY**
- **NOT a medical device.**
- **NOT clinical validation.**
- **Does NOT provide dosing recommendations.**
- CGM and insulin/carb units have not been independently confirmed.
- Uncertainty calibration is for research benchmarking, not medical confidence.

## Repository Structure
- `data/`: Raw/processed data (Git-ignored).
- `src/glucotwin/`: Main application logic (data parsing, twin engine, model, counterfactual simulator).
- `scripts/`: Evaluation pipelines and dashboard runner.
- `tests/`: Extensive automated unit and integration tests.
- `docs/`: Validation, architecture, and qualification reports.
