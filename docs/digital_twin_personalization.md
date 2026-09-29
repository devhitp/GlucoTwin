# Digital Twin Personalization Contract

## Purpose
This contract establishes how the Digital Twin isolates and adapts to patient-specific physiological traits without violating causal boundaries or introducing target leakage.

## Parameter Distinctions
The architecture must clearly separate:
- **A. Population-level parameters**: Weights and constants derived from the training set, universally applied across all patients (e.g., neural network weights, global decay constants).
- **B. Patient-specific parameters**: Adaptive baselines or traits continuously updated based on an individual patient's real-time data stream.

## Adaptive Personalized Quantities
Potential patient-specific quantities include:
- Patient Glucose Baseline
- Glucose Variability (e.g., historical standard deviation)
- Insulin Sensitivity (empirical response characteristics)
- Meal Absorption (empirical response characteristics)

## Causal Constraints
Every personalized quantity must strictly adhere to the following rules:
1. **Causal Construction**: Computed exclusively from observations occurring at `t <= T`.
2. **No Future Information**: The calculation must never aggregate over the target prediction window.
3. **No Test-Set Leakage**: Personalized parameters must not be pre-calculated over the entire chronological patient record and then fed into a model. They must be constructed dynamically as if time is moving forward.
4. **Inference Availability**: The exact calculation logic used during training must be reproducible at inference time for a new, unseen patient.
