# What-If Counterfactual Simulation Engine

> **SAFETY WARNING:** The counterfactual engine explores model-implied trajectories under hypothetical input changes. It does **not** estimate a clinically validated treatment effect and must **not** be used to determine insulin dosing or medical treatment. 

## 1. Architecture

The What-If engine uses the established `TwinEngine` physiological dynamics model but wraps it in a secure simulation boundary:
1. It deep-copies the `TwinState` at time T.
2. It runs a **baseline** simulation assuming no unexpected interventions (no carbs, no bolus).
3. It runs a **counterfactual** simulation from the *exact same initial state*, applying a single hypothetical change (e.g., adding carbs or injecting bolus at a specific offset).
4. It computes the **delta** between the two simulated trajectories.

## 2. Scenario Representation

Scenarios are represented by a typed, immutable `CounterfactualScenario` object.

Supported scenarios:
- `BASELINE`: No hypothetical change.
- `MEAL_PERTURBATION`: Simulates consuming `meal_carbs` at `meal_offset_minutes`.
- `INSULIN_TIMING`: Simulates injecting `insulin_bolus` at `insulin_offset_minutes`.

*Note: The engine explicitly rejects negative inputs and explicitly forbids searching for "optimal" doses.*

## 3. Causality & Missing Data Guarantee

- **Historical Isolation**: The historical state is passed as a copy and is never mutated by the simulation.
- **Future Blindness**: The simulation operates strictly forward in time. It is **impossible** for future actual CGM readings, meals, or insulin events to enter the counterfactual trajectory.
- **Missing Data**: By definition, the future is missing. The engine simulates forward using the `TwinDynamics` default behavior for missing inputs.

## 4. Uncertainty & Safety Boundaries

- **Uncertainty**: Counterfactual uncertainty bounds are currently uncalibrated. Output trajectories reflect deterministic physiological approximations.
- **Safety**: The API requires acknowledgment that results are exploratory and not for clinical decision support. The engine is deliberately restricted from offering "dose recommendations."

## 5. Limitations

- CGM and physical units (insulin, carbs) remain UNKNOWN/unvalidated.
- The model does not capture complex physiological dynamics like exercise, illness, or stress.
- The 70 mg/dL threshold used for evaluation remains a PROVISIONAL engineering limit and has no standing in the counterfactual simulation logic itself.
