# Prediction Horizon Contract

## Purpose
This contract establishes the fixed forecasting boundaries for the Sprint 8 Digital Twin architecture.

## Time Structure
- **Timestep Resolution**: 5 minutes (aligned to CGM sampling).
- **Trajectory Length**: Predictions are strictly bounded to the primary and secondary horizons.

## Horizons
1. **Primary Horizon**: 30 minutes (6 timesteps).
2. **Secondary Horizon**: 60 minutes (12 timesteps).

## Target Interpretation
The prediction targets map to the probability of the glucose trajectory crossing the provisional hypoglycemia threshold within the defined horizon.
- Labels are defined as binary outcomes `[0, 1]` based on whether `min(G(t+1)...G(t+H)) < HYPO_THRESHOLD`.
- The evaluation targets established in Sprint 7/7.5 remain strictly unchanged to ensure longitudinal comparability.

## Handling of Missing Future Observations (Evaluation)
If future observations within the required horizon are missing from the ground-truth dataset, the label is considered NaN (unverifiable). The evaluation metric engine explicitly excludes these windows.
