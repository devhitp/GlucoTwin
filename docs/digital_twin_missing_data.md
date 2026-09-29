# Digital Twin Missing Data Contract

## Purpose
This contract defines the strict policies for handling missing, invalid, or sparse data within the Digital Twin state updates and predictions.

## Definitions
- **Missing**: An expected measurement at a given physiological timestep does not exist.
- **Unavailable**: A feature (e.g., wearables) was never recorded for this patient.
- **Invalid**: A recorded measurement violates physical/physiological limits (e.g., negative glucose).
- **Stale**: A measurement exists but the elapsed time since it was recorded exceeds physiological relevance.
- **Zero-valued**: An explicit record of 0.0 (e.g., 0.0 carbs). This is NOT missing data.

## Policies

### 1. Missing CGM
- **Short gaps (<= 15 minutes)**: May be linearly interpolated to maintain continuous state updates.
- **Long gaps (> 15 minutes)**: Must NOT be interpolated. The Digital Twin state must reflect the uncertainty of a disconnected trajectory.
- *(Note: 15 minutes is currently an unresolved design parameter and subject to future physiological tuning).*

### 2. Missing Insulin
- Do NOT automatically treat missing insulin rows as zero-valued. 
- If a background basal rate is known, it may be assumed to continue, provided the elapsed time does not exceed reasonable pump suspension durations.
- Missing bolus events are treated as structurally absent. Do NOT impute bolus deliveries.

### 3. Missing Meal/Carb Data
- Do NOT automatically treat missing carb rows as zero-valued.
- Treat meals as episodic events. If no meal is recorded, the contextual variable `time_since_meal` increments, but we do not impute meal events.

### 4. Missing Wearable Data
- As defined in the Input Contract, wearable absence is treated as "Unavailable".
- No imputation is permitted.

### 5. Missing Patient Baseline Information
- If historical data is insufficient to compute personalized baseline metrics (e.g., `patient_glucose_baseline`), the Twin must fall back to population-level priors.
