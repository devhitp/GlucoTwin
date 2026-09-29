# Digital Twin Input Contract

## Purpose
This contract defines the exact inputs consumed by the Sprint 8 Digital Twin architecture. The Twin must gracefully degrade and remain structurally valid even when optional signals are missing.

## Input Categories and Priority

### A. REQUIRED: CORE OBSERVATIONS
These inputs are mandatory for the Digital Twin to initialize and update its state. If these are systematically missing or invalid, the Twin cannot operate.
- **`timestamps`**: Sequence of observation times (subject to the Timestamp Handling Contract).
- **`glucose`**: Current CGM reading.
- **`glucose_history`**: Recent trailing window of glucose observations.
- **`glucose_roc`**: Glucose rate-of-change (derivative).
- **`insulin`**: Total insulin activity or proxy.
- **`basal`**: Basal insulin rate/dose.
- **`bolus`**: Bolus insulin dose.
- **`carbohydrates`**: Dietary carbohydrate/meal intake.

### B. REQUIRED: CONTEXT
Calendar-dependent features must be interpreted loosely and solely as proxy variables for circadian or behavioral rhythms, due to unknown timestamp provenance.
- **`time_since_meal`**: Elapsed physiological time since last recorded carbs.
- **`time_since_bolus`**: Elapsed physiological time since last recorded bolus.
- **`hour_of_day`**: Proxy for circadian rhythm (unverified timezone alignment).

### C. OPTIONAL: WEARABLE SIGNALS
The MetaboNet dataset contains extremely sparse wearable coverage. The Digital Twin is explicitly designed *not* to rely on these.
- **`heart_rate`**
- **`galvanic_skin_response`** (EDA)
- **`skin_temp`**
- **`steps`** (Activity)

## Missing Optional Input Behavior
- **Imputation**: Do NOT invent or impute missing wearable signals. 
- **Masking**: If an optional signal is absent, it must be represented as a masked/null tensor or explicit absence indicator.
- **Structural Validity**: The Digital Twin must support Mode A (Core physiological inputs only) and Mode B (Core inputs + available wearable signals). The underlying equations or neural architecture must bypass the wearable embeddings gracefully when absent.
