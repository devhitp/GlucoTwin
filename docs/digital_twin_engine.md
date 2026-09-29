# Digital Twin Engine — Architecture & Reference

> **DISCLAIMER:** This Digital Twin is a research simulation engine and is not clinically validated.
> All outputs are model-simulated glucose trajectories and must not be interpreted as medical advice.

## Known Limitations & Unit Caveats

| Item | Status |
|------|--------|
| CGM unit | **UNKNOWN** — assumed numerical scale, NOT confirmed mg/dL |
| Insulin unit | **UNKNOWN** |
| Carbohydrate unit | **UNKNOWN** |
| Hypoglycemia threshold (70) | **PROVISIONAL** — engineering placeholder only |
| Timestamp anomaly (2025–2027 dates) | **UNRESOLVED** — future dates exist with no authoritative provenance |

---

## 1. Architecture

```
src/glucotwin/twin/
├── __init__.py            — Public API exports
├── observations.py        — TwinObservation: typed, unit-agnostic input
├── state.py               — TwinState: immutable physiological state vector
├── parameters.py          — TwinParameters: population & personalized params
├── dynamics.py            — TwinDynamics: discrete-time step engine
├── initializer.py         — TwinInitializer: causal burn-in initializer
├── simulator.py           — TwinEngine: update + forecast interface
├── trajectory.py          — TwinTrajectory: typed forecast output
└── personalization.py     — PersonalizationEngine: safe adaptive params
```

The Twin module is deliberately isolated from `src/glucotwin/modeling/` (LightGBM baselines). It does not share parameters, labels, or any prediction targets.

---

## 2. State Representation

`TwinState` is a frozen dataclass carrying:

| Field | Engineering meaning |
|-------|---------------------|
| `glucose` | G(t): current modeled glucose state |
| `insulin_action` | X(t): latent insulin-action / glucose-disposal state |
| `insulin_state` | I(t): latent insulin exposure / active insulin state |
| `meal_state` | M(t): latent meal absorption state |
| `context_state` | S_sleep(t): contextual state (circadian, nocturnal) |
| `uncertainty` | Dict with calibration metadata |
| `quality_flags` | List of active quality flags |

> These are engineering constructs. They do NOT correspond to validated physiological measurements.

---

## 3. Parameters

`TwinParameters` are model coefficients, **not** clinically validated constants:

| Parameter | Role |
|-----------|------|
| `baseline_glucose` | Reference level for mean-reversion drift |
| `glucose_drift_coeff` | Strength of return-to-baseline force |
| `insulin_sensitivity` | Coupling of insulin action to glucose change |
| `insulin_clearance_rate` | Decay rate of insulin state |
| `meal_absorption_rate` | Decay rate of meal absorption state |
| `carb_conversion_coeff` | Carbohydrate-to-glucose conversion scale |
| `is_personalized` | Boolean flag: True if adapted from patient history |

Population defaults ship with `TwinParameters.default_population_params()`.

---

## 4. Observation Model

`TwinObservation` explicitly represents each physiological signal's presence:

- `None` means **structurally absent** — never auto-converted to zero.
- Missing glucose triggers simulation-forward behavior, not imputation.
- Missing wearables (HR, GSR, skin_temp, steps) are silently ignored; they do not break the engine.

---

## 5. Initialization

`TwinInitializer.initialize(history)` is the causal entry point:

1. Requires at least one valid glucose observation in `history`.
2. Burns in all historical observations in chronological order.
3. **Never consumes observations after the last history timestamp.**
4. Raises `ValueError` on: empty history, all-missing glucose, timestamp reversal.
5. Falls back to population-level parameters when patient history is insufficient.

---

## 6. Dynamics

`TwinDynamics.step()` implements a **physiological-inspired discrete-time model**:

```
# Meal absorption state
meal_decay = meal_absorption_rate × M(t)
M(t+1) = M(t) − meal_decay + new_carbohydrates
M(t+1) = max(0, M(t+1))

# Insulin exposure state
insulin_input = bolus + basal × (dt_minutes / 60)
I(t+1) = I(t) − insulin_clearance_rate × I(t) + insulin_input
I(t+1) = max(0, I(t+1))

# Insulin action state (delay compartment)
X(t+1) = X(t) − insulin_clearance_rate × X(t) + insulin_clearance_rate × I(t)
X(t+1) = max(0, X(t+1))

# Glucose state
if glucose observed:
    G(t+1) = observed_glucose        ← anchored to sensor
else:
    drift = glucose_drift_coeff × (baseline_glucose − G(t))
    meal_effect = carb_conversion_coeff × meal_decay
    insulin_effect = insulin_sensitivity × X(t)
    G(t+1) = G(t) + drift + meal_effect − insulin_effect
    G(t+1) = max(10, G(t+1))         ← numerical floor
```

> This is a **physiological-inspired engineering approximation.** It is not a validated pharmacokinetic model. It does not reproduce real human physiology.

---

## 7. Update Loop

`TwinEngine.update(observation)`:
1. Validates timestamp is ≥ current state timestamp (raises on reversal).
2. Calculates elapsed `dt` in minutes.
3. Calls `TwinDynamics.step()`.
4. Emits quality flags: `MISSING_GLUCOSE`, `LONG_GAP` (>15 min gap).
5. Returns new immutable `TwinState`.

**Determinism guarantee:** Same state + same observation + same parameters → same resulting state.

---

## 8. Forecasting

`TwinEngine.forecast(horizon_minutes)` returns a `TwinTrajectory`:
- Supported horizons: **30 minutes** (6 steps), **60 minutes** (12 steps).
- Forecast uses zero new inputs (no fabricated future meals or insulin).
- Does **not** mutate the engine's internal state.
- Returns structured `uncertainty = {"status": "not_calibrated"}` until statistical calibration is implemented.

---

## 9. Personalization

`PersonalizationEngine.adapt(history)` safely estimates patient-specific baseline glucose:

- Requires ≥ 10 valid glucose observations (configurable).
- Uses **only past observations** — never future data.
- Falls back to population defaults if insufficient history.
- Returns a `TwinParameters` with `is_personalized=True`.

---

## 10. Missing Data

| Condition | Behavior |
|-----------|----------|
| Missing glucose (≤15 min gap) | Simulate forward without interpolation |
| Missing glucose (>15 min gap) | Simulate forward + flag `LONG_GAP` |
| Missing insulin/bolus | Treated as no new event (not zero-valued) |
| Missing meals/carbs | Treated as no new event (not zero-valued) |
| Missing wearables | Ignored — engine runs in Mode A (Core only) |

---

## 11. Wearable Degradation

- **Mode A** (Core only): All wearables absent — full function.
- **Mode B** (Core + wearables): Wearable data stored in observations for future extensions.
- In Sprint 8, wearables do not directly influence glucose dynamics.

---

## 12. Uncertainty

Uncertainty metadata is carried in `TwinState.uncertainty` and `TwinTrajectory.uncertainty`:

```json
{"status": "not_calibrated"}
```

Future sprints will implement conformal prediction or ensemble-based intervals.  
**No fake confidence percentages are produced.**

---

## 13. Quality Flags

| Flag | Meaning |
|------|---------|
| `INITIALIZED` | State freshly created from initialization |
| `MISSING_GLUCOSE` | Most recent observation has no valid glucose |
| `LONG_GAP` | More than 15 minutes elapsed without glucose observation |

---

## 14. Numerical Stability Safeguards

- Meal state and insulin states clipped to `max(0.0, ...)` — preventing negative exposure.
- Simulated glucose floored at `10.0` to prevent physiologically impossible negatives.
- Timestamp reversal raises `ValueError` — never silently processed.

---

## 15. Future Extensions (Not Implemented in Sprint 8)

- What-If counterfactual simulation engine
- Conformal prediction / ensemble uncertainty
- LSTM/TCN hybrid dynamics
- SHAP explainability layer
- Clinical alerting
- Dashboard
- Autonomous insulin suggestion (explicitly prohibited)
