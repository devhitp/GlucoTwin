# Hybrid Feature Contract

> All features must be computed from observations at or before time T.
> Twin forecast values at T+30 and T+60 are **model-simulated** — they are NOT real future observations.

## Feature Definitions

| Feature | Definition | Source | Horizon | Causal | Missing behavior |
|---------|-----------|--------|---------|--------|-----------------|
| `twin_glucose_current` | G(t): Twin-modeled glucose at T | TwinState | T | ✅ | NaN |
| `twin_insulin_action` | X(t): Insulin-action state | TwinState | T | ✅ | NaN |
| `twin_insulin_state` | I(t): Insulin exposure state | TwinState | T | ✅ | NaN |
| `twin_meal_state` | M(t): Meal absorption state | TwinState | T | ✅ | NaN |
| `twin_context_state` | S(t): Contextual state | TwinState | T | ✅ | NaN |
| `twin_glucose_t5` | Simulated G(T+5m) | Forecast | Simulated | ✅ | NaN |
| `twin_glucose_t15` | Simulated G(T+15m) | Forecast | Simulated | ✅ | NaN |
| `twin_glucose_t30` | Simulated G(T+30m) | Forecast | Simulated | ✅ | NaN |
| `twin_glucose_t60` | Simulated G(T+60m) | Forecast | Simulated | ✅ | NaN |
| `twin_slope_30m` | (G(T+30)-G(T)) / 30 | Forecast | Simulated | ✅ | NaN |
| `twin_slope_60m` | (G(T+60)-G(T)) / 60 | Forecast | Simulated | ✅ | NaN |
| `twin_delta_15m` | G(T+15) - G(T) | Forecast | Simulated | ✅ | NaN |
| `twin_delta_30m` | G(T+30) - G(T) | Forecast | Simulated | ✅ | NaN |
| `twin_delta_60m` | G(T+60) - G(T) | Forecast | Simulated | ✅ | NaN |
| `twin_residual_current` | observed_glucose(T) - Twin G(T) | State vs Obs | T | ✅ | NaN |
| `twin_has_long_gap` | 1 if LONG_GAP flag active | QualityFlag | T | ✅ | 0.0 |
| `twin_has_missing_glucose` | 1 if MISSING_GLUCOSE flag | QualityFlag | T | ✅ | 0.0 |
| `twin_uncertainty_calibrated` | 1 if uncertainty is calibrated | Metadata | T | ✅ | 0.0 |

## Critical Causality Rules

1. **Simulated ≠ observed.** `twin_glucose_t30` is a model-simulated value produced by the Twin dynamics engine. It is **never** the actual glucose observed at T+30.
2. **Residuals are at T only.** `twin_residual_current` compares observed glucose at T against the Twin's modeled glucose at T. It never uses future actual glucose.
3. **No future lookback.** No feature uses observations from `(T, T+horizon]`.
