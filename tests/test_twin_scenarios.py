"""
Synthetic scenario tests for the Digital Twin engine.
All scenarios use deterministic synthetic data, never real patient records.
DO NOT assert that values represent clinical physiology.
"""
import pytest
import math
from datetime import datetime, timedelta

from src.glucotwin.twin import (
    TwinState, TwinObservation, TwinParameters,
    TwinInitializer, TwinEngine, TwinTrajectory
)


T0 = datetime(2026, 1, 1, 8, 0)


def _engine_at(glucose=110.0, t=None):
    t = t or T0
    state = TwinState(t, glucose, 0.0, 0.0, 0.0, 0.0)
    return TwinEngine(state)


def _obs(minutes, glucose=None, bolus=None, carbs=None, basal=None, **kw):
    return TwinObservation(
        timestamp=T0 + timedelta(minutes=minutes),
        glucose=glucose,
        bolus=bolus,
        carbohydrates=carbs,
        basal=basal,
        **kw
    )


# ── Scenario A: Stable glucose, no interventions ──────────────────────────────
def test_scenario_a_stable_glucose():
    """When glucose is observed every step, state tracks observation exactly."""
    engine = _engine_at(110.0)
    for i in range(1, 7):
        engine.update(_obs(i * 5, glucose=110.0))
    assert engine.current_state.glucose == 110.0


# ── Scenario B: Recorded meal event ──────────────────────────────────────────
def test_scenario_b_meal_event():
    engine = _engine_at(100.0)
    engine.update(_obs(5, glucose=None, carbs=50.0))
    assert engine.current_state.meal_state > 0.0


# ── Scenario C: Recorded insulin event ───────────────────────────────────────
def test_scenario_c_insulin_event():
    engine = _engine_at(180.0)
    engine.update(_obs(5, glucose=None, bolus=5.0))
    assert engine.current_state.insulin_state > 0.0


# ── Scenario D: Meal + insulin interaction ────────────────────────────────────
def test_scenario_d_meal_and_insulin():
    engine = _engine_at(100.0)
    engine.update(_obs(5, glucose=None, carbs=50.0, bolus=5.0))
    state = engine.current_state
    assert state.meal_state > 0.0
    assert state.insulin_state > 0.0


# ── Scenario E: Missing wearable data ────────────────────────────────────────
def test_scenario_e_missing_wearables():
    engine = _engine_at(110.0)
    obs = _obs(5, glucose=110.0, heart_rate=None, steps=None, galvanic_skin_response=None)
    state = engine.update(obs)
    assert state.glucose == 110.0


# ── Scenario F: Short glucose gap (<= 15m) ───────────────────────────────────
def test_scenario_f_short_gap():
    engine = _engine_at(110.0)
    # 10-minute gap (missing glucose)
    obs = TwinObservation(timestamp=T0 + timedelta(minutes=10), glucose=None)
    state = engine.update(obs)
    assert "LONG_GAP" not in state.quality_flags
    assert not math.isnan(state.glucose)
    assert state.is_valid


# ── Scenario G: Long glucose gap (> 15m) ─────────────────────────────────────
def test_scenario_g_long_gap():
    engine = _engine_at(110.0)
    obs = TwinObservation(timestamp=T0 + timedelta(minutes=20), glucose=None)
    state = engine.update(obs)
    assert "LONG_GAP" in state.quality_flags


# ── Scenario H: Future observation supplied accidentally ──────────────────────
def test_scenario_h_future_obs_rejected():
    engine = _engine_at(110.0)
    engine.update(_obs(5, glucose=115.0))
    # Try to feed an observation from the past (before current state)
    past_obs = TwinObservation(timestamp=T0, glucose=100.0)
    with pytest.raises(ValueError):
        engine.update(past_obs)


# ── Scenario I: Timestamp reversal ───────────────────────────────────────────
def test_scenario_i_timestamp_reversal_in_history():
    h = [
        TwinObservation(timestamp=T0, glucose=110.0),
        TwinObservation(timestamp=T0 - timedelta(minutes=5), glucose=105.0),  # reversal
    ]
    with pytest.raises(ValueError):
        TwinInitializer.initialize(h)


# ── Scenario J: Extreme parameter input ──────────────────────────────────────
def test_scenario_j_invalid_parameter_negative_rates():
    """Engine can be constructed but must not produce NaN from extreme params."""
    p = TwinParameters(
        glucose_drift_coeff=1.0,
        insulin_clearance_rate=0.9999,
        meal_absorption_rate=0.9999,
        carb_conversion_coeff=100.0,
    )
    state = TwinState(T0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state, params=p)
    obs = TwinObservation(timestamp=T0 + timedelta(minutes=5), glucose=None, carbohydrates=50.0)
    new_state = engine.update(obs)
    assert new_state.is_valid  # Should not blow up to NaN/Inf


# ── Unit metadata contract ────────────────────────────────────────────────────
def test_unit_metadata_remains_unknown():
    from src.glucotwin.config.clinical import get_glucose_unit
    assert get_glucose_unit() == "unknown"


# ── Provisional threshold stays outside Twin ─────────────────────────────────
def test_provisional_threshold_not_in_twin_dynamics():
    """TwinDynamics must not import or reference clinical threshold."""
    import inspect, src.glucotwin.twin.dynamics as dyn
    src_code = inspect.getsource(dyn)
    assert "get_hypo_threshold" not in src_code
    assert "hypo_th" not in src_code


# ── Personalization causality ─────────────────────────────────────────────────
def test_personalization_causality():
    """Personalized params derived purely from history, not future."""
    from src.glucotwin.twin.personalization import PersonalizationEngine
    history = [TwinObservation(T0 + timedelta(minutes=i*5), glucose=100.0 + i) for i in range(20)]
    params = PersonalizationEngine.adapt(history)
    assert params.is_personalized
    assert params.baseline_glucose > 0


# ── Fallback to default parameters ───────────────────────────────────────────
def test_fallback_to_default_params():
    from src.glucotwin.twin.personalization import PersonalizationEngine
    # Fewer than minimum data requirement → falls back to population defaults
    history = [TwinObservation(T0, glucose=100.0)]
    params = PersonalizationEngine.adapt(history)
    assert not params.is_personalized
    default = TwinParameters.default_population_params()
    assert params.baseline_glucose == default.baseline_glucose


# ── Wearable partial availability ─────────────────────────────────────────────
def test_wearable_partial_availability():
    engine = _engine_at(110.0)
    obs = TwinObservation(
        timestamp=T0 + timedelta(minutes=5),
        glucose=110.0,
        heart_rate=72.0,  # only HR available
        steps=None,
        galvanic_skin_response=None,
    )
    state = engine.update(obs)
    assert state.glucose == 110.0


# ── Numerical stability ───────────────────────────────────────────────────────
def test_numerical_stability_many_steps():
    engine = _engine_at(100.0)
    for i in range(1, 200):
        obs = TwinObservation(timestamp=T0 + timedelta(minutes=i*5), glucose=None)
        state = engine.update(obs)
        assert state.is_valid
        assert not math.isinf(state.glucose)


# ── Forecast does not consume future observations ────────────────────────────
def test_forecast_does_not_change_state():
    engine = _engine_at(110.0)
    state_before = engine.current_state
    _ = engine.forecast(30)
    _ = engine.forecast(60)
    assert engine.current_state.timestamp == state_before.timestamp
    assert engine.current_state.glucose == state_before.glucose


# ── Uncertainty metadata ──────────────────────────────────────────────────────
def test_uncertainty_metadata():
    engine = _engine_at(110.0)
    traj = engine.forecast(30)
    assert traj.uncertainty.get("status") == "not_calibrated"


# ── Serialization/deserialization (basic) ─────────────────────────────────────
def test_state_is_immutable():
    state = TwinState(T0, 110.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(Exception):
        state.glucose = 200.0  # frozen dataclass → AttributeError
