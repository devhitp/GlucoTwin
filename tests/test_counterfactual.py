"""
Sprint 10 counterfactual engine tests — Phase 19 scenarios A-L.
All scenarios use deterministic synthetic state. No real patient data.
"""
import pytest
import copy
import math
from datetime import datetime, timedelta

from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.parameters import TwinParameters
from src.glucotwin.counterfactual import (
    CounterfactualScenario,
    ScenarioType,
    CounterfactualSimulator,
    CounterfactualResult,
    validate_no_future_data_in_scenario,
)

T0 = datetime(2026, 1, 1, 9, 0)


def _state(g=110.0, ia=0.5, is_=1.0, ms=5.0):
    return TwinState(T0, g, ia, is_, ms, 0.0)


def _sim():
    return CounterfactualSimulator(TwinParameters.default_population_params())


# ── A: Baseline scenario reproduces baseline trajectory ─────────────────────
def test_a_baseline_scenario_is_deterministic():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.baseline()
    res1 = sim.simulate(state, scenario, 30)
    res2 = sim.simulate(state, scenario, 30)
    assert res1.baseline_glucose == res2.baseline_glucose
    assert res1.counterfactual_glucose == res2.counterfactual_glucose
    assert all(d == 0.0 for d in res1.delta_glucose)


# ── B: Meal perturbation changes counterfactual vs baseline ─────────────────
def test_b_meal_perturbation_changes_trajectory():
    state = _state(g=100.0, ms=0.0)
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(carbs=50.0, offset_minutes=5.0)
    res = sim.simulate(state, scenario, 30)
    # Meal should raise meal state, eventually raising glucose in counterfactual
    assert res.counterfactual_glucose != res.baseline_glucose
    # Delta should eventually be positive (meal raises glucose)
    assert any(d > 0 for d in res.delta_glucose)


# ── C: Insulin timing perturbation changes counterfactual ───────────────────
def test_c_insulin_timing_perturbation_changes_trajectory():
    state = _state(g=150.0, ia=0.0, is_=0.0)
    sim = _sim()
    scenario = CounterfactualScenario.insulin_timing(bolus=5.0, offset_minutes=5.0)
    res = sim.simulate(state, scenario, 30)
    assert res.counterfactual_glucose != res.baseline_glucose


# ── D: Determinism — same state + same scenario = identical result ──────────
def test_d_determinism():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(50.0, 10.0)
    r1 = sim.simulate(state, scenario, 60)
    r2 = sim.simulate(state, scenario, 60)
    assert r1.baseline_glucose == r2.baseline_glucose
    assert r1.counterfactual_glucose == r2.counterfactual_glucose
    assert r1.delta_glucose == r2.delta_glucose


# ── E: Future observations cannot enter the scenario ────────────────────────
def test_e_scenario_cannot_reference_future_observations():
    with pytest.raises(ValueError):
        validate_no_future_data_in_scenario(
            CounterfactualScenario(
                scenario_type=ScenarioType.MEAL_PERTURBATION,
                meal_carbs=30.0,
                meal_offset_minutes=5.0,
                description="Using observed future glucose for scenario.",
            )
        )


# ── F: Historical state remains unchanged after simulation ──────────────────
def test_f_historical_state_unchanged():
    state = _state(g=110.0)
    original_glucose = state.glucose
    original_ts = state.timestamp
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(50.0, 5.0)
    _ = sim.simulate(state, scenario, 30)
    # Original state must be unmodified
    assert state.glucose == original_glucose
    assert state.timestamp == original_ts


# ── G: Baseline and counterfactual start from identical state ───────────────
def test_g_baseline_and_counterfactual_share_initial_state():
    state = _state(g=110.0)
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(50.0, 5.0)
    res = sim.simulate(state, scenario, 30)
    assert res.start_timestamp == state.timestamp


# ── H: 30m output has exactly 6 points ─────────────────────────────────────
def test_h_30m_output_has_6_points():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.baseline()
    res = sim.simulate(state, scenario, 30)
    assert len(res.timestamps) == 6
    assert len(res.baseline_glucose) == 6
    assert len(res.counterfactual_glucose) == 6
    assert len(res.delta_glucose) == 6


# ── I: 60m output has exactly 12 points ─────────────────────────────────────
def test_i_60m_output_has_12_points():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.baseline()
    res = sim.simulate(state, scenario, 60)
    assert len(res.timestamps) == 12
    assert len(res.baseline_glucose) == 12
    assert len(res.counterfactual_glucose) == 12


# ── J: Invalid scenario is rejected ──────────────────────────────────────────
def test_j_invalid_scenario_rejected():
    with pytest.raises(ValueError):
        CounterfactualScenario(
            scenario_type=ScenarioType.MEAL_PERTURBATION,
            meal_carbs=None,
            meal_offset_minutes=5.0,
        ).validate()


def test_j_negative_horizon_rejected():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.baseline()
    with pytest.raises(ValueError):
        sim.simulate(state, scenario, -10)


# ── K: Unknown units remain unknown ──────────────────────────────────────────
def test_k_units_remain_unknown():
    from src.glucotwin.config.clinical import get_glucose_unit
    assert get_glucose_unit() == "unknown"


# ── L: No clinical threshold required by simulator ───────────────────────────
def test_l_no_clinical_threshold_in_simulator():
    import inspect
    import src.glucotwin.counterfactual.simulator as sim_mod
    src_code = inspect.getsource(sim_mod)
    assert "get_hypo_threshold" not in src_code
    assert "hypo_th" not in src_code
    assert "mg/dL" not in src_code


# ── Safety note in result ─────────────────────────────────────────────────────
def test_safety_note_present():
    state = _state()
    sim = _sim()
    res = sim.simulate(state, CounterfactualScenario.baseline(), 30)
    assert "not medical advice" in res.safety_note.lower()


# ── Numerical stability ───────────────────────────────────────────────────────
def test_numerical_stability_counterfactual():
    state = _state(g=100.0, ia=0.0, is_=0.0, ms=0.0)
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(carbs=200.0, offset_minutes=5.0)
    res = sim.simulate(state, scenario, 60)
    for g in res.counterfactual_glucose:
        assert not math.isnan(g)
        assert not math.isinf(g)
        assert g >= 10.0  # floor from dynamics


# ── Delta = counterfactual - baseline ─────────────────────────────────────────
def test_delta_is_counterfactual_minus_baseline():
    state = _state()
    sim = _sim()
    scenario = CounterfactualScenario.meal_perturbation(50.0, 5.0)
    res = sim.simulate(state, scenario, 30)
    for i, (b, c, d) in enumerate(
        zip(res.baseline_glucose, res.counterfactual_glucose, res.delta_glucose)
    ):
        assert abs((c - b) - d) < 1e-9, f"Step {i}: delta mismatch"
