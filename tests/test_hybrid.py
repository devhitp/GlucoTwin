"""
Sprint 9 hybrid model tests.
All scenarios use deterministic synthetic data.
DO NOT use real patient records.
"""
import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.glucotwin.twin import TwinState, TwinObservation, TwinParameters, TwinEngine, TwinInitializer
from src.glucotwin.twin.trajectory import TwinTrajectory
from src.glucotwin.hybrid.trajectory_features import extract_twin_features, TwinFeatures
from src.glucotwin.hybrid.model import HybridModel, ModelConfig, BASELINE_FEATURE_COLS, TWIN_FEATURE_COLS
from src.glucotwin.hybrid.calibration import PlattCalibrator, IsotonicCalibrator, select_calibrator
from src.glucotwin.hybrid.uncertainty import BootstrapEnsemble
from src.glucotwin.hybrid.twin_eval import evaluate_twin_forecast

T0 = datetime(2026, 1, 1, 8, 0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_state(g=110.0, ia=0.5, is_=1.0, ms=5.0, flags=None):
    return TwinState(T0, g, ia, is_, ms, 0.0, quality_flags=flags or [])


def _make_trajectory(values, minutes=30):
    n = minutes // 5
    ts = [T0 + timedelta(minutes=5 * (i + 1)) for i in range(n)]
    return TwinTrajectory(
        start_timestamp=T0,
        horizon_minutes=minutes,
        timestamps=ts,
        predicted_glucose=list(values) + [110.0] * max(0, n - len(values)),
    )


def _synthetic_df(n=200, seed=0, include_twin=True):
    """Synthetic model-ready dataframe with baseline + twin columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "glucose_current": rng.uniform(60, 200, n),
        "glucose_roc_5m": rng.uniform(-3, 3, n),
        "glucose_roc_15m": rng.uniform(-2, 2, n),
        "glucose_roc_30m": rng.uniform(-1.5, 1.5, n),
        "glucose_mean_30m": rng.uniform(80, 180, n),
        "glucose_std_30m": rng.uniform(5, 30, n),
        "glucose_min_30m": rng.uniform(60, 120, n),
        "glucose_max_30m": rng.uniform(120, 200, n),
        "insulin_recent_sum": rng.uniform(0, 10, n),
        "carbs_recent_sum": rng.uniform(0, 80, n),
        "time_since_meal": rng.uniform(0, 300, n),
        "time_since_bolus": rng.uniform(0, 300, n),
        "patient_glucose_baseline": rng.uniform(90, 140, n),
        "hour_of_day": rng.integers(0, 24, n).astype(float),
        "steps_available": rng.choice([0.0, 1.0], n),
        "future_hypoglycemia_30m": rng.choice([0, 1], n, p=[0.88, 0.12]),
        "future_hypoglycemia_60m": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    if include_twin:
        df["twin_glucose_current"] = df["glucose_current"] + rng.normal(0, 5, n)
        df["twin_insulin_action"] = rng.uniform(0, 2, n)
        df["twin_insulin_state"] = rng.uniform(0, 3, n)
        df["twin_meal_state"] = rng.uniform(0, 20, n)
        df["twin_context_state"] = rng.uniform(0, 1, n)
        df["twin_glucose_t5"] = df["twin_glucose_current"] + rng.uniform(-5, 5, n)
        df["twin_glucose_t15"] = df["twin_glucose_current"] + rng.uniform(-10, 10, n)
        df["twin_glucose_t30"] = df["twin_glucose_current"] + rng.uniform(-15, 15, n)
        df["twin_glucose_t60"] = df["twin_glucose_current"] + rng.uniform(-20, 20, n)
        df["twin_slope_30m"] = rng.uniform(-1, 1, n)
        df["twin_slope_60m"] = rng.uniform(-0.5, 0.5, n)
        df["twin_delta_15m"] = rng.uniform(-10, 10, n)
        df["twin_delta_30m"] = rng.uniform(-15, 15, n)
        df["twin_delta_60m"] = rng.uniform(-20, 20, n)
        df["twin_residual_current"] = rng.normal(0, 3, n)
        df["twin_has_long_gap"] = 0.0
        df["twin_has_missing_glucose"] = 0.0
        df["twin_uncertainty_calibrated"] = 0.0
    return df


# ── A: Baseline-only model ───────────────────────────────────────────────────
def test_baseline_model_trains_and_evaluates():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.BASELINE)
    model.fit(train, "future_hypoglycemia_30m")
    result = model.evaluate(test, "future_hypoglycemia_30m", "30m")
    assert result.config == "baseline"
    assert result.metrics.get("roc_auc") != "unavailable"


# ── B: Twin-only model ────────────────────────────────────────────────────────
def test_twin_only_model_trains_and_evaluates():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.TWIN_ONLY)
    model.fit(train, "future_hypoglycemia_30m")
    result = model.evaluate(test, "future_hypoglycemia_30m", "30m")
    assert result.config == "twin_only"
    assert result.metrics.get("roc_auc") != "unavailable"


# ── C: Hybrid model ───────────────────────────────────────────────────────────
def test_hybrid_model_trains_and_evaluates():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.HYBRID)
    model.fit(train, "future_hypoglycemia_30m")
    result = model.evaluate(test, "future_hypoglycemia_30m", "30m")
    assert result.config == "hybrid"
    assert len(model.feature_cols) > len(
        [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    )


# ── D: Future observation leakage attempt ────────────────────────────────────
def test_future_observation_not_in_twin_features():
    """Twin features must only use T; future real glucose must NOT appear."""
    state = _make_state(g=110.0)
    traj30 = _make_trajectory([108, 106, 104, 102, 100, 98])
    traj60 = _make_trajectory([v - 5 for v in [108, 106, 104, 102, 100, 98]] + [90] * 6, 60)
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=110.0)
    # twin_glucose_t30 must come from trajectory, not real future
    assert feats.twin_glucose_t30 == pytest.approx(98.0, abs=0.1)
    # There is no "future_obs_t30" field
    assert not hasattr(feats, "future_obs_t30")


# ── E: Future residual leakage attempt ────────────────────────────────────────
def test_residual_uses_current_obs_only():
    """Residual must compare Twin vs observed at T, not at T+horizon."""
    state = _make_state(g=100.0)
    traj30 = _make_trajectory([98] * 6)
    traj60 = _make_trajectory([95] * 12, 60)
    # observed_glucose is at T, NOT future
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=105.0)
    # residual = 105.0 - 100.0
    assert feats.twin_residual_current == pytest.approx(5.0, abs=1e-6)


# ── F: Test-patient contamination guard ──────────────────────────────────────
def test_model_feature_cols_not_fitted_on_test():
    """Feature columns are resolved from train df, test uses same set."""
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.BASELINE)
    model.fit(train, "future_hypoglycemia_30m")
    # If test had extra columns, they should be ignored
    test_extra = test.copy()
    test_extra["secret_future_col"] = 999.0
    probs = model.predict_proba(test_extra)  # must not crash or use secret col
    assert len(probs) == len(test_extra)


# ── G: Calibration leakage — must fit on val, not test ───────────────────────
def test_calibration_fits_on_val_not_test():
    rng = np.random.default_rng(7)
    probs_val = rng.uniform(0, 1, 200)
    y_val = rng.choice([0, 1], 200, p=[0.88, 0.12])
    cal = PlattCalibrator()
    cal.fit(probs_val, y_val)
    assert cal.is_fitted
    # Calibrate test set with val-fitted calibrator
    probs_test = rng.uniform(0, 1, 50)
    cal_probs = cal.calibrate(probs_test)
    assert len(cal_probs) == 50
    assert np.all((cal_probs >= 0) & (cal_probs <= 1))


# ── H: Missing wearable data ──────────────────────────────────────────────────
def test_missing_wearables_in_features():
    state = _make_state()
    traj30 = _make_trajectory([110.0] * 6)
    traj60 = _make_trajectory([110.0] * 12, 60)
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=110.0)
    assert feats.twin_glucose_current is not None  # core feature present


# ── I: Missing insulin ────────────────────────────────────────────────────────
def test_feature_extraction_missing_insulin():
    state = TwinState(T0, 110.0, 0.0, 0.0, 0.0, 0.0)  # no insulin
    traj30 = _make_trajectory([110.0] * 6)
    traj60 = _make_trajectory([110.0] * 12, 60)
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=110.0)
    assert feats.twin_insulin_action == pytest.approx(0.0)
    assert feats.twin_insulin_state == pytest.approx(0.0)


# ── J: Missing meal data ──────────────────────────────────────────────────────
def test_feature_extraction_missing_meal():
    state = TwinState(T0, 110.0, 0.0, 0.0, 0.0, 0.0)  # meal_state=0
    traj30 = _make_trajectory([110.0] * 6)
    traj60 = _make_trajectory([110.0] * 12, 60)
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=110.0)
    assert feats.twin_meal_state == pytest.approx(0.0)


# ── K: Uncalibrated uncertainty metadata ──────────────────────────────────────
def test_uncalibrated_uncertainty_metadata():
    state = _make_state()
    traj30 = _make_trajectory([110.0] * 6)
    traj60 = _make_trajectory([110.0] * 12, 60)
    feats = extract_twin_features(state, traj30, traj60)
    assert feats.twin_uncertainty_calibrated == 0.0  # not_calibrated


# ── L: Bootstrap ensemble uncertainty ─────────────────────────────────────────
def test_bootstrap_ensemble_uncertainty():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    feature_cols = [c for c in TWIN_FEATURE_COLS if c in train.columns]
    y_train = train["future_hypoglycemia_30m"]

    ens = BootstrapEnsemble(n_members=5, random_state=42)
    ens.fit(train, y_train, feature_cols)
    assert ens.is_fitted

    mean, lower, upper, std = ens.predict_uncertainty(test)
    assert len(mean) == len(test)
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
    assert np.all(std >= 0)


# ── M: 30m target evaluation ──────────────────────────────────────────────────
def test_30m_target_evaluation():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.HYBRID)
    model.fit(train, "future_hypoglycemia_30m")
    result = model.evaluate(test, "future_hypoglycemia_30m", "30m")
    assert result.horizon == "30m"
    assert "brier_score" in result.metrics


# ── N: 60m target evaluation ──────────────────────────────────────────────────
def test_60m_target_evaluation():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.HYBRID)
    model.fit(train, "future_hypoglycemia_60m")
    result = model.evaluate(test, "future_hypoglycemia_60m", "60m")
    assert result.horizon == "60m"


# ── Twin trajectory evaluation ────────────────────────────────────────────────
def test_twin_forecast_evaluation_metrics():
    pred = [110.0, 108.0, 105.0, 100.0, 95.0, 90.0]
    obs  = [112.0, 109.0, 106.0, 101.0, 96.0, 91.0]
    metrics = evaluate_twin_forecast(pred, obs, 30)
    assert metrics.n_valid == 6
    assert metrics.mae == pytest.approx(7.0 / 6.0, abs=1e-4)
    assert metrics.mean_bias == pytest.approx(-7.0 / 6.0, abs=1e-4)


def test_twin_forecast_evaluation_all_missing():
    metrics = evaluate_twin_forecast([], [], 30)
    assert metrics.n_valid == 0
    assert math.isnan(metrics.mae)


# ── Causality regression ──────────────────────────────────────────────────────
def test_twin_future_obs_not_modifying_engine_state():
    """forecast() must not mutate the engine state."""
    engine = TwinEngine(TwinState(T0, 110.0, 0.0, 0.0, 0.0, 0.0))
    _ = engine.forecast(60)
    assert engine.current_state.timestamp == T0  # state unchanged


# ── Calibrator selection ──────────────────────────────────────────────────────
def test_calibrator_selection_by_size():
    cal_small = select_calibrator(500)
    cal_large = select_calibrator(2000)
    assert isinstance(cal_small, PlattCalibrator)
    assert isinstance(cal_large, IsotonicCalibrator)


# ── Provisional threshold not in hybrid dynamics ──────────────────────────────
def test_provisional_threshold_absent_from_hybrid():
    import inspect
    import src.glucotwin.hybrid.trajectory_features as tf_mod
    import src.glucotwin.hybrid.model as model_mod
    for mod in [tf_mod, model_mod]:
        src_code = inspect.getsource(mod)
        assert "get_hypo_threshold" not in src_code
        assert "hypo_th" not in src_code


# ── HybridResult threshold note ───────────────────────────────────────────────
def test_hybrid_result_carries_provisional_note():
    df = _synthetic_df(400)
    train, test = df.iloc[:300], df.iloc[300:]
    model = HybridModel(ModelConfig.HYBRID)
    model.fit(train, "future_hypoglycemia_30m")
    result = model.evaluate(test, "future_hypoglycemia_30m", "30m")
    assert "provisional" in result.threshold_note.lower()
