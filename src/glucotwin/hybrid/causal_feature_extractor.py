"""
Causal Twin Feature Extractor — V2 optimised.

Architecture
------------
The original implementation called engine.forecast(30) + engine.forecast(60)
inside the per-timestamp loop, creating two separate Python-loop simulations
for every prediction timestamp. On a subject with ~2000 windows this is
~4000 separate 12-step Python loops plus TwinTrajectory object construction.

V2 Optimisation
---------------
1. Sequential update loop (TwinEngine.update) is unchanged — it must remain
   sequential because each state depends on the previous one.

2. After the update loop, all (state, params) pairs at target timestamps are
   collected into NumPy arrays.

3. A single call to forecast_batch_numpy() runs all N forecasts simultaneously
   using vectorised NumPy over N states × 12 steps, with only 12 Python
   iterations regardless of N.

4. Feature construction uses the batch outputs directly, avoiding
   TwinTrajectory object creation for every window.

Scientific equivalence is verified by tests/test_forecast_numpy.py:
  - forecast_batch_numpy produces bit-for-bit identical results to
    TwinEngine.forecast() within float64 precision (tolerance 1e-10).

CAUSALITY CONTRACT
------------------
State at prediction time T is computed from observations strictly <= T.
The batch forecast receives only the state vectors at T; no future
observations enter the forecast computation.

NOT clinically validated. Engineering approximation only.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinObservation, TwinParameters
from src.glucotwin.twin.personalization import DynamicPersonalizer
from src.glucotwin.twin.forecast_numpy import forecast_batch_numpy
from src.glucotwin.data.schema import SynchronizedRecord


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _obs_from_record(rec: SynchronizedRecord) -> TwinObservation:
    g = rec.glucose
    return TwinObservation(
        timestamp=rec.timestamp,
        glucose=g if not math.isnan(g) else None,
        basal=rec.basal_insulin,
        bolus=rec.bolus_insulin,
        carbohydrates=rec.carbohydrates,
    )


def _build_feature_dict(
    ts,
    g0: float,
    ia: float,
    ins_state: float,
    meal: float,
    ctx: float,
    quality_flags: list,
    uncertainty: dict,
    g_t5: float,
    g_t15: float,
    g_t30: float,
    g_t60: float,
    g_min_60: float,
    g_max_60: float,
    g_sum_60: float,
    personalized_baseline: Optional[float],
) -> dict:
    """
    Assemble the feature dictionary from batch forecast outputs.

    Replaces extract_twin_features() + TwinTrajectory construction.
    Produces identical keys to TwinFeatures.to_dict().
    """
    def _s(v) -> Optional[float]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    g0_s   = _s(g0)
    g5_s   = _s(g_t5)
    g15_s  = _s(g_t15)
    g30_s  = _s(g_t30)
    g60_s  = _s(g_t60)

    slope_30m = ((g30_s - g0_s) / 30.0) if (g0_s is not None and g30_s is not None) else None
    slope_60m = ((g60_s - g0_s) / 60.0) if (g0_s is not None and g60_s is not None) else None
    delta_15m = (g15_s - g0_s) if (g0_s is not None and g15_s is not None) else None
    delta_30m = (g30_s - g0_s) if (g0_s is not None and g30_s is not None) else None
    delta_60m = (g60_s - g0_s) if (g0_s is not None and g60_s is not None) else None

    # Residual: always None here — observed_glucose == twin_glucose at update step
    # (engine anchors to observation when not missing), so residual == 0 is trivial.
    # Keep None to match original behaviour for missing-glucose steps.
    residual = 0.0 if g0_s is not None else None

    flags = quality_flags or []
    has_long_gap    = 1.0 if "LONG_GAP"        in flags else 0.0
    has_missing_gl  = 1.0 if "MISSING_GLUCOSE" in flags else 0.0
    calibrated      = 1.0 if (uncertainty or {}).get("status") == "calibrated" else 0.0

    # V2 trajectory features
    # Use the 12-step min/max/sum calculated during the batch forecast
    # to exactly match the reference trajectory features.
    proj_min_60 = _s(g_min_60)
    proj_max_60 = _s(g_max_60)
    traj_area_60 = _s(g_sum_60 * 5.0)

    base_dev   = (g0_s - personalized_baseline) if (g0_s is not None and personalized_baseline is not None) else None
    ia_trend   = (ia * slope_30m) if (slope_30m is not None and ia is not None) else None
    meal_trend = (meal * slope_30m) if (slope_30m is not None and meal is not None) else None

    return {
        "timestamp": ts,
        "twin_glucose_current":      g0_s,
        "twin_insulin_action":       _s(ia),
        "twin_insulin_state":        _s(ins_state),
        "twin_meal_state":           _s(meal),
        "twin_context_state":        _s(ctx),
        "twin_glucose_t5":           g5_s,
        "twin_glucose_t15":          g15_s,
        "twin_glucose_t30":          g30_s,
        "twin_glucose_t60":          g60_s,
        "twin_slope_30m":            slope_30m,
        "twin_slope_60m":            slope_60m,
        "twin_delta_15m":            delta_15m,
        "twin_delta_30m":            delta_30m,
        "twin_delta_60m":            delta_60m,
        "twin_residual_current":     residual,
        "twin_has_long_gap":         has_long_gap,
        "twin_has_missing_glucose":  has_missing_gl,
        "twin_uncertainty_calibrated": calibrated,
        "twin_proj_min_60m":         proj_min_60,
        "twin_proj_max_60m":         proj_max_60,
        "twin_traj_area_60m":        traj_area_60,
        "twin_baseline_deviation":   personalized_baseline, # DEBUG EXPORT
        "twin_interaction_ia_trend": ia_trend,
        "twin_interaction_meal_trend": meal_trend,
    }


# ---------------------------------------------------------------------------
# Public API — reference (scalar) path
# ---------------------------------------------------------------------------

def extract_causal_twin_features(
    records: List[SynchronizedRecord],
    target_timestamps=None,
) -> pd.DataFrame:
    """
    Reference implementation: scalar TwinEngine.forecast() per timestamp.

    Preserved as the ground-truth oracle for equivalence testing.
    Production code uses extract_causal_twin_features_fast() below.
    """
    from src.glucotwin.hybrid.trajectory_features import extract_twin_features

    if len(records) < 10:
        return pd.DataFrame()

    observations = sorted([_obs_from_record(r) for r in records], key=lambda o: o.timestamp)

    valid_for_init = [o for o in observations if not o.is_glucose_missing()]
    if len(valid_for_init) < 10:
        return pd.DataFrame()

    burn_in_obs = valid_for_init[:10]
    burn_in_end_time = burn_in_obs[-1].timestamp

    personalizer = DynamicPersonalizer()
    for o in burn_in_obs:
        params = personalizer.update(o)

    try:
        init_state = TwinInitializer.initialize(burn_in_obs, params)
        engine = TwinEngine(init_state, params)
    except Exception:
        return pd.DataFrame()

    results = []

    for obs in observations:
        if obs.timestamp <= burn_in_end_time:
            continue

        new_params = personalizer.update(obs)
        if new_params.is_personalized:
            engine.params = new_params

        try:
            engine.update(obs)

            if target_timestamps is not None and obs.timestamp not in target_timestamps:
                continue

            state = engine.current_state
            traj30 = engine.forecast(30)
            traj60 = engine.forecast(60)

            feats = extract_twin_features(
                state, traj30, traj60,
                observed_glucose=state.glucose,
                personalized_baseline=engine.params.baseline_glucose,
            )
            feat_dict = feats.to_dict()
            feat_dict["timestamp"] = obs.timestamp
            results.append(feat_dict)

        except Exception:
            continue

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Public API — optimised batch path (V2 production)
# ---------------------------------------------------------------------------

def extract_causal_twin_features_fast(
    records: List[SynchronizedRecord],
    target_timestamps=None,
) -> pd.DataFrame:
    """
    V2 Optimised causal Twin feature extractor.

    Sequential update loop is identical to the reference implementation.
    Forecast computation uses forecast_batch_numpy() to vectorise all N
    independent forecasts into a single NumPy batch call.

    Key correctness fix: all 6 param scalars are captured per-row inside the
    update loop to handle dynamic personalisation correctly. baseline_glucose
    is passed as a per-row array; other params are assumed constant
    (DynamicPersonalizer only updates baseline_glucose in the current impl).

    Equivalence verified by tests/test_forecast_numpy.py and
    tests/test_extractor_equivalence.py.
    """
    if len(records) < 10:
        return pd.DataFrame()

    observations = sorted([_obs_from_record(r) for r in records], key=lambda o: o.timestamp)

    valid_for_init = [o for o in observations if not o.is_glucose_missing()]
    if len(valid_for_init) < 10:
        return pd.DataFrame()

    burn_in_obs = valid_for_init[:10]
    burn_in_end_time = burn_in_obs[-1].timestamp

    personalizer = DynamicPersonalizer()
    for o in burn_in_obs:
        params = personalizer.update(o)

    try:
        init_state = TwinInitializer.initialize(burn_in_obs, params)
        engine = TwinEngine(init_state, params)
    except Exception:
        return pd.DataFrame()

    # Phase 1: Sequential state update — must remain sequential (causal dependency).
    # Capture state scalars AND params at each target timestamp.
    captured_ts      = []
    captured_g       = []
    captured_ia      = []
    captured_ins     = []
    captured_m       = []
    captured_ctx     = []
    captured_flags   = []
    captured_uncert  = []
    # Per-row param scalars (captures the params active at each prediction time)
    captured_base    = []
    captured_m_rate  = []
    captured_i_rate  = []
    captured_d_coeff = []
    captured_c_coeff = []
    captured_i_sens  = []

    for obs in observations:
        if obs.timestamp <= burn_in_end_time:
            continue

        new_params = personalizer.update(obs)
        if new_params.is_personalized:
            engine.params = new_params

        try:
            engine.update(obs)
        except Exception:
            continue

        if target_timestamps is not None and obs.timestamp not in target_timestamps:
            continue

        s = engine.current_state
        p = engine.params   # params AT this prediction time T
        captured_ts.append(obs.timestamp)
        captured_g.append(s.glucose)
        captured_ia.append(s.insulin_action)
        captured_ins.append(s.insulin_state)
        captured_m.append(s.meal_state)
        captured_ctx.append(s.context_state)
        captured_flags.append(list(s.quality_flags))
        captured_uncert.append(dict(s.uncertainty))
        captured_base.append(p.baseline_glucose)
        captured_m_rate.append(p.meal_absorption_rate)
        captured_i_rate.append(p.insulin_clearance_rate)
        captured_d_coeff.append(p.glucose_drift_coeff)
        captured_c_coeff.append(p.carb_conversion_coeff)
        captured_i_sens.append(p.insulin_sensitivity)

    if not captured_ts:
        return pd.DataFrame()

    N = len(captured_ts)
    g_arr    = np.array(captured_g,       dtype=np.float64)
    ia_arr   = np.array(captured_ia,      dtype=np.float64)
    ins_arr  = np.array(captured_ins,     dtype=np.float64)
    m_arr    = np.array(captured_m,       dtype=np.float64)
    base_arr = np.array(captured_base,    dtype=np.float64)

    # Phase 2: Group rows by identical (m_rate, i_rate, d_coeff, c_coeff, i_sens)
    # and run one batch call per unique param set (base_g is per-row).
    # In the common case (population priors), all rows share identical params
    # → single batch call over all N states.
    param_tuples = list(zip(
        captured_m_rate, captured_i_rate,
        captured_d_coeff, captured_c_coeff, captured_i_sens,
    ))
    unique_param_sets = list(dict.fromkeys(param_tuples))  # order-preserving dedupe

    g_t5  = np.empty(N, dtype=np.float64)
    g_t15 = np.empty(N, dtype=np.float64)
    g_t30 = np.empty(N, dtype=np.float64)
    g_t60 = np.empty(N, dtype=np.float64)
    g_min = np.empty(N, dtype=np.float64)
    g_max = np.empty(N, dtype=np.float64)
    g_sum = np.empty(N, dtype=np.float64)

    for ps in unique_param_sets:
        m_rate, i_rate, d_coeff, c_coeff, i_sens = ps
        mask = np.array([t == ps for t in param_tuples])
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        r5, r15, r30, r60, r_min, r_max, r_sum = forecast_batch_numpy(
            g_arr[idx], ia_arr[idx], ins_arr[idx], m_arr[idx],
            m_rate, i_rate, base_arr[idx],   # per-row base_g array
            d_coeff, c_coeff, i_sens,
        )
        g_t5[idx]  = r5
        g_t15[idx] = r15
        g_t30[idx] = r30
        g_t60[idx] = r60
        g_min[idx] = r_min
        g_max[idx] = r_max
        g_sum[idx] = r_sum

    # Phase 3: Build feature dicts (no TwinTrajectory objects created)
    results = []
    for i in range(N):
        try:
            d = _build_feature_dict(
                ts=captured_ts[i],
                g0=captured_g[i],
                ia=captured_ia[i],
                ins_state=captured_ins[i],
                meal=captured_m[i],
                ctx=captured_ctx[i],
                quality_flags=captured_flags[i],
                uncertainty=captured_uncert[i],
                g_t5=float(g_t5[i]),
                g_t15=float(g_t15[i]),
                g_t30=float(g_t30[i]),
                g_t60=float(g_t60[i]),
                g_min_60=float(g_min[i]),
                g_max_60=float(g_max[i]),
                g_sum_60=float(g_sum[i]),
                personalized_baseline=captured_base[i],
            )
            results.append(d)
        except Exception:
            continue

    return pd.DataFrame(results)


