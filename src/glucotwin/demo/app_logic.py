"""
GlucoTwin demo application logic.

Provides the end-to-end inference pipeline for the Streamlit dashboard,
calling real project components (not hardcoded outputs).
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

from src.glucotwin.data.schema import SynchronizedRecord
from src.glucotwin.data.preprocessing.pipeline import PreprocessingPipeline
from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinObservation, TwinParameters
from src.glucotwin.twin.personalization import PersonalizationEngine
from src.glucotwin.hybrid.trajectory_features import extract_twin_features
from src.glucotwin.counterfactual.scenarios import CounterfactualScenario, ScenarioType
from src.glucotwin.counterfactual.simulator import CounterfactualSimulator
from src.glucotwin.modeling.experiments.feature_matrix import LABEL_30M, LABEL_60M, FEATURE_COLS
from src.glucotwin.hybrid.model import BASELINE_FEATURE_COLS, TWIN_FEATURE_COLS


DISCLAIMER = (
    "⚠️ RESEARCH PROTOTYPE — NOT a medical device. "
    "NOT clinical validation. "
    "CGM units UNKNOWN. Thresholds PROVISIONAL. "
    "Do NOT use for medical decisions."
)


def build_twin_from_records(
    records: List[SynchronizedRecord],
) -> Tuple[TwinEngine, TwinParameters]:
    """Initialize and return a personalized TwinEngine from records."""
    obs_list = []
    for r in records:
        g = r.glucose
        obs_list.append(TwinObservation(
            timestamp=r.timestamp,
            glucose=g if (g is not None and not math.isnan(g)) else None,
            basal=r.basal_insulin,
            bolus=r.bolus_insulin,
            carbohydrates=r.carbohydrates,
        ))
    obs_list.sort(key=lambda o: o.timestamp)

    # Personalize parameters
    params = PersonalizationEngine.adapt(obs_list)

    valid_obs = [o for o in obs_list if not o.is_glucose_missing()]
    if len(valid_obs) < 3:
        raise ValueError("Insufficient valid glucose observations to initialize Twin.")

    burn_in = valid_obs[:10] if len(valid_obs) >= 10 else valid_obs
    init_state = TwinInitializer.initialize(burn_in, params)
    engine = TwinEngine(init_state, params)

    # Run forward through remaining observations
    burn_in_end = burn_in[-1].timestamp
    for obs in obs_list:
        if obs.timestamp <= burn_in_end:
            continue
        try:
            engine.update(obs)
        except Exception:
            continue

    return engine, params


def run_demo_inference(
    records: List[SynchronizedRecord],
    model_30m=None,
    model_60m=None,
) -> Dict[str, Any]:
    """
    Run the full end-to-end inference pipeline on a list of records.

    Returns a dict with:
      - twin_state: current state fields
      - twin_params: personalization info
      - trajectory_30m: list of (t, glucose) for 30-min forecast
      - trajectory_60m: list of (t, glucose) for 60-min forecast
      - twin_features: TwinFeatures dict
      - glucose_history: list of (ts, glucose)
      - last_glucose: last observed glucose
      - risk_30m: predicted hypoglycemia risk if model provided
      - risk_60m: predicted hypoglycemia risk if model provided
      - feature_importances: dict (if model provided)
      - data_quality: dict
    """
    if len(records) < 10:
        raise ValueError("Need at least 10 records to run inference.")

    # Build Twin
    engine, params = build_twin_from_records(records)
    state = engine.current_state
    traj30 = engine.forecast(30)
    traj60 = engine.forecast(60)
    feats = extract_twin_features(state, traj30, traj60, observed_glucose=state.glucose)

    # Glucose history
    glucose_history = [
        (r.timestamp, r.glucose)
        for r in records
        if r.glucose is not None and not math.isnan(r.glucose)
    ]

    # Trajectory as list of (offset_minutes, glucose)
    traj30_pts = [
        (int((ts - state.timestamp).total_seconds() / 60), g)
        for ts, g in zip(traj30.timestamps, traj30.predicted_glucose)
    ]
    traj60_pts = [
        (int((ts - state.timestamp).total_seconds() / 60), g)
        for ts, g in zip(traj60.timestamps, traj60.predicted_glucose)
    ]

    # Data quality
    n_glucose = sum(1 for r in records if r.glucose is not None)
    n_total = len(records)
    gaps = []
    sorted_r = sorted(records, key=lambda r: r.timestamp)
    for i in range(1, len(sorted_r)):
        dt = (sorted_r[i].timestamp - sorted_r[i-1].timestamp).total_seconds() / 60
        if dt > 10:
            gaps.append(dt)

    data_quality = {
        "n_records": n_total,
        "cgm_availability_pct": round(100 * n_glucose / n_total, 1) if n_total else 0,
        "n_gaps_over_10m": len(gaps),
        "max_gap_minutes": round(max(gaps), 1) if gaps else 0,
        "quality_flags": state.quality_flags,
        "meal_records": sum(1 for r in records if r.carbohydrates and r.carbohydrates > 0),
        "insulin_records": sum(1 for r in records if (r.bolus_insulin and r.bolus_insulin > 0) or
                               (r.basal_insulin and r.basal_insulin > 0)),
    }

    # Risk prediction
    risk_30m = None
    risk_60m = None
    feat_imp_30m = None
    feat_imp_60m = None

    if model_30m is not None or model_60m is not None:
        # Build a single-row feature vector
        feat_dict = feats.to_dict()

        # Try to extract last baseline/CGM features
        try:
            df_proc = PreprocessingPipeline.process_patient(records)
            if not df_proc.empty:
                last_row = df_proc.iloc[-1]
                for c in BASELINE_FEATURE_COLS:
                    if c in last_row.index and c not in feat_dict:
                        feat_dict[c] = last_row[c]
        except Exception:
            pass

        row_df = pd.DataFrame([feat_dict])

        if model_30m is not None:
            try:
                risk_30m = float(model_30m.predict_proba(row_df)[0])
                feat_imp_30m = dict(zip(
                    model_30m.feature_cols,
                    model_30m._model.model.feature_importances_
                ))
            except Exception:
                pass

        if model_60m is not None:
            try:
                risk_60m = float(model_60m.predict_proba(row_df)[0])
                feat_imp_60m = dict(zip(
                    model_60m.feature_cols,
                    model_60m._model.model.feature_importances_
                ))
            except Exception:
                pass

    return {
        "twin_state": {
            "timestamp": str(state.timestamp),
            "glucose": round(state.glucose, 1) if state.glucose else None,
            "insulin_action": round(state.insulin_action, 3),
            "insulin_state": round(state.insulin_state, 3),
            "meal_state": round(state.meal_state, 3),
            "quality_flags": state.quality_flags,
        },
        "twin_params": {
            "baseline_glucose": round(params.baseline_glucose, 1),
            "is_personalized": params.is_personalized,
            "insulin_sensitivity": round(params.insulin_sensitivity, 3),
        },
        "twin_features": feats.to_dict(),
        "glucose_history": glucose_history,
        "last_glucose": glucose_history[-1][1] if glucose_history else None,
        "trajectory_30m": traj30_pts,
        "trajectory_60m": traj60_pts,
        "risk_30m": risk_30m,
        "risk_60m": risk_60m,
        "feature_importances_30m": feat_imp_30m,
        "feature_importances_60m": feat_imp_60m,
        "data_quality": data_quality,
    }


def run_whatif_simulation(
    records: List[SynchronizedRecord],
    scenario: CounterfactualScenario,
) -> Dict[str, Any]:
    """
    Run a What-If counterfactual simulation.
    Returns baseline trajectory and counterfactual trajectory.
    """
    scenario.validate()

    engine, params = build_twin_from_records(records)
    simulator = CounterfactualSimulator(params)

    res = simulator.simulate(engine.current_state, scenario, horizon_minutes=60)

    baseline_pts = list(zip(range(5, 65, 5), res.baseline_glucose))
    cf_pts = list(zip(range(5, 65, 5), res.counterfactual_glucose))

    delta = [
        (t, round(cf - bl, 2))
        for (t, bl), (_, cf) in zip(baseline_pts, cf_pts)
    ]

    return {
        "scenario": scenario.description,
        "baseline": baseline_pts,
        "counterfactual": cf_pts,
        "delta": delta,
        "horizon_minutes": 60,
        "disclaimer": "Simulation only — not a treatment recommendation.",
    }
