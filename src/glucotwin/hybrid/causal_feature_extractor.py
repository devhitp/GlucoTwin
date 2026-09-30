"""
Causal Twin Feature Extractor.
Processes a sequence of SynchronizedRecords forward in time,
generating Twin features for each timestamp using strictly causal information.
"""
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Tuple

from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinObservation, TwinParameters
from src.glucotwin.twin.personalization import PersonalizationEngine
from src.glucotwin.hybrid.trajectory_features import extract_twin_features
from src.glucotwin.data.schema import SynchronizedRecord


def _obs_from_record(rec: SynchronizedRecord) -> TwinObservation:
    g = rec.glucose
    return TwinObservation(
        timestamp=rec.timestamp,
        glucose=g if not math.isnan(g) else None,
        basal=rec.basal_insulin,
        bolus=rec.bolus_insulin,
        carbohydrates=rec.carbohydrates,
    )


def extract_causal_twin_features(records: List[SynchronizedRecord], target_timestamps=None) -> pd.DataFrame:
    """
    Simulates the Digital Twin forward in time.
    For each record, updates the twin state with observations strictly BEFORE or AT the record's timestamp.
    Then, extracts Twin features (forecasts) and returns a DataFrame of these features.
    
    If target_timestamps is provided, only extracts features for those timestamps to save CPU.
    """
    if len(records) < 10:
        return pd.DataFrame()
        
    observations = sorted([_obs_from_record(r) for r in records], key=lambda o: o.timestamp)
    
    # We need a burn-in period to initialize the Twin state.
    valid_for_init = [o for o in observations if not o.is_glucose_missing()]
    if len(valid_for_init) < 10:
        return pd.DataFrame()
        
    burn_in_obs = valid_for_init[:10]
    burn_in_end_time = burn_in_obs[-1].timestamp
    
    # Personalize parameters using the burn-in history
    params = PersonalizationEngine.adapt(burn_in_obs)
    
    try:
        init_state = TwinInitializer.initialize(burn_in_obs, params)
        engine = TwinEngine(init_state, params)
    except Exception:
        return pd.DataFrame()

    results = []
    
    # Track historical observations for dynamic personalization
    history = list(burn_in_obs)

    # Fast-forward engine to the end of burn-in
    # Track running stats for O(1) personalization updates
    valid_burn_in = [o.glucose for o in burn_in_obs if o.glucose is not None and not math.isnan(o.glucose)]
    running_sum = sum(valid_burn_in)
    running_count = len(valid_burn_in)
    
    for obs in observations:
        if obs.timestamp <= burn_in_end_time:
            continue
            
        # Update dynamic personalization incrementally
        if obs.glucose is not None and not math.isnan(obs.glucose):
            running_sum += obs.glucose
            running_count += 1
            
        # Periodically update the engine's parameters
        if running_count % 100 == 0 and running_count >= 10:
            baseline = running_sum / running_count
            engine.params = TwinParameters(
                baseline_glucose=baseline,
                glucose_drift_coeff=params.glucose_drift_coeff,
                insulin_sensitivity=params.insulin_sensitivity,
                insulin_clearance_rate=params.insulin_clearance_rate,
                meal_absorption_rate=params.meal_absorption_rate,
                carb_conversion_coeff=params.carb_conversion_coeff,
                is_personalized=True,
            )

        try:
            engine.update(obs)
            
            # Skip expensive feature extraction if timestamp is not needed
            if target_timestamps is not None and obs.timestamp not in target_timestamps:
                continue
                
            state = engine.current_state
            
            # Extract features based on current causal state
            traj30 = engine.forecast(30)
            traj60 = engine.forecast(60)
            
            feats = extract_twin_features(
                state, traj30, traj60,
                observed_glucose=state.glucose
            )
            
            feat_dict = feats.to_dict()
            feat_dict["timestamp"] = obs.timestamp
            results.append(feat_dict)
            
        except Exception:
            # If dynamics crash (e.g. weird data gap), skip feature extraction for this step
            continue

    return pd.DataFrame(results)
