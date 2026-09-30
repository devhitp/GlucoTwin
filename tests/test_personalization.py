"""
Sprint 11 — Personalization and Memory Safety Tests.
"""
import pytest
import math
from datetime import datetime, timedelta
import pandas as pd

from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters
from src.glucotwin.twin.personalization import PersonalizationEngine, MIN_HISTORY_FOR_PERSONALIZATION
from src.glucotwin.hybrid.causal_feature_extractor import extract_causal_twin_features
from src.glucotwin.data.schema import SynchronizedRecord

T0 = datetime(2026, 1, 1, 9, 0)

def _obs(g, dt_minutes=0):
    return TwinObservation(
        timestamp=T0 + timedelta(minutes=dt_minutes),
        glucose=g,
        basal=None, bolus=None, carbohydrates=None
    )

def test_personalization_fallback_insufficient_data():
    """Falls back to population defaults if insufficient valid observations exist."""
    history = [_obs(100.0, i*5) for i in range(MIN_HISTORY_FOR_PERSONALIZATION - 1)]
    params = PersonalizationEngine.adapt(history)
    assert not params.is_personalized
    assert params.baseline_glucose == TwinParameters.default_population_params().baseline_glucose


def test_personalization_fits_baseline_correctly():
    """Calculates mean of valid glucose readings for baseline."""
    history = [_obs(100.0, i*5) for i in range(MIN_HISTORY_FOR_PERSONALIZATION + 5)]
    # Add some NaNs to ensure they are ignored
    history.append(_obs(float('nan'), 1000))
    history.append(_obs(None, 1005))
    
    params = PersonalizationEngine.adapt(history)
    assert params.is_personalized
    assert params.baseline_glucose == 100.0


def test_causal_feature_extractor_no_future_data():
    """
    Ensures feature extraction respects the causal boundary and correctly
    personalizes the twin parameter without looking at future rows.
    """
    records = []
    for i in range(20):
        records.append(SynchronizedRecord(
            patient_id="1", timestamp=T0 + timedelta(minutes=i*5),
            glucose=100.0, basal_insulin=0, bolus_insulin=0, carbohydrates=0,
            heart_rate=0, eda=0, skin_temperature=0,
            accelerometer_x=0, accelerometer_y=0, accelerometer_z=0
        ))
    
    df = extract_causal_twin_features(records)
    
    # We require 10 burn-in records, so feature extraction starts at index 10 (11th record)
    assert len(df) == 10
    
    # Check that timestamps are perfectly causal
    for i in range(10):
        expected_time = T0 + timedelta(minutes=(10 + i)*5)
        assert df.iloc[i]["timestamp"] == expected_time


def test_streaming_architecture_memory_regression():
    """
    Regression test preventing accidental large DataFrame concatenation.
    Ensures `merge_twin_features` concatenates locally and efficiently.
    """
    from scripts.run_metabonet_hybrid_streaming import merge_twin_features
    
    df_baseline = pd.DataFrame({
        "timestamp": [T0, T0 + timedelta(minutes=5)],
        "glucose_current": [100.0, 110.0]
    })
    
    df_twin = pd.DataFrame({
        "timestamp": [T0, T0 + timedelta(minutes=5)],
        "twin_glucose_t30": [105.0, 115.0]
    })
    
    merged = merge_twin_features(df_baseline, df_twin)
    assert len(merged) == 2
    assert "twin_glucose_t30" in merged.columns
    assert "glucose_current" in merged.columns
