import pytest
from datetime import datetime, timedelta

from src.glucotwin.twin import (
    TwinState, 
    TwinObservation, 
    TwinParameters, 
    TwinInitializer, 
    TwinEngine
)

def test_twin_state_validity():
    s = TwinState(datetime(2026, 1, 1), 100.0, 0.0, 0.0, 0.0, 0.0)
    assert s.is_valid
    
    # Check invalid state
    import math
    s_invalid = TwinState(datetime(2026, 1, 1), float('nan'), 0.0, 0.0, 0.0, 0.0)
    assert not s_invalid.is_valid

def test_twin_observation_missing_glucose():
    obs = TwinObservation(timestamp=datetime(2026, 1, 1), glucose=None)
    assert obs.is_glucose_missing()
    
    obs2 = TwinObservation(timestamp=datetime(2026, 1, 1), glucose=100.0)
    assert not obs2.is_glucose_missing()

def test_causal_initialization():
    t0 = datetime(2026, 1, 1, 12, 0)
    history = [
        TwinObservation(timestamp=t0 - timedelta(minutes=15), glucose=110.0),
        TwinObservation(timestamp=t0 - timedelta(minutes=10), glucose=112.0),
        TwinObservation(timestamp=t0 - timedelta(minutes=5), glucose=115.0),
        TwinObservation(timestamp=t0, glucose=120.0),
    ]
    
    state = TwinInitializer.initialize(history)
    assert state.glucose == 120.0
    assert state.timestamp == t0

def test_initialization_rejects_empty():
    with pytest.raises(ValueError):
        TwinInitializer.initialize([])

def test_future_data_rejection():
    t0 = datetime(2026, 1, 1, 12, 0)
    engine = TwinEngine(TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0))
    
    # Try updating with past observation
    obs_past = TwinObservation(timestamp=t0 - timedelta(minutes=5), glucose=110.0)
    with pytest.raises(ValueError):
        engine.update(obs_past)

def test_deterministic_update_and_stability():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    # Update
    obs = TwinObservation(timestamp=t0 + timedelta(minutes=5), glucose=110.0)
    new_state = engine.update(obs)
    
    assert new_state.timestamp == t0 + timedelta(minutes=5)
    assert new_state.glucose == 110.0  # Anchored to observation

def test_meal_absorption():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    obs = TwinObservation(timestamp=t0 + timedelta(minutes=5), glucose=None, carbohydrates=50.0)
    new_state = engine.update(obs)
    
    assert new_state.meal_state > 0.0

def test_insulin_state_evolution():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    obs = TwinObservation(timestamp=t0 + timedelta(minutes=5), glucose=None, bolus=5.0)
    new_state = engine.update(obs)
    
    assert new_state.insulin_state > 0.0
    assert new_state.insulin_action >= 0.0

def test_forecast_length():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    traj30 = engine.forecast(30)
    assert len(traj30.timestamps) == 6
    assert traj30.horizon_minutes == 30
    
    traj60 = engine.forecast(60)
    assert len(traj60.timestamps) == 12
    assert traj60.horizon_minutes == 60

def test_wearable_absence():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    # Missing wearables shouldn't crash
    obs = TwinObservation(
        timestamp=t0 + timedelta(minutes=5), 
        glucose=110.0, 
        heart_rate=None, 
        steps=None
    )
    new_state = engine.update(obs)
    assert new_state.glucose == 110.0

def test_missing_glucose_simulation():
    t0 = datetime(2026, 1, 1, 12, 0)
    state = TwinState(t0, 100.0, 0.0, 0.0, 0.0, 0.0)
    engine = TwinEngine(state)
    
    obs = TwinObservation(timestamp=t0 + timedelta(minutes=5), glucose=None)
    new_state = engine.update(obs)
    
    # Drift pushes it towards baseline (110 default)
    assert new_state.glucose > 100.0 
    assert "MISSING_GLUCOSE" in new_state.quality_flags
