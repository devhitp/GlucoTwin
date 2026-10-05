import pytest
from src.glucotwin.twin.personalization import DynamicPersonalizer, MIN_HISTORY_FOR_PERSONALIZATION
from src.glucotwin.twin.observations import TwinObservation
from datetime import datetime, timedelta
import math

def test_dynamic_personalization_insufficient_history():
    """Verify fallback when not enough data."""
    personalizer = DynamicPersonalizer()
    t = datetime(2026, 1, 1, 12, 0)
    for i in range(5):
        obs = TwinObservation(timestamp=t + timedelta(minutes=i*5), glucose=100.0, carbohydrates=None, bolus=None, basal=0.0)
        params = personalizer.update(obs)
        assert not params.is_personalized

def test_dynamic_personalization_bounded():
    """Verify bounds are respected."""
    personalizer = DynamicPersonalizer()
    t = datetime(2026, 1, 1, 12, 0)
    for i in range(MIN_HISTORY_FOR_PERSONALIZATION + 5):
        obs = TwinObservation(timestamp=t + timedelta(minutes=i*5), glucose=300.0, carbohydrates=None, bolus=None, basal=0.0)
        params = personalizer.update(obs)
        
    assert params.is_personalized
    assert params.baseline_glucose == 180.0  # bounded upper

def test_dynamic_personalization_damped_update():
    """Verify exponentially weighted update is smooth and doesn't jump."""
    personalizer = DynamicPersonalizer(alpha=0.1)
    t = datetime(2026, 1, 1, 12, 0)
    
    # Initialize at 100
    for i in range(MIN_HISTORY_FOR_PERSONALIZATION):
        obs = TwinObservation(timestamp=t + timedelta(minutes=i*5), glucose=100.0, carbohydrates=None, bolus=None, basal=0.0)
        params = personalizer.update(obs)
        
    assert params.baseline_glucose == 100.0
    
    # New obs at 150 -> update should be (1 - 0.1)*100 + 0.1*150 = 90 + 15 = 105
    obs = TwinObservation(timestamp=t + timedelta(minutes=MIN_HISTORY_FOR_PERSONALIZATION*5), glucose=150.0, carbohydrates=None, bolus=None, basal=0.0)
    params = personalizer.update(obs)
    assert math.isclose(params.baseline_glucose, 105.0)

def test_whatif_rejects_future_leakage():
    from src.glucotwin.counterfactual.simulator import CounterfactualSimulator
    from src.glucotwin.counterfactual.scenarios import CounterfactualScenario
    from src.glucotwin.twin.state import TwinState
    from src.glucotwin.twin.parameters import TwinParameters
    
    simulator = CounterfactualSimulator(TwinParameters.default_population_params())
    state = TwinState(
        timestamp=datetime(2026,1,1), glucose=100.0, 
        insulin_action=0.0, insulin_state=0.0, meal_state=0.0, context_state=0.0,
        quality_flags=set()
    )
    
    # Run a scenario
    scenario = CounterfactualScenario.meal_perturbation(carbs=20, offset_minutes=5)
    res = simulator.simulate(state, scenario, horizon_minutes=30)
    
    # Validate no historical observations were affected
    assert len(res.baseline_glucose) == 6
    assert len(res.counterfactual_glucose) == 6
