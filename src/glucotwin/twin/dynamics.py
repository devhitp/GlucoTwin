from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters
from datetime import timedelta

class TwinDynamics:
    """
    Deterministic, interpretable discrete-time physiological-inspired dynamics model.
    Updates state over a 5-minute timestep.
    
    WARNING: Not clinically validated. Engineering approximation only.
    """
    
    @staticmethod
    def step(
        current_state: TwinState, 
        obs: TwinObservation, 
        params: TwinParameters, 
        dt_minutes: float = 5.0
    ) -> TwinState:
        """
        Advances the TwinState by dt_minutes using the current observation.
        """
        # Ensure we don't go backwards in time
        if obs.timestamp < current_state.timestamp:
            raise ValueError(f"Observation timestamp {obs.timestamp} is before state timestamp {current_state.timestamp}")
            
        # 1. Update meal absorption state
        # Zero carbs means no new event, not missing data. 
        new_carbs = obs.carbohydrates if obs.carbohydrates is not None else 0.0
        # Simple decay + new input
        meal_decay = params.meal_absorption_rate * current_state.meal_state
        next_meal = current_state.meal_state - meal_decay + new_carbs
        next_meal = max(0.0, next_meal)
        
        # 2. Update insulin exposure state
        new_bolus = obs.bolus if obs.bolus is not None else 0.0
        new_basal = obs.basal if obs.basal is not None else 0.0
        # Convert basal (U/hr) to U/5min loosely if needed, or treat generically
        # Since units are unknown, we treat generic inputs.
        insulin_input = new_bolus + (new_basal * (dt_minutes / 60.0))
        
        ins_decay = params.insulin_clearance_rate * current_state.insulin_state
        next_insulin = current_state.insulin_state - ins_decay + insulin_input
        next_insulin = max(0.0, next_insulin)
        
        # 3. Update insulin action state (delay compartment)
        action_decay = params.insulin_clearance_rate * current_state.insulin_action
        action_input = params.insulin_clearance_rate * current_state.insulin_state
        next_action = current_state.insulin_action - action_decay + action_input
        next_action = max(0.0, next_action)
        
        # 4. Update glucose state
        # If glucose observation is valid, we anchor to it (filtering step).
        # If missing, we simulate forward.
        if not obs.is_glucose_missing():
            # Observation anchors the state
            next_glucose = obs.glucose
        else:
            # Simulate forward
            drift = params.glucose_drift_coeff * (params.baseline_glucose - current_state.glucose)
            meal_effect = params.carb_conversion_coeff * meal_decay
            ins_effect = params.insulin_sensitivity * current_state.insulin_action
            
            next_glucose = current_state.glucose + drift + meal_effect - ins_effect
            # Basic numerical protection (prevent negative glucose)
            next_glucose = max(10.0, next_glucose)
            
        next_timestamp = current_state.timestamp + timedelta(minutes=dt_minutes)
        
        return TwinState(
            timestamp=next_timestamp,
            glucose=next_glucose,
            insulin_action=next_action,
            insulin_state=next_insulin,
            meal_state=next_meal,
            context_state=current_state.context_state,  # carry forward
            quality_flags=current_state.quality_flags.copy(),
            uncertainty=current_state.uncertainty.copy()
        )
