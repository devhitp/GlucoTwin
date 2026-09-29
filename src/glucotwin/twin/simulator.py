from typing import List, Optional
from datetime import timedelta
import copy

from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters
from src.glucotwin.twin.dynamics import TwinDynamics
from src.glucotwin.twin.trajectory import TwinTrajectory

class TwinEngine:
    """
    The main interface for the Digital Twin simulation engine.
    """
    
    def __init__(self, initial_state: TwinState, params: TwinParameters = None):
        self._state = initial_state
        self._params = params if params is not None else TwinParameters.default_population_params()
        
    @property
    def current_state(self) -> TwinState:
        return self._state
        
    def update(self, obs: TwinObservation) -> TwinState:
        """
        Updates the engine's internal state with a new observation.
        """
        dt = (obs.timestamp - self._state.timestamp).total_seconds() / 60.0
        
        if dt < 0:
            raise ValueError("Cannot update with an observation from the past relative to current state.")
        if dt == 0:
            return self._state
            
        # Optional: Add quality flag logic
        flags = self._state.quality_flags.copy()
        if obs.is_glucose_missing():
            if "MISSING_GLUCOSE" not in flags:
                flags.append("MISSING_GLUCOSE")
        else:
            if "MISSING_GLUCOSE" in flags:
                flags.remove("MISSING_GLUCOSE")
                
        if dt > 15.0 and "LONG_GAP" not in flags:
            flags.append("LONG_GAP")
        elif dt <= 15.0 and "LONG_GAP" in flags:
            flags.remove("LONG_GAP")
            
        # Generate new state
        new_state = TwinDynamics.step(self._state, obs, self._params, dt_minutes=dt)
        
        # Inject updated flags
        # Hacky workaround since dataclasses are frozen, we instantiate a new one
        new_state = TwinState(
            timestamp=new_state.timestamp,
            glucose=new_state.glucose,
            insulin_action=new_state.insulin_action,
            insulin_state=new_state.insulin_state,
            meal_state=new_state.meal_state,
            context_state=new_state.context_state,
            quality_flags=flags,
            uncertainty=new_state.uncertainty
        )
        
        self._state = new_state
        return self._state
        
    def forecast(self, horizon_minutes: int) -> TwinTrajectory:
        """
        Forecasts the glucose trajectory without consuming any future real observations.
        Defaults forward inputs to zero (no new meals/bolus).
        """
        if horizon_minutes <= 0:
            raise ValueError("Horizon must be positive.")
            
        forecast_state = copy.deepcopy(self._state)
        predicted_glucose = []
        timestamps = []
        
        step_minutes = 5.0
        steps = int(horizon_minutes / step_minutes)
        
        for i in range(1, steps + 1):
            next_timestamp = self._state.timestamp + timedelta(minutes=i * step_minutes)
            
            # Create a dummy missing observation for the forecast step
            dummy_obs = TwinObservation(
                timestamp=next_timestamp,
                glucose=None,
                bolus=0.0,
                carbohydrates=0.0,
                # Basal could technically be assumed to continue, but setting to 0 for simplicity
                basal=0.0
            )
            
            forecast_state = TwinDynamics.step(forecast_state, dummy_obs, self._params, dt_minutes=step_minutes)
            
            timestamps.append(next_timestamp)
            predicted_glucose.append(forecast_state.glucose)
            
        return TwinTrajectory(
            start_timestamp=self._state.timestamp,
            horizon_minutes=horizon_minutes,
            timestamps=timestamps,
            predicted_glucose=predicted_glucose,
            uncertainty={"status": "not_calibrated"},
            quality_flags=self._state.quality_flags.copy(),
            parameter_version="personalized" if self._params.is_personalized else "default",
            input_completeness=1.0 # placeholder
        )
