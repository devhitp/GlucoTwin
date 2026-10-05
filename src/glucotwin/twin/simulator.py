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

    @property
    def params(self) -> TwinParameters:
        return self._params

    @params.setter
    def params(self, new_params: TwinParameters):
        self._params = new_params
        
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
            
        predicted_glucose = []
        timestamps = []
        
        step_minutes = 5.0
        steps = int(horizon_minutes / step_minutes)
        
        # Local scalars for ultra-fast tight loop (replaces deepcopy and dataclasses)
        curr_g = self._state.glucose
        curr_ia = self._state.insulin_action
        curr_ins = self._state.insulin_state
        curr_meal = self._state.meal_state
        
        m_rate = self._params.meal_absorption_rate
        i_rate = self._params.insulin_clearance_rate
        base_g = self._params.baseline_glucose
        d_coeff = self._params.glucose_drift_coeff
        c_coeff = self._params.carb_conversion_coeff
        i_sens = self._params.insulin_sensitivity
        
        curr_ts = self._state.timestamp
        dt_td = timedelta(minutes=step_minutes)
        
        for i in range(1, steps + 1):
            curr_ts += dt_td
            
            # Step dynamics mathematically
            meal_decay = m_rate * curr_meal
            next_meal = max(0.0, curr_meal - meal_decay)
            
            ins_decay = i_rate * curr_ins
            next_ins = max(0.0, curr_ins - ins_decay)
            
            action_decay = i_rate * curr_ia
            action_input = i_rate * curr_ins
            next_ia = max(0.0, curr_ia - action_decay + action_input)
            
            drift = d_coeff * (base_g - curr_g)
            meal_effect = c_coeff * meal_decay
            ins_effect = i_sens * curr_ia
            
            curr_g = max(10.0, curr_g + drift + meal_effect - ins_effect)
            curr_meal = next_meal
            curr_ins = next_ins
            curr_ia = next_ia
            
            timestamps.append(curr_ts)
            predicted_glucose.append(curr_g)
            
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
