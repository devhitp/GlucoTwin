from typing import List
from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters
from src.glucotwin.twin.dynamics import TwinDynamics

class TwinInitializer:
    """
    Initializes the Digital Twin state using STRICTLY historical and causal data.
    """
    
    @staticmethod
    def initialize(
        history: List[TwinObservation], 
        params: TwinParameters = None
    ) -> TwinState:
        """
        Creates an initial state by burning in the historical sequence.
        Requires at least one valid glucose observation to anchor the state.
        Never consumes future observations.
        """
        if not history:
            raise ValueError("Cannot initialize without historical observations.")
            
        if params is None:
            params = TwinParameters.default_population_params()
            
        # Find first valid glucose to start
        start_idx = -1
        for i, obs in enumerate(history):
            if not obs.is_glucose_missing():
                start_idx = i
                break
                
        if start_idx == -1:
            raise ValueError("Insufficient valid glucose in history to initialize.")
            
        first_obs = history[start_idx]
        
        # Create initial raw state
        current_state = TwinState(
            timestamp=first_obs.timestamp,
            glucose=first_obs.glucose,
            insulin_action=0.0,
            insulin_state=0.0,
            meal_state=0.0,
            context_state=0.0,
            quality_flags=["INITIALIZED"],
            uncertainty={"status": "not_calibrated"}
        )
        
        # Burn-in forward
        for obs in history[start_idx+1:]:
            # Calculate elapsed time in minutes
            dt = (obs.timestamp - current_state.timestamp).total_seconds() / 60.0
            if dt < 0:
                raise ValueError("Timestamp reversal detected in initialization history.")
            if dt == 0:
                continue # Skip duplicate timestamps or handle appropriately
                
            # Step the dynamics
            # Note: The dynamics step advances to current_state.timestamp + dt
            current_state = TwinDynamics.step(current_state, obs, params, dt_minutes=dt)
            
        return current_state
