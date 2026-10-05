from typing import List, Optional
import math
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters

MIN_HISTORY_FOR_PERSONALIZATION = 10  # configurable

def _is_nan(v: float) -> bool:
    return math.isnan(v)

class PersonalizationEngine:
    """
    Static causal personalization layer (V1).
    Only adapts parameters that can be estimated from strictly historical data.
    """
    @staticmethod
    def adapt(history: List[TwinObservation], params: TwinParameters = None) -> TwinParameters:
        if params is None:
            params = TwinParameters.default_population_params()

        valid_glucose = [
            obs.glucose for obs in history
            if obs.glucose is not None and not _is_nan(obs.glucose)
        ]

        if len(valid_glucose) < MIN_HISTORY_FOR_PERSONALIZATION:
            return params

        baseline = sum(valid_glucose) / len(valid_glucose)

        return TwinParameters(
            baseline_glucose=baseline,
            glucose_drift_coeff=params.glucose_drift_coeff,
            insulin_sensitivity=params.insulin_sensitivity,
            insulin_clearance_rate=params.insulin_clearance_rate,
            meal_absorption_rate=params.meal_absorption_rate,
            carb_conversion_coeff=params.carb_conversion_coeff,
            is_personalized=True,
        )


class DynamicPersonalizer:
    """
    V2 Dynamic Personalization Engine.
    Provides causally-updated patient-specific parameters using an exponentially
    weighted moving average to adapt safely as new observations arrive.
    
    Prevents parameter oscillation by applying bounding, regularized update rates, 
    and minimum observation requirements.
    """
    def __init__(self, params: Optional[TwinParameters] = None, alpha: float = 0.05):
        """
        Args:
            params: Initial parameters (or population defaults).
            alpha: Update rate for exponentially weighted moving average (0 < alpha <= 1).
                   Smaller alpha = slower, more damped update.
        """
        self.params = params or TwinParameters.default_population_params()
        self.alpha = alpha
        self._valid_obs_count = 0
        
        # Bounds for baseline glucose
        self._bg_lower = 80.0
        self._bg_upper = 180.0

    def update(self, obs: TwinObservation) -> TwinParameters:
        """
        Process a single causal observation and update parameters dynamically.
        Only data <= T influences the Twin parameters.
        Returns the updated TwinParameters.
        """
        if obs.glucose is not None and not _is_nan(obs.glucose):
            self._valid_obs_count += 1
            
            if self._valid_obs_count == MIN_HISTORY_FOR_PERSONALIZATION:
                # First time we have enough data, do a hard initialization
                # In real scenario, we'd average the first 10, but for stream just set it to current
                safe_bg = max(self._bg_lower, min(self._bg_upper, obs.glucose))
                self.params = TwinParameters(
                    baseline_glucose=safe_bg,
                    glucose_drift_coeff=self.params.glucose_drift_coeff,
                    insulin_sensitivity=self.params.insulin_sensitivity,
                    insulin_clearance_rate=self.params.insulin_clearance_rate,
                    meal_absorption_rate=self.params.meal_absorption_rate,
                    carb_conversion_coeff=self.params.carb_conversion_coeff,
                    is_personalized=True,
                )
            elif self._valid_obs_count > MIN_HISTORY_FOR_PERSONALIZATION:
                # Regularized / damped exponentially weighted update
                new_estimate = (1 - self.alpha) * self.params.baseline_glucose + self.alpha * obs.glucose
                
                # Bounded causal update
                safe_bg = max(self._bg_lower, min(self._bg_upper, new_estimate))
                
                self.params = TwinParameters(
                    baseline_glucose=safe_bg,
                    glucose_drift_coeff=self.params.glucose_drift_coeff,
                    insulin_sensitivity=self.params.insulin_sensitivity,
                    insulin_clearance_rate=self.params.insulin_clearance_rate,
                    meal_absorption_rate=self.params.meal_absorption_rate,
                    carb_conversion_coeff=self.params.carb_conversion_coeff,
                    is_personalized=True,
                )
                
        return self.params
