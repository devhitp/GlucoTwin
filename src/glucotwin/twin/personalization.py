from typing import List
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.parameters import TwinParameters

MIN_HISTORY_FOR_PERSONALIZATION = 10  # configurable


class PersonalizationEngine:
    """
    Safe causal personalization layer.
    Only adapts parameters that can be estimated from strictly historical data.
    Respects all causal boundaries and fallback policies.
    """

    @staticmethod
    def adapt(history: List[TwinObservation], params: TwinParameters = None) -> TwinParameters:
        """
        Adapts TwinParameters based on historical observations.
        Falls back to population defaults if insufficient history exists.
        Never accesses future data.
        """
        if params is None:
            params = TwinParameters.default_population_params()

        valid_glucose = [
            obs.glucose for obs in history
            if obs.glucose is not None and not _is_nan(obs.glucose)
        ]

        if len(valid_glucose) < MIN_HISTORY_FOR_PERSONALIZATION:
            # Insufficient history → return defaults (not personalized)
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


def _is_nan(v: float) -> bool:
    import math
    return math.isnan(v)
