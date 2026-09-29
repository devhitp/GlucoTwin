from dataclasses import dataclass

@dataclass(frozen=True)
class TwinParameters:
    """
    Model parameters for the physiological-inspired simulation.
    These are NOT clinically validated physiological constants.
    """
    # Glucose baseline / drift reference
    baseline_glucose: float = 110.0
    
    # Glucose intrinsic decay/drift coefficient
    glucose_drift_coeff: float = 0.05
    
    # Insulin sensitivity / action coupling coefficient
    insulin_sensitivity: float = 1.0
    
    # Insulin action time constant / clearance
    insulin_clearance_rate: float = 0.1
    
    # Meal absorption coefficient
    meal_absorption_rate: float = 0.05
    
    # Meal to glucose conversion efficiency
    carb_conversion_coeff: float = 2.0
    
    # Metadata for provenance
    is_personalized: bool = False
    
    @classmethod
    def default_population_params(cls) -> "TwinParameters":
        """Returns standard population priors."""
        return cls()
