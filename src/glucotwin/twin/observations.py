from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass(frozen=True)
class TwinObservation:
    """
    Unit-agnostic representation of a physiological observation at time t.
    All variables represent unverified scales (e.g. glucose unit="unknown").
    """
    timestamp: datetime
    
    # Required core observations
    glucose: float
    glucose_roc: Optional[float] = None  # Rate of change
    
    # Required but sparse variables (None implies structurally absent/no event)
    basal: Optional[float] = None
    bolus: Optional[float] = None
    carbohydrates: Optional[float] = None
    
    # Context (elapsed physiological time)
    time_since_meal: Optional[float] = None
    time_since_bolus: Optional[float] = None
    
    # Optional wearable signals
    heart_rate: Optional[float] = None
    galvanic_skin_response: Optional[float] = None
    skin_temp: Optional[float] = None
    steps: Optional[float] = None

    def is_glucose_missing(self) -> bool:
        """Returns True if glucose is nan or absent."""
        import math
        return self.glucose is None or math.isnan(self.glucose)
