from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime

@dataclass(frozen=True)
class TwinState:
    """
    Immutable representation of the Digital Twin physiological state at time t.
    
    WARNING: These are engineering constructs for numerical simulation.
    DO NOT interpret as clinically validated physiological quantities.
    """
    timestamp: datetime
    
    # G(t): current modeled glucose state
    glucose: float
    
    # X(t): latent insulin-action / glucose-disposal state
    insulin_action: float
    
    # I(t): latent insulin-related state (e.g. exposure/active insulin)
    insulin_state: float
    
    # M(t): latent meal absorption state
    meal_state: float
    
    # S_sleep(t): contextual state
    context_state: float
    
    # Metadata
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Checks for exploding numerical states."""
        import math
        return not (math.isnan(self.glucose) or math.isinf(self.glucose))
