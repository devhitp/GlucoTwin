from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class TwinTrajectory:
    """
    Simulated forward trajectory from the Digital Twin.
    DO NOT interpret as clinical medical advice or guaranteed outcomes.
    """
    start_timestamp: datetime
    horizon_minutes: int
    timestamps: List[datetime]
    predicted_glucose: List[float]
    
    # Metadata
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    parameter_version: str = "default"
    input_completeness: float = 1.0
