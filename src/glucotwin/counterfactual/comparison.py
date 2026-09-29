"""
Counterfactual comparison result.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class CounterfactualResult:
    """
    Typed result of a What-If simulation.

    INTERPRETATION NOTE:
    This result represents exploratory model-simulated trajectories
    under hypothetical input changes.
    It does NOT estimate a clinically validated treatment effect and
    must not be used to determine insulin dosing or medical treatment.
    """
    start_timestamp: datetime
    horizon_minutes: int
    scenario_description: str
    timestamps: List[datetime]

    baseline_glucose: List[float]
    counterfactual_glucose: List[float]
    delta_glucose: List[float]           # counterfactual - baseline

    uncertainty: Dict[str, Any] = field(default_factory=lambda: {"status": "not_calibrated"})
    quality_flags: List[str] = field(default_factory=list)
    parameter_version: str = "default"
    input_completeness: float = 1.0
    safety_note: str = (
        "Exploratory model simulation only. "
        "Not medical advice or a treatment recommendation."
    )
