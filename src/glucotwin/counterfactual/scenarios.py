"""
Counterfactual scenario model for the GlucoTwin What-If engine.

A scenario describes ONLY a hypothetical modification to forward inputs.
It never modifies historical observations.
It never recommends doses or treatments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ScenarioType(str, Enum):
    BASELINE = "baseline"                  # No hypothetical change
    MEAL_PERTURBATION = "meal_perturbation"
    INSULIN_TIMING = "insulin_timing"


@dataclass(frozen=True)
class CounterfactualScenario:
    """
    Typed, immutable description of a single What-If scenario.

    SAFETY NOTE:
    This object does NOT constitute medical advice, a treatment plan,
    or a dosing recommendation. It is an input to a model simulation only.

    Fields:
        scenario_type:      What kind of hypothetical change.
        meal_carbs:         (meal_perturbation) Hypothetical carbohydrate amount
                            to inject at t_offset_minutes after T.
        meal_offset_minutes: (meal_perturbation) When in the forecast to add the meal.
        insulin_bolus:      (insulin_timing) Hypothetical bolus amount.
        insulin_offset_minutes: (insulin_timing) When in the forecast to deliver the bolus.
        description:        Free-text explanation of the scenario.
    """
    scenario_type: ScenarioType
    meal_carbs: Optional[float] = None
    meal_offset_minutes: Optional[float] = None
    insulin_bolus: Optional[float] = None
    insulin_offset_minutes: Optional[float] = None
    description: str = ""

    def validate(self) -> None:
        """Raises ValueError on invalid scenario configuration."""
        if self.scenario_type == ScenarioType.MEAL_PERTURBATION:
            if self.meal_carbs is None or self.meal_carbs < 0:
                raise ValueError("meal_perturbation requires meal_carbs >= 0.")
            if self.meal_offset_minutes is None or self.meal_offset_minutes < 0:
                raise ValueError("meal_perturbation requires meal_offset_minutes >= 0.")
        elif self.scenario_type == ScenarioType.INSULIN_TIMING:
            if self.insulin_bolus is None or self.insulin_bolus < 0:
                raise ValueError("insulin_timing requires insulin_bolus >= 0.")
            if self.insulin_offset_minutes is None or self.insulin_offset_minutes < 0:
                raise ValueError("insulin_timing requires insulin_offset_minutes >= 0.")

    @classmethod
    def baseline(cls) -> "CounterfactualScenario":
        return cls(
            scenario_type=ScenarioType.BASELINE,
            description="Baseline: no hypothetical change to forward inputs.",
        )

    @classmethod
    def meal_perturbation(
        cls, carbs: float, offset_minutes: float, description: str = ""
    ) -> "CounterfactualScenario":
        return cls(
            scenario_type=ScenarioType.MEAL_PERTURBATION,
            meal_carbs=carbs,
            meal_offset_minutes=offset_minutes,
            description=description or f"Hypothetical meal of {carbs} units at +{offset_minutes}m.",
        )

    @classmethod
    def insulin_timing(
        cls, bolus: float, offset_minutes: float, description: str = ""
    ) -> "CounterfactualScenario":
        return cls(
            scenario_type=ScenarioType.INSULIN_TIMING,
            insulin_bolus=bolus,
            insulin_offset_minutes=offset_minutes,
            description=description or f"Hypothetical bolus of {bolus} units at +{offset_minutes}m.",
        )
