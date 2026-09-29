"""
Counterfactual scenario validation helpers.
"""
from src.glucotwin.counterfactual.scenarios import CounterfactualScenario, ScenarioType


def validate_no_future_data_in_scenario(scenario: CounterfactualScenario) -> None:
    """
    Asserts the scenario contains only hypothetical modifications, never
    references future actual observations, and is a valid typed scenario.
    Raises ValueError if any constraint is violated.
    """
    scenario.validate()
    # Scenario must not claim to reproduce observed future data
    desc_lower = scenario.description.lower()
    forbidden = ["observed future", "actual glucose", "future label", "target"]
    for phrase in forbidden:
        if phrase in desc_lower:
            raise ValueError(
                f"Scenario description contains forbidden phrase: '{phrase}'. "
                "Scenarios must be hypothetical, not references to future observed data."
            )
