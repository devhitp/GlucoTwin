"""
What-If counterfactual simulator.

Operates entirely on the causal TwinState at time T.
Historical observations are NEVER modified.
Future actual observations CANNOT enter the simulation.
"""
from __future__ import annotations
import copy
from datetime import timedelta
from typing import List

from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.parameters import TwinParameters
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.twin.dynamics import TwinDynamics
from src.glucotwin.counterfactual.scenarios import CounterfactualScenario, ScenarioType
from src.glucotwin.counterfactual.comparison import CounterfactualResult

STEP_MINUTES = 5.0


class CounterfactualSimulator:
    """
    Runs baseline and counterfactual simulations from the same initial state.

    Causality guarantee:
        - The initial_state is treated as read-only.
        - Both simulations start from IDENTICAL copies of initial_state.
        - Future real observations are NEVER accessed by this simulator.
    """

    def __init__(self, params: TwinParameters = None) -> None:
        self._params = params or TwinParameters.default_population_params()

    def simulate(
        self,
        initial_state: TwinState,
        scenario: CounterfactualScenario,
        horizon_minutes: int,
    ) -> CounterfactualResult:
        """
        Run a single scenario simulation from initial_state for horizon_minutes.
        Returns a CounterfactualResult containing baseline + counterfactual trajectories.
        """
        scenario.validate()
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive.")

        n_steps = int(horizon_minutes / STEP_MINUTES)

        # Run baseline (no intervention)
        baseline_glucose = self._forward(initial_state, scenario=None, n_steps=n_steps)

        # Run counterfactual
        counterfactual_glucose = self._forward(initial_state, scenario=scenario, n_steps=n_steps)

        timestamps = [
            initial_state.timestamp + timedelta(minutes=(i + 1) * STEP_MINUTES)
            for i in range(n_steps)
        ]
        delta = [
            c - b for c, b in zip(counterfactual_glucose, baseline_glucose)
        ]

        return CounterfactualResult(
            start_timestamp=initial_state.timestamp,
            horizon_minutes=horizon_minutes,
            scenario_description=scenario.description,
            timestamps=timestamps,
            baseline_glucose=baseline_glucose,
            counterfactual_glucose=counterfactual_glucose,
            delta_glucose=delta,
            quality_flags=list(initial_state.quality_flags),
            parameter_version="personalized" if self._params.is_personalized else "default",
        )

    def _forward(
        self,
        initial_state: TwinState,
        scenario: CounterfactualScenario,
        n_steps: int,
    ) -> List[float]:
        """
        Simulate n_steps forward from a deep-copied initial state.
        Applies scenario modifications at the appropriate step, if any.
        """
        state = copy.deepcopy(initial_state)
        glucose_trajectory: List[float] = []

        for step_i in range(n_steps):
            elapsed = (step_i + 1) * STEP_MINUTES
            next_ts = initial_state.timestamp + timedelta(minutes=elapsed)

            # Defaults: no new meal, no new bolus
            carbs = 0.0
            bolus = 0.0

            # Apply scenario intervention at the matching step
            if scenario is not None and scenario.scenario_type != ScenarioType.BASELINE:
                if scenario.scenario_type == ScenarioType.MEAL_PERTURBATION:
                    if (
                        scenario.meal_offset_minutes is not None
                        and abs(elapsed - scenario.meal_offset_minutes) < STEP_MINUTES / 2
                    ):
                        carbs = scenario.meal_carbs

                elif scenario.scenario_type == ScenarioType.INSULIN_TIMING:
                    if (
                        scenario.insulin_offset_minutes is not None
                        and abs(elapsed - scenario.insulin_offset_minutes) < STEP_MINUTES / 2
                    ):
                        bolus = scenario.insulin_bolus

            obs = TwinObservation(
                timestamp=next_ts,
                glucose=None,  # No future glucose available
                carbohydrates=carbs if carbs > 0 else None,
                bolus=bolus if bolus > 0 else None,
                basal=0.0,
            )

            state = TwinDynamics.step(state, obs, self._params, dt_minutes=STEP_MINUTES)
            glucose_trajectory.append(state.glucose)

        return glucose_trajectory
