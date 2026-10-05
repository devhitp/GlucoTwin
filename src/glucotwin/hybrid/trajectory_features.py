"""
Twin-derived feature extractor.

All features are computed from observations at or before time T.
Twin forecast values (at T+30, T+60) are model-SIMULATED future states,
NOT real future observations.

CAUSALITY CONTRACT: No feature in this module may consume an actual
observation from (T, T+horizon].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math

from src.glucotwin.twin.state import TwinState
from src.glucotwin.twin.trajectory import TwinTrajectory


@dataclass
class TwinFeatures:
    """Named container for Twin-derived ML features at a given timestamp T."""

    # --- Current state features ---
    twin_glucose_current: Optional[float] = None       # G(t): current modeled glucose
    twin_insulin_action: Optional[float] = None        # X(t): insulin-action state
    twin_insulin_state: Optional[float] = None         # I(t): insulin exposure state
    twin_meal_state: Optional[float] = None            # M(t): meal absorption state
    twin_context_state: Optional[float] = None         # S(t): context state

    # --- 30-minute SIMULATED trajectory features ---
    twin_glucose_t5: Optional[float] = None            # Forecast G(T+5m)
    twin_glucose_t15: Optional[float] = None           # Forecast G(T+15m)
    twin_glucose_t30: Optional[float] = None           # Forecast G(T+30m)

    # --- 60-minute SIMULATED trajectory features ---
    twin_glucose_t60: Optional[float] = None           # Forecast G(T+60m)

    # --- Trajectory shape features ---
    twin_slope_30m: Optional[float] = None             # (G(T+30) - G(T)) / 30
    twin_slope_60m: Optional[float] = None             # (G(T+60) - G(T)) / 60
    twin_delta_15m: Optional[float] = None             # G(T+15) - G(T)
    twin_delta_30m: Optional[float] = None             # G(T+30) - G(T)
    twin_delta_60m: Optional[float] = None             # G(T+60) - G(T)

    # --- Residual feature (current only — causal) ---
    twin_residual_current: Optional[float] = None      # observed_glucose - twin_glucose

    # --- Quality metadata (encoded as numeric for ML) ---
    twin_has_long_gap: float = 0.0                     # 1.0 if LONG_GAP flag active
    twin_has_missing_glucose: float = 0.0              # 1.0 if MISSING_GLUCOSE flag
    twin_uncertainty_calibrated: float = 0.0           # 1.0 if uncertainty is calibrated

    # --- V2 Dynamic Trajectory Features ---
    twin_proj_min_60m: Optional[float] = None          # Predicted minimum over 60m
    twin_proj_max_60m: Optional[float] = None          # Predicted maximum over 60m
    twin_traj_area_60m: Optional[float] = None         # Area under trajectory
    twin_baseline_deviation: Optional[float] = None    # current G - personalized baseline
    twin_interaction_ia_trend: Optional[float] = None  # insulin action x slope 30m
    twin_interaction_meal_trend: Optional[float] = None # meal state x slope 30m

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def extract_twin_features(
    state: TwinState,
    trajectory_30m: TwinTrajectory,
    trajectory_60m: TwinTrajectory,
    observed_glucose: Optional[float] = None,
    personalized_baseline: Optional[float] = None,
) -> TwinFeatures:
    """
    Extract ML-ready features from Twin state + trajectory at time T.

    Args:
        state: Current TwinState at time T.
        trajectory_30m: Simulated 30-minute forward trajectory from T.
        trajectory_60m: Simulated 60-minute forward trajectory from T.
        observed_glucose: Actual CGM reading at T (used only for residual, not forecast).

    Returns:
        TwinFeatures instance with all computable fields populated.
    """
    def _safe(v: float) -> Optional[float]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    g0 = _safe(state.glucose)

    # --- Trajectory lookup helpers ---
    def _at_step(traj: TwinTrajectory, step_index: int) -> Optional[float]:
        """0-indexed step into trajectory list."""
        if traj is None or step_index >= len(traj.predicted_glucose):
            return None
        return _safe(traj.predicted_glucose[step_index])

    g5  = _at_step(trajectory_30m, 0)   # T+5
    g15 = _at_step(trajectory_30m, 2)   # T+15
    g30 = _at_step(trajectory_30m, 5)   # T+30
    g60 = _at_step(trajectory_60m, 11)  # T+60

    # --- Slope / delta features ---
    slope_30m = ((g30 - g0) / 30.0) if (g0 is not None and g30 is not None) else None
    slope_60m = ((g60 - g0) / 60.0) if (g0 is not None and g60 is not None) else None
    delta_15m = ((g15 - g0)) if (g0 is not None and g15 is not None) else None
    delta_30m = ((g30 - g0)) if (g0 is not None and g30 is not None) else None
    delta_60m = ((g60 - g0)) if (g0 is not None and g60 is not None) else None

    # --- Residual: observed vs Twin current glucose (CAUSAL — at T only) ---
    residual = None
    if observed_glucose is not None and g0 is not None:
        obs = _safe(observed_glucose)
        if obs is not None:
            residual = obs - g0

    # --- Quality flags ---
    flags = state.quality_flags or []
    has_long_gap = 1.0 if "LONG_GAP" in flags else 0.0
    has_missing_gl = 1.0 if "MISSING_GLUCOSE" in flags else 0.0
    unc = state.uncertainty or {}
    calibrated = 1.0 if unc.get("status") == "calibrated" else 0.0

    # --- V2 Dynamic Trajectory Features ---
    proj_min_60 = None
    proj_max_60 = None
    traj_area_60 = None
    if trajectory_60m and trajectory_60m.predicted_glucose:
        valid_g = [g for g in trajectory_60m.predicted_glucose if g is not None]
        if valid_g:
            proj_min_60 = min(valid_g)
            proj_max_60 = max(valid_g)
            # Area under curve (simple sum of 5-min intervals)
            traj_area_60 = sum(valid_g) * 5.0
            
    base_dev = None
    if g0 is not None and personalized_baseline is not None:
        base_dev = g0 - personalized_baseline
        
    ia_trend = None
    meal_trend = None
    if slope_30m is not None:
        if state.insulin_action is not None:
            ia_trend = state.insulin_action * slope_30m
        if state.meal_state is not None:
            meal_trend = state.meal_state * slope_30m

    return TwinFeatures(
        twin_glucose_current=g0,
        twin_insulin_action=_safe(state.insulin_action),
        twin_insulin_state=_safe(state.insulin_state),
        twin_meal_state=_safe(state.meal_state),
        twin_context_state=_safe(state.context_state),
        twin_glucose_t5=g5,
        twin_glucose_t15=g15,
        twin_glucose_t30=g30,
        twin_glucose_t60=g60,
        twin_slope_30m=slope_30m,
        twin_slope_60m=slope_60m,
        twin_delta_15m=delta_15m,
        twin_delta_30m=delta_30m,
        twin_delta_60m=delta_60m,
        twin_residual_current=residual,
        twin_has_long_gap=has_long_gap,
        twin_has_missing_glucose=has_missing_gl,
        twin_uncertainty_calibrated=calibrated,
        twin_proj_min_60m=proj_min_60,
        twin_proj_max_60m=proj_max_60,
        twin_traj_area_60m=traj_area_60,
        twin_baseline_deviation=personalized_baseline, # DEBUG EXPORT
        twin_interaction_ia_trend=ia_trend,
        twin_interaction_meal_trend=meal_trend,
    )
