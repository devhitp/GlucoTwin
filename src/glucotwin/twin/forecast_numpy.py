"""
Vectorised batch forecast for GlucoTwin V2.

SCIENTIFIC CONTRACT
-------------------
Equations are mathematically identical to TwinEngine.forecast() in simulator.py,
which uses explicit Euler with dt=5 minutes and zero external inputs.

Variable update order (must exactly match simulator.py lines 98-119):
  Step uses ALL old values; then all state variables are simultaneously replaced.

  meal_decay  = m_rate  * curr_meal          (old meal)
  next_meal   = max(0, curr_meal - meal_decay)

  ins_decay   = i_rate  * curr_ins           (old ins)
  next_ins    = max(0, curr_ins - ins_decay)

  action_decay = i_rate * curr_ia            (old ia)
  action_input = i_rate * curr_ins           (old ins — NOT next_ins)
  next_ia      = max(0, curr_ia - action_decay + action_input)

  drift        = d_coeff * (base_g - curr_g) (old g)
  meal_effect  = c_coeff * meal_decay
  ins_effect   = i_sens  * curr_ia           (old ia)
  next_g       = max(10, curr_g + drift + meal_effect - ins_effect)

  curr_meal = next_meal
  curr_ins  = next_ins
  curr_ia   = next_ia
  curr_g    = next_g

One 60-minute run captures all required horizon points:
  T+5  → step 1 (0-indexed: 0)
  T+15 → step 3 (0-indexed: 2)
  T+30 → step 6 (0-indexed: 5)
  T+60 → step 12 (0-indexed: 11)

This replaces two separate forecast(30) + forecast(60) calls per timestamp
with a single 12-step loop over N states simultaneously.

CAUSALITY: The caller (extract_causal_twin_features_fast) passes only states
that were computed from observations <= prediction time T. The forecast
receives no future observations.

NOT clinically validated. Engineering approximation only.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

# These indices exactly match _at_step() lookups in trajectory_features.py:
#   g5  = _at_step(traj30, 0)  → step index 0 = T+5
#   g15 = _at_step(traj30, 2)  → step index 2 = T+15
#   g30 = _at_step(traj30, 5)  → step index 5 = T+30
#   g60 = _at_step(traj60, 11) → step index 11 = T+60
_CAPTURE_STEPS = {0, 2, 5, 11}
_IDX_T5  = 0
_IDX_T15 = 2
_IDX_T30 = 5
_IDX_T60 = 11
_TOTAL_STEPS = 12  # 60 min / 5 min


def forecast_batch_numpy(
    g0:   np.ndarray,   # shape (N,) float64
    ia0:  np.ndarray,   # shape (N,) float64
    ins0: np.ndarray,   # shape (N,) float64
    m0:   np.ndarray,   # shape (N,) float64
    m_rate:  float,
    i_rate:  float,
    base_g,            # float or np.ndarray shape (N,) — per-row baseline supported
    d_coeff: float,
    c_coeff: float,
    i_sens:  float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised 60-minute forecast for N independent initial states.

    Returns (g_t5, g_t15, g_t30, g_t60, g_min, g_max, g_sum) each shape (N,).

    All N states are advanced simultaneously using NumPy array operations.
    No Python loop over states — only a Python loop over 12 time steps,
    which is constant regardless of N.
    """
    N = g0.shape[0]

    # Working state arrays (all old values; explicit Euler)
    g   = g0.astype(np.float64, copy=True)
    ia  = ia0.astype(np.float64, copy=True)
    ins = ins0.astype(np.float64, copy=True)
    m   = m0.astype(np.float64, copy=True)

    # Pre-allocate outputs
    g_t5  = np.empty(N, dtype=np.float64)
    g_t15 = np.empty(N, dtype=np.float64)
    g_t30 = np.empty(N, dtype=np.float64)
    g_t60 = np.empty(N, dtype=np.float64)
    
    g_min = np.full(N, np.inf, dtype=np.float64)
    g_max = np.full(N, -np.inf, dtype=np.float64)
    g_sum = np.zeros(N, dtype=np.float64)

    for step in range(_TOTAL_STEPS):
        # All reads use old values (explicit Euler — matches scalar loop)
        meal_decay   = m_rate * m
        ins_decay    = i_rate * ins
        action_decay = i_rate * ia
        action_input = i_rate * ins          # old ins, same as scalar line 109
        drift        = d_coeff * (base_g - g)
        meal_effect  = c_coeff * meal_decay
        ins_effect   = i_sens * ia           # old ia, same as scalar line 114

        # Update all states simultaneously
        m   = np.maximum(0.0, m   - meal_decay)
        ins = np.maximum(0.0, ins - ins_decay)
        ia  = np.maximum(0.0, ia  - action_decay + action_input)
        g   = np.maximum(10.0, g  + drift + meal_effect - ins_effect)
        
        g_min = np.minimum(g_min, g)
        g_max = np.maximum(g_max, g)
        g_sum += g

        # Capture required horizon points
        if step == _IDX_T5:
            g_t5[:] = g
        elif step == _IDX_T15:
            g_t15[:] = g
        elif step == _IDX_T30:
            g_t30[:] = g
        elif step == _IDX_T60:
            g_t60[:] = g

    return g_t5, g_t15, g_t30, g_t60, g_min, g_max, g_sum


def forecast_single_numpy(
    g0: float, ia0: float, ins0: float, m0: float,
    m_rate: float, i_rate: float, base_g: float,
    d_coeff: float, c_coeff: float, i_sens: float,
) -> Tuple[float, float, float, float]:
    """
    Convenience wrapper for a single initial state.
    Returns (g_t5, g_t15, g_t30, g_t60).
    Used in equivalence tests.
    """
    r = forecast_batch_numpy(
        np.array([g0]), np.array([ia0]),
        np.array([ins0]), np.array([m0]),
        m_rate, i_rate, base_g, d_coeff, c_coeff, i_sens,
    )
    return float(r[0][0]), float(r[1][0]), float(r[2][0]), float(r[3][0])
