"""
Twin trajectory evaluation vs actual future glucose.

CRITICAL SEPARATION:
  - Future observations are used ONLY as evaluation targets here.
  - They are NEVER fed into feature construction or model training.

Metrics: MAE, RMSE, Median Absolute Error, Mean Bias.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
import numpy as np


@dataclass
class TwinForecastMetrics:
    horizon_minutes: int
    n_valid: int
    mae: float
    rmse: float
    median_abs_error: float
    mean_bias: float
    note: str = (
        "Twin forecast vs observed glucose. CGM units unconfirmed. "
        "Metrics are on the dataset's numerical glucose scale."
    )


def evaluate_twin_forecast(
    predicted: List[float],
    observed: List[float],
    horizon_minutes: int,
) -> TwinForecastMetrics:
    """
    Compare Twin-simulated trajectory against actual future observations.

    Args:
        predicted: List of Twin-predicted glucose values.
        observed: List of actual future glucose values (ONLY used as eval target).
        horizon_minutes: The forecast horizon (30 or 60).

    Returns:
        TwinForecastMetrics with aggregate error statistics.
    """
    pairs = [
        (p, o)
        for p, o in zip(predicted, observed)
        if p is not None
        and o is not None
        and not math.isnan(p)
        and not math.isnan(o)
    ]
    if not pairs:
        return TwinForecastMetrics(
            horizon_minutes=horizon_minutes,
            n_valid=0,
            mae=float("nan"),
            rmse=float("nan"),
            median_abs_error=float("nan"),
            mean_bias=float("nan"),
        )

    pred_arr = np.array([p for p, _ in pairs])
    obs_arr = np.array([o for _, o in pairs])
    errors = pred_arr - obs_arr

    return TwinForecastMetrics(
        horizon_minutes=horizon_minutes,
        n_valid=len(pairs),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        median_abs_error=float(np.median(np.abs(errors))),
        mean_bias=float(np.mean(errors)),
    )
