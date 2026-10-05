"""
Hybrid model configurations for Sprint 9.

MODEL A: Baseline-only LightGBM (existing features)
MODEL B: Twin-only LightGBM (twin-derived features only)
MODEL C: Hybrid LightGBM (baseline + twin-derived features)

All models use the same patient-held-out split and LightGBM infrastructure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from src.glucotwin.modeling.lightgbm_model import LightGBMModel
from src.glucotwin.modeling.evaluation import Evaluation


class ModelConfig(str, Enum):
    BASELINE = "baseline"
    TWIN_ONLY = "twin_only"
    HYBRID = "hybrid"
    V2_DYNAMIC_HYBRID = "v2_dynamic_hybrid"


# Baseline feature columns that existed before Sprint 9
BASELINE_FEATURE_COLS = [
    "glucose_current",
    "glucose_roc_5m",
    "glucose_roc_15m",
    "glucose_roc_30m",
    "glucose_mean_30m",
    "glucose_std_30m",
    "glucose_min_30m",
    "glucose_max_30m",
    "insulin_recent_sum",
    "carbs_recent_sum",
    "time_since_meal",
    "time_since_bolus",
    "patient_glucose_baseline",
    "hour_of_day",
    "steps_available",
]

# Twin-derived feature columns (produced by trajectory_features.py)
TWIN_FEATURE_COLS = [
    "twin_glucose_current",
    "twin_insulin_action",
    "twin_insulin_state",
    "twin_meal_state",
    "twin_context_state",
    "twin_glucose_t5",
    "twin_glucose_t15",
    "twin_glucose_t30",
    "twin_glucose_t60",
    "twin_slope_30m",
    "twin_slope_60m",
    "twin_delta_15m",
    "twin_delta_30m",
    "twin_delta_60m",
    "twin_residual_current",
    "twin_has_long_gap",
    "twin_has_missing_glucose",
    "twin_uncertainty_calibrated",
]

TWIN_V2_FEATURE_COLS = [
    "twin_proj_min_60m",
    "twin_proj_max_60m",
    "twin_traj_area_60m",
    "twin_baseline_deviation",
    "twin_interaction_ia_trend",
    "twin_interaction_meal_trend",
]


def _select_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Select only columns that exist in the dataframe."""
    avail = [c for c in cols if c in df.columns]
    return df[avail].copy()


@dataclass
class HybridResult:
    """Stores evaluation results for one model configuration and horizon."""
    config: str
    horizon: str
    metrics: Dict[str, Any]
    feature_cols: List[str]
    n_train: int
    n_test: int
    threshold_note: str = (
        "Results use a provisional threshold under unresolved CGM unit provenance."
    )


class HybridModel:
    """
    Trains and evaluates A/B/C model configurations for one target horizon.
    All calibration is done on validation data only.
    """

    def __init__(self, config: ModelConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self._model = LightGBMModel(random_state=random_state)
        self._feature_cols: List[str] = []

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        if self.config == ModelConfig.BASELINE:
            return [c for c in BASELINE_FEATURE_COLS if c in df.columns]
        elif self.config == ModelConfig.TWIN_ONLY:
            return [c for c in TWIN_FEATURE_COLS if c in df.columns]
        elif self.config == ModelConfig.HYBRID:
            all_cols = BASELINE_FEATURE_COLS + TWIN_FEATURE_COLS
            return [c for c in all_cols if c in df.columns]
        else:  # V2_DYNAMIC_HYBRID
            all_cols = BASELINE_FEATURE_COLS + TWIN_FEATURE_COLS + TWIN_V2_FEATURE_COLS
            return [c for c in all_cols if c in df.columns]

    def fit(
        self,
        df_train: pd.DataFrame,
        target: str,
        df_val: Optional[pd.DataFrame] = None,
    ) -> None:
        self._feature_cols = self._resolve_features(df_train)
        if not self._feature_cols:
            raise ValueError(f"No feature columns found for config={self.config}.")

        X_train = df_train[self._feature_cols].fillna(0.0)
        y_train = df_train[target]

        X_val, y_val = None, None
        if df_val is not None:
            X_val = df_val[self._feature_cols].fillna(0.0)
            y_val = df_val[target]

        self._model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self._feature_cols].fillna(0.0)
        return self._model.predict_proba(X)

    def evaluate(self, df_test: pd.DataFrame, target: str, horizon: str) -> HybridResult:
        probs = self.predict_proba(df_test)
        y_true = df_test[target].values
        metrics = Evaluation.calculate_metrics(y_true, probs)

        # Add Brier score
        if len(np.unique(y_true)) >= 2:
            metrics["brier_score"] = float(np.mean((probs - y_true) ** 2))

        return HybridResult(
            config=self.config.value,
            horizon=horizon,
            metrics=metrics,
            feature_cols=self._feature_cols,
            n_train=0,   # filled by caller
            n_test=len(df_test),
        )

    @property
    def feature_cols(self) -> List[str]:
        return self._feature_cols
