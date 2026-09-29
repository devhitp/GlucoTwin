"""
Error analysis utilities for Sprint 7.

Analyses aggregate false-positive and false-negative patterns
across clinically-relevant dimensions. Never prints patient timelines
or identifiable records — all outputs are aggregate summaries.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# CGM bins used for error breakdown (in mg/dL — PROVISIONAL units)
GLUCOSE_BINS = [0, 54, 70, 90, 120, 180, 250, 600]
GLUCOSE_BIN_LABELS = ["<54", "54-70", "70-90", "90-120", "120-180", "180-250", ">250"]

# Rate-of-change bins (mg/dL per minute)
ROC_BINS = [-np.inf, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, np.inf]
ROC_BIN_LABELS = ["<<-2", "-2..-1", "-1..-0.5", "stable", "0.5..1", "1..2", ">>2"]


def _safe_bin(series: pd.Series, bins, labels) -> pd.Series:
    """Bin a series; return NaN for out-of-range values gracefully."""
    return pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=True)


def error_analysis(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, Any]:
    """
    Compute aggregate FP/FN breakdown across clinically-relevant dimensions.

    Parameters
    ----------
    X_test  : feature matrix (test set)
    y_true  : ground-truth binary labels
    y_pred  : predicted binary labels (after threshold)
    label   : name of the prediction target (e.g. 'lgbm_30m')

    Returns
    -------
    Dict of aggregate breakdown tables (JSON-serialisable).
    """
    df = X_test.copy()
    df["_y_true"] = y_true
    df["_y_pred"] = y_pred
    df["_error_type"] = "TN"
    df.loc[(df["_y_true"] == 1) & (df["_y_pred"] == 1), "_error_type"] = "TP"
    df.loc[(df["_y_true"] == 1) & (df["_y_pred"] == 0), "_error_type"] = "FN"
    df.loc[(df["_y_true"] == 0) & (df["_y_pred"] == 1), "_error_type"] = "FP"

    result: Dict[str, Any] = {"label": label, "n_samples": len(df)}

    # -- Breakdown by current glucose range ------------------------------------
    if "glucose_current" in df.columns:
        df["_glucose_bin"] = _safe_bin(df["glucose_current"], GLUCOSE_BINS, GLUCOSE_BIN_LABELS)
        result["by_glucose_range"] = (
            df.groupby(["_glucose_bin", "_error_type"], observed=True)
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Breakdown by glucose rate-of-change -----------------------------------
    if "glucose_roc_5m" in df.columns:
        df["_roc_bin"] = _safe_bin(df["glucose_roc_5m"], ROC_BINS, ROC_BIN_LABELS)
        result["by_roc"] = (
            df.groupby(["_roc_bin", "_error_type"], observed=True)
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Breakdown by time of day (night/day) ----------------------------------
    if "is_night_clock_based" in df.columns:
        result["by_night_day"] = (
            df.groupby(["is_night_clock_based", "_error_type"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Breakdown by recent bolus insulin (any bolus in last 30m) -------------
    if "bolus_insulin_recent" in df.columns:
        df["_had_recent_bolus"] = (df["bolus_insulin_recent"] > 0.0).astype(int)
        result["by_recent_bolus"] = (
            df.groupby(["_had_recent_bolus", "_error_type"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Breakdown by recent carbohydrates ------------------------------------
    if "carbs_recent" in df.columns:
        df["_had_recent_carbs"] = (df["carbs_recent"] > 0.0).astype(int)
        result["by_recent_carbs"] = (
            df.groupby(["_had_recent_carbs", "_error_type"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Breakdown by recording gap (already hypoglycemic) ---------------------
    if "minutes_since_last_glucose" in df.columns:
        df["_long_gap"] = (df["minutes_since_last_glucose"] > 15).astype(int)
        result["by_recording_gap"] = (
            df.groupby(["_long_gap", "_error_type"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )

    # -- Aggregate FP/FN summary counts ----------------------------------------
    counts = df["_error_type"].value_counts().to_dict()
    result["error_counts"] = {
        "TP": int(counts.get("TP", 0)),
        "TN": int(counts.get("TN", 0)),
        "FP": int(counts.get("FP", 0)),
        "FN": int(counts.get("FN", 0)),
    }

    return result
