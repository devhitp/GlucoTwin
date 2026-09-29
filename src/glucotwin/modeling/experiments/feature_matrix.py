"""
Aggregate feature matrix builder for Sprint 7.

Assembles the feature-label matrix across all subjects,
enforcing leakage-safe splitting per subject, then concatenates.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from src.glucotwin.data.preprocessing.pipeline import PreprocessingPipeline
from src.glucotwin.modeling.experiments.temporal_split import (
    chronological_split_3way,
    assert_no_temporal_leakage,
)

# Feature columns — wearables intentionally excluded from primary model
FEATURE_COLS = [
    "glucose_current",
    "glucose_roc_5m",
    "glucose_roc_15m",
    "glucose_roc_30m",
    "glucose_rolling_mean_30m",
    "glucose_rolling_std_30m",
    "glucose_rolling_min_30m",
    "glucose_rolling_max_30m",
    "glucose_rolling_mean_60m",
    "glucose_rolling_std_60m",
    "glucose_rolling_min_60m",
    "glucose_rolling_max_60m",
    "minutes_since_last_glucose",
    "basal_insulin_current",
    "bolus_insulin_recent",
    "total_insulin_last_30m",
    "total_insulin_last_60m",
    "carbs_recent",
    "minutes_since_last_meal",
    "patient_glucose_baseline",
    "patient_glucose_std",
    "hour_of_day",
    "is_night_clock_based",
    "day_of_week",
]

LABEL_30M = "future_hypoglycemia_30m"
LABEL_60M = "future_hypoglycemia_60m"


def _safe_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return only feature cols that actually exist in this subject's df."""
    return [c for c in FEATURE_COLS if c in df.columns]


def build_subject_matrices(
    records,  # List[SynchronizedRecord]
    embargo_minutes: int = 60,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Process one subject's records through the full pipeline,
    apply chronological split, and return (train, val, test) DataFrames.
    Returns None if the subject produces insufficient data.
    """
    df = PreprocessingPipeline.process_patient(records)
    if df.empty or len(df) < 50:
        return None

    # Drop rows with missing labels (end-of-record boundary)
    df_30 = df.dropna(subset=[LABEL_30M])
    df_60 = df.dropna(subset=[LABEL_60M])

    if len(df_30) < 30 or len(df_60) < 30:
        return None

    # Use 30m label df as primary; both share the same feature set
    train, val, test = chronological_split_3way(df_30, embargo_minutes=embargo_minutes)

    if train.empty or val.empty or test.empty:
        return None

    try:
        assert_no_temporal_leakage(train, val, test, embargo_minutes)
    except AssertionError:
        return None  # Skip subjects with boundary issues

    return train, val, test


def aggregate_splits(
    all_trains: List[pd.DataFrame],
    all_vals: List[pd.DataFrame],
    all_tests: List[pd.DataFrame],
    label_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Concatenate per-subject splits into global matrices.
    Shuffle train only (within the concatenated block — subjects are already temporally ordered).
    """
    train = pd.concat(all_trains, ignore_index=True)
    val = pd.concat(all_vals, ignore_index=True)
    test = pd.concat(all_tests, ignore_index=True)

    feat_cols = _safe_feature_cols(train)

    def split_xy(df):
        X = df[feat_cols].fillna(0)
        y = df[label_col].astype(int)
        return X, y

    X_tr, y_tr = split_xy(train)
    X_val, y_val = split_xy(val)
    X_te, y_te = split_xy(test)

    # Shuffle train rows (different subjects interleaved)
    idx = np.random.default_rng(42).permutation(len(X_tr))
    X_tr, y_tr = X_tr.iloc[idx].reset_index(drop=True), y_tr.iloc[idx].reset_index(drop=True)

    return X_tr, y_tr, X_val, y_val, X_te, y_te
