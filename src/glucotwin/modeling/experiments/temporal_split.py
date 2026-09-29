"""
Temporal splitting with embargo for Sprint 7.

Within-subject chronological split:
  Train: first 60% of observations
  Val:   next 20% of observations
  Test:  last 20% of observations

An embargo gap (default 60 minutes) is removed from the boundaries
to prevent historical feature windows from overlapping label horizons.
"""
import pandas as pd
from typing import Tuple


EMBARGO_MINUTES = 60   # documented, configurable


def chronological_split_3way(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    embargo_minutes: int = EMBARGO_MINUTES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train, val, test) DataFrames with time-based embargo gaps.
    Assumes df.index is a DatetimeIndex in ascending order.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    df = df.sort_index()
    n = len(df)

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_end_time = df.index[train_end_idx - 1]
    val_end_time = df.index[val_end_idx - 1]

    embargo = pd.Timedelta(minutes=embargo_minutes)

    train = df[df.index <= train_end_time]
    val = df[(df.index > train_end_time + embargo) & (df.index <= val_end_time)]
    test = df[df.index > val_end_time + embargo]

    return train, val, test


def assert_no_temporal_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    embargo_minutes: int = EMBARGO_MINUTES,
) -> None:
    """Raise AssertionError if any temporal boundary is violated."""
    embargo = pd.Timedelta(minutes=embargo_minutes)
    if not train.empty and not val.empty:
        assert train.index.max() + embargo < val.index.min(), (
            f"Train/Val overlap: train ends {train.index.max()}, val starts {val.index.min()}"
        )
    if not val.empty and not test.empty:
        assert val.index.max() + embargo < test.index.min(), (
            f"Val/Test overlap: val ends {val.index.max()}, test starts {test.index.min()}"
        )
    if not train.empty and not test.empty:
        assert train.index.max() < test.index.min(), (
            f"Train/Test direct overlap"
        )
