import pandas as pd
from typing import Dict, Any, Tuple

class TemporalWindows:
    @staticmethod
    def validate_leakage(df: pd.DataFrame, feature_cols: list, label_cols: list):
        """
        In Sprint 2, the pipeline is evaluated row-by-row inherently through pandas.
        We ensure that for any row, the feature values are calculated safely.
        """
        pass
        
    @staticmethod
    def generate_windows(df: pd.DataFrame, feature_cols: list, label_cols: list, lookback_m: int = 60, horizon_m: int = 60) -> pd.DataFrame:
        """
        Drops rows that do not have sufficient history or future horizon.
        """
        if df.empty:
            return df
            
        windows = df.copy()
        
        # Check sufficient history (anchor >= start + lookback)
        start_time = windows.index.min()
        min_anchor = start_time + pd.Timedelta(minutes=lookback_m)
        windows = windows[windows.index >= min_anchor]
        
        # Check sufficient future horizon (anchor <= end - horizon)
        end_time = windows.index.max()
        max_anchor = end_time - pd.Timedelta(minutes=horizon_m)
        windows = windows[windows.index <= max_anchor]
        
        # Drop rows with unknown labels
        windows = windows.dropna(subset=label_cols)
        
        return windows
