import pandas as pd
from typing import List, Tuple

class FeatureSanitizer:
    @staticmethod
    def sanitize(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts model-safe features and target label.
        Excludes future-looking fields, identifiers, and timestamps.
        """
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
            
        y = df[target_col]
        
        # Exclude metadata, target, and future-looking columns
        exclude_prefixes = ['future_', 'patient_id']
        feature_cols = []
        for c in df.columns:
            if not any(c.startswith(prefix) for prefix in exclude_prefixes):
                feature_cols.append(c)
                
        X = df[feature_cols].copy()
        
        # Handle infinities if any (replace with nan, which LightGBM can handle)
        import numpy as np
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return X, y
