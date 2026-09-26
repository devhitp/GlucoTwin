import pandas as pd
import numpy as np

class CGMFeatures:
    @staticmethod
    def generate(df: pd.DataFrame, interpolate_limit: int = 2) -> pd.DataFrame:
        """
        Generate causal CGM features.
        Assumes df is indexed by a sorted timestamp.
        """
        if 'glucose' not in df.columns or df.empty:
            return df
            
        features = df.copy()
        
        # Track if it was originally missing
        features['glucose_missing'] = features['glucose'].isna().astype(int)
        
        # Interpolate short gaps carefully
        features['glucose_current'] = features['glucose'].interpolate(method='time', limit=interpolate_limit)
        features['glucose_interpolated'] = (features['glucose_missing'] == 1) & (features['glucose_current'].notna())
        features['glucose_interpolated'] = features['glucose_interpolated'].astype(int)
        
        # Deltas
        # Shift using index time to ensure causal nature
        features['glucose_prev_5m'] = features['glucose_current'].shift(freq='5min')
        features['glucose_prev_15m'] = features['glucose_current'].shift(freq='15min')
        
        features['glucose_delta_5m'] = features['glucose_current'] - features['glucose_prev_5m']
        features['glucose_delta_15m'] = features['glucose_current'] - features['glucose_prev_15m']
        
        # Rate of change (mg/dL per minute)
        features['glucose_roc_5m'] = features['glucose_delta_5m'] / 5.0
        
        # Rolling stats (backward looking only!)
        features['glucose_rolling_mean_30m'] = features['glucose_current'].rolling('30min').mean()
        features['glucose_rolling_std_30m'] = features['glucose_current'].rolling('30min').std()
        
        features['glucose_rolling_mean_60m'] = features['glucose_current'].rolling('60min').mean()
        features['glucose_rolling_std_60m'] = features['glucose_current'].rolling('60min').std()
        
        # Clean up temporary columns
        features = features.drop(columns=['glucose_prev_5m', 'glucose_prev_15m'])
        
        return features
