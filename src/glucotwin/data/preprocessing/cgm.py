import pandas as pd
import numpy as np

class CGMFeatures:
    @staticmethod
    def generate(df: pd.DataFrame, interpolate_limit: int = 2) -> pd.DataFrame:
        """
        Generate causal CGM features (all backward-looking — no future leakage).
        Assumes df is indexed by a sorted DatetimeIndex.
        """
        if 'glucose' not in df.columns or df.empty:
            return df

        features = df.copy()

        # -- Missingness indicator -------------------------------------------------
        features['glucose_missing'] = features['glucose'].isna().astype(int)

        # Interpolate short gaps only (max 2 consecutive missing = 10 min)
        features['glucose_current'] = features['glucose'].interpolate(
            method='time', limit=interpolate_limit
        )
        features['glucose_interpolated'] = (
            (features['glucose_missing'] == 1) & (features['glucose_current'].notna())
        ).astype(int)

        # -- Minutes since last valid glucose observation ---------------------------
        valid_ts = features.index.to_series().where(features['glucose_current'].notna())
        last_valid = valid_ts.ffill()
        features['minutes_since_last_glucose'] = (
            features.index.to_series() - last_valid
        ).dt.total_seconds() / 60.0

        # -- Lagged values (causal shifts) -----------------------------------------
        features['_glucose_prev_5m']  = features['glucose_current'].shift(freq='5min')
        features['_glucose_prev_15m'] = features['glucose_current'].shift(freq='15min')
        features['_glucose_prev_30m'] = features['glucose_current'].shift(freq='30min')

        # -- Deltas ----------------------------------------------------------------
        features['glucose_delta_5m']  = features['glucose_current'] - features['_glucose_prev_5m']
        features['glucose_delta_15m'] = features['glucose_current'] - features['_glucose_prev_15m']

        # -- Rates of change (mg/dL per minute) ------------------------------------
        features['glucose_roc_5m']  = features['glucose_delta_5m']  / 5.0
        features['glucose_roc_15m'] = features['glucose_delta_15m'] / 15.0
        features['glucose_roc_30m'] = (
            (features['glucose_current'] - features['_glucose_prev_30m']) / 30.0
        )

        # -- Rolling statistics (backward-looking only) ----------------------------
        gc = features['glucose_current']
        features['glucose_rolling_mean_30m'] = gc.rolling('30min').mean()
        features['glucose_rolling_std_30m']  = gc.rolling('30min').std()
        features['glucose_rolling_min_30m']  = gc.rolling('30min').min()
        features['glucose_rolling_max_30m']  = gc.rolling('30min').max()

        features['glucose_rolling_mean_60m'] = gc.rolling('60min').mean()
        features['glucose_rolling_std_60m']  = gc.rolling('60min').std()
        features['glucose_rolling_min_60m']  = gc.rolling('60min').min()
        features['glucose_rolling_max_60m']  = gc.rolling('60min').max()

        # -- Clean up internal temporaries -----------------------------------------
        features = features.drop(
            columns=['_glucose_prev_5m', '_glucose_prev_15m', '_glucose_prev_30m']
        )

        return features
