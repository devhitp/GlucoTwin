import pandas as pd


class InsulinFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate causal insulin features (all backward-looking — no future leakage).
        Assumes df is indexed by a sorted DatetimeIndex.
        """
        if 'bolus_insulin' not in df.columns or df.empty:
            return df

        features = df.copy()

        # Fill missing bolus with 0 for cumulative rolling sums
        bolus = features['bolus_insulin'].fillna(0.0)

        # -- Rolling bolus sums (backward-looking) --------------------------------
        features['insulin_bolus_last_15m'] = bolus.rolling('15min').sum()
        features['insulin_bolus_last_30m'] = bolus.rolling('30min').sum()
        features['insulin_bolus_last_60m'] = bolus.rolling('60min').sum()

        # Sprint-7 canonical aliases used by feature_matrix.py
        features['bolus_insulin_recent']  = features['insulin_bolus_last_30m']
        features['total_insulin_last_30m'] = features['insulin_bolus_last_30m']
        features['total_insulin_last_60m'] = features['insulin_bolus_last_60m']

        # -- Time since latest bolus event ----------------------------------------
        bolus_times = features.index.to_series().where(bolus > 0)
        last_bolus_time = bolus_times.ffill()
        features['minutes_since_bolus'] = (
            features.index.to_series() - last_bolus_time
        ).dt.total_seconds() / 60.0

        # -- Current basal rate (forward-filled) ----------------------------------
        if 'basal_insulin' in features.columns:
            features['basal_current'] = features['basal_insulin'].ffill()
            # Sprint-7 canonical alias
            features['basal_insulin_current'] = features['basal_current']

        return features
