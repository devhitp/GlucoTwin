import pandas as pd

class InsulinFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        if 'bolus_insulin' not in df.columns or df.empty:
            return df
            
        features = df.copy()
        
        # Fill missing bolus with 0 for cumulative sums
        bolus = features['bolus_insulin'].fillna(0.0)
        
        features['insulin_bolus_last_15m'] = bolus.rolling('15min').sum()
        features['insulin_bolus_last_30m'] = bolus.rolling('30min').sum()
        features['insulin_bolus_last_60m'] = bolus.rolling('60min').sum()
        
        # Time since latest bolus
        # Get series of timestamps where bolus > 0
        bolus_times = features.index.to_series().where(bolus > 0)
        # Forward fill to get the time of the *last* bolus at each row
        last_bolus_time = bolus_times.ffill()
        
        # Calculate minutes since last bolus
        features['minutes_since_bolus'] = (features.index.to_series() - last_bolus_time).dt.total_seconds() / 60.0
        
        # Current basal rate
        if 'basal_insulin' in features.columns:
            # Basal might be sparse, forward fill to get current active rate
            features['basal_current'] = features['basal_insulin'].ffill()
            
        return features
