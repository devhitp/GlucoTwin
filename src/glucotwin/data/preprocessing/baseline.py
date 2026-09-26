import pandas as pd

class BaselineFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate causal patient-specific baseline features.
        These use exponentially weighted moving averages or long rolling windows (e.g., 24h)
        to capture the patient's current baseline WITHOUT looking at the future.
        """
        if df.empty:
            return df
            
        features = df.copy()
        
        if 'glucose' in features.columns:
            # 24-hour rolling mean of glucose as a daily baseline
            # min_periods=1 allows it to start computing immediately, but it's causal.
            features['personal_glucose_baseline_24h'] = features['glucose'].rolling('24h', min_periods=1).mean()
            features['glucose_deviation_from_baseline'] = features['glucose'] - features['personal_glucose_baseline_24h']
            
        if 'heart_rate' in features.columns:
            features['personal_hr_baseline_24h'] = features['heart_rate'].rolling('24h', min_periods=1).mean()
            
        return features
