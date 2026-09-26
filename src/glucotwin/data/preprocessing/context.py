import pandas as pd

class ContextFeatures:
    @staticmethod
    def generate(df: pd.DataFrame, night_start_hour: int = 22, night_end_hour: int = 6) -> pd.DataFrame:
        if df.empty:
            return df
            
        features = df.copy()
        
        # Datetime is in the index
        features['hour_of_day'] = features.index.hour
        features['minute_of_day'] = features.index.hour * 60 + features.index.minute
        features['day_of_week'] = features.index.dayofweek
        
        # Night indicator (clock-based)
        if night_start_hour > night_end_hour:
            # e.g., 22 to 6
            is_night = (features['hour_of_day'] >= night_start_hour) | (features['hour_of_day'] < night_end_hour)
        else:
            is_night = (features['hour_of_day'] >= night_start_hour) & (features['hour_of_day'] < night_end_hour)
            
        features['is_night_clock_based'] = is_night.astype(int)
        
        return features
