import pandas as pd

class MealFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        if 'carbohydrates' not in df.columns or df.empty:
            return df
            
        features = df.copy()
        
        # Fill missing with 0 for cumulative features
        carbs = features['carbohydrates'].fillna(0.0)
        
        features['carbs_last_30m'] = carbs.rolling('30min').sum()
        features['carbs_last_60m'] = carbs.rolling('60min').sum()
        
        # Time since latest meal
        meal_times = features.index.to_series().where(carbs > 0)
        last_meal_time = meal_times.ffill()
        
        features['minutes_since_meal'] = (features.index.to_series() - last_meal_time).dt.total_seconds() / 60.0
        
        return features
