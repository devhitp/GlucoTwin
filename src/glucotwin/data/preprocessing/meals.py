import pandas as pd


class MealFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate causal meal/carbohydrate features (all backward-looking).
        Assumes df is indexed by a sorted DatetimeIndex.
        """
        if 'carbohydrates' not in df.columns or df.empty:
            return df

        features = df.copy()

        # Fill missing carbs with 0 for cumulative rolling sums
        carbs = features['carbohydrates'].fillna(0.0)

        # -- Rolling carbohydrate sums (backward-looking) --------------------------
        features['carbs_last_30m'] = carbs.rolling('30min').sum()
        features['carbs_last_60m'] = carbs.rolling('60min').sum()

        # Sprint-7 canonical alias
        features['carbs_recent'] = features['carbs_last_30m']

        # -- Time since latest meal event ------------------------------------------
        meal_times = features.index.to_series().where(carbs > 0)
        last_meal_time = meal_times.ffill()
        features['minutes_since_meal'] = (
            features.index.to_series() - last_meal_time
        ).dt.total_seconds() / 60.0

        # Sprint-7 canonical alias
        features['minutes_since_last_meal'] = features['minutes_since_meal']

        return features
