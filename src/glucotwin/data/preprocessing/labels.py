import pandas as pd

class Labels:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate future research event labels.
        Labels are allowed to look at the future, but they must be kept separate from features.
        """
        if 'glucose' not in df.columns or df.empty:
            return df
            
        labels = df.copy()
        
        # We need to look forward to see if glucose drops below threshold
        # using a rolling window reversed (looking forward)
        
        # In pandas, rolling forward can be done by reversing the series, rolling, and reversing back.
        # Alternatively, we can use an indexer.
        
        # Create a forward-looking rolling min for 30m and 60m
        # We use a custom indexer or simply shift if the frequency is fixed, 
        # but since index is datetime, we can use `indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=...)`
        # However, a simpler way is sorting descending, rolling, and sorting back.
        
        df_rev = labels.sort_index(ascending=False)
        
        # 30 minutes forward
        min_30m_fwd = df_rev['glucose'].rolling('30min', min_periods=1).min().sort_index()
        # 60 minutes forward
        min_60m_fwd = df_rev['glucose'].rolling('60min', min_periods=1).min().sort_index()
        
        # Hypoglycemia: < 70
        # Severe Hypoglycemia: < 54
        
        # 30m labels (does it cross threshold in the next 30m?)
        # Strictly speaking, "T < timestamp <= T + 30". The rolling min includes T.
        # So we should actually shift by 1 period forward first, or just subtract the current value if we strictly want > T.
        # A simpler approach:
        future_glucose_30m = df_rev['glucose'].shift(1).rolling('30min', min_periods=1).min().sort_index()
        future_glucose_60m = df_rev['glucose'].shift(1).rolling('60min', min_periods=1).min().sort_index()
        
        labels['future_hypoglycemia_30m'] = (future_glucose_30m < 70).astype(float)
        labels['future_severe_hypoglycemia_30m'] = (future_glucose_30m < 54).astype(float)
        
        labels['future_hypoglycemia_60m'] = (future_glucose_60m < 70).astype(float)
        labels['future_severe_hypoglycemia_60m'] = (future_glucose_60m < 54).astype(float)
        
        # Nocturnal label
        if 'is_night_clock_based' in labels.columns:
            labels['future_nocturnal_hypoglycemia_30m'] = (
                (labels['future_hypoglycemia_30m'] == 1) & (labels['is_night_clock_based'] == 1)
            ).astype(float)
            labels['future_nocturnal_hypoglycemia_60m'] = (
                (labels['future_hypoglycemia_60m'] == 1) & (labels['is_night_clock_based'] == 1)
            ).astype(float)
            
        # Handle unknown futures (if we are at the end of the dataset, there is no future data)
        # If the future window has no observations, the rolling min will be NaN.
        labels.loc[future_glucose_30m.isna(), ['future_hypoglycemia_30m', 'future_severe_hypoglycemia_30m', 'future_nocturnal_hypoglycemia_30m']] = None
        labels.loc[future_glucose_60m.isna(), ['future_hypoglycemia_60m', 'future_severe_hypoglycemia_60m', 'future_nocturnal_hypoglycemia_60m']] = None
        
        return labels
