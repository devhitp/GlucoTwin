import pandas as pd
from typing import Tuple

class DataSplitter:
    @staticmethod
    def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically to avoid temporal leakage.
        """
        if df.empty:
            return df, pd.DataFrame()
            
        # Ensure sorted
        df = df.sort_index()
        
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df
