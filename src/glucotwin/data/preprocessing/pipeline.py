import pandas as pd
from typing import List, Tuple
from src.glucotwin.data.schema import SynchronizedRecord
from .quality import DataQuality
from .cgm import CGMFeatures
from .insulin import InsulinFeatures
from .meals import MealFeatures
from .wearables import WearableFeatures
from .context import ContextFeatures
from .baseline import BaselineFeatures
from .labels import Labels
from .windows import TemporalWindows

class PreprocessingPipeline:
    @staticmethod
    def process_patient(records: List[SynchronizedRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in records])
        df = df.set_index('timestamp').sort_index()
        
        # 1. Quality
        df = DataQuality.filter_quality(df)
        
        # 2. Features
        df = CGMFeatures.generate(df)
        df = InsulinFeatures.generate(df)
        df = MealFeatures.generate(df)
        df = WearableFeatures.generate(df)
        df = ContextFeatures.generate(df)
        df = BaselineFeatures.generate(df)
        
        # 3. Labels
        df = Labels.generate(df)
        
        return df
        
    @staticmethod
    def extract_windows(df: pd.DataFrame, lookback_m: int = 60, horizon_m: int = 60) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if not c.startswith('future_')]
        label_cols = [c for c in df.columns if c.startswith('future_')]
        
        return TemporalWindows.generate_windows(df, feature_cols, label_cols, lookback_m, horizon_m)
