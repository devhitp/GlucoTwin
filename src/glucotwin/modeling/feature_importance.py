import pandas as pd
from typing import Dict

class FeatureImportance:
    @staticmethod
    def get_importance(model, feature_names: list) -> pd.DataFrame:
        if not hasattr(model, 'is_trained') or not model.is_trained:
            return pd.DataFrame()
            
        importance = model.model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df
