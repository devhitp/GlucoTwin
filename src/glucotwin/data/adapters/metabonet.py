import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

# Canonical Schema for GlucoTwin
@dataclass
class CanonicalRecord:
    subject_id: str
    timestamp: datetime
    glucose: Optional[float] = None
    insulin: Optional[float] = None
    basal: Optional[float] = None
    bolus: Optional[float] = None
    carbs: Optional[float] = None
    steps: Optional[float] = None
    heart_rate: Optional[float] = None
    eda: Optional[float] = None
    skin_temperature: Optional[float] = None

class MetaboNetAdapter:
    """
    Adapter to convert MetaboNet schema into the GlucoTwin canonical schema.
    """
    @staticmethod
    def map_columns() -> Dict[str, str]:
        """
        Maps actual MetaboNet columns to Canonical expected keys.
        """
        return {
            "id": "subject_id",
            "date": "timestamp",
            "CGM": "glucose",
            "insulin": "insulin",
            "basal": "basal",
            "bolus": "bolus",
            "carbs": "carbs",
            "steps": "steps",
            "heartrate": "heart_rate",
            "galvanic_skin_response": "eda",
            "skin_temp": "skin_temperature"
        }

    @classmethod
    def convert_row_group(cls, df: pd.DataFrame) -> List[CanonicalRecord]:
        """
        Converts a raw pandas DataFrame chunk (from MetaboNet parquet) to a list of CanonicalRecords.
        """
        records = []
        mapping = cls.map_columns()
        
        # Keep only needed columns if they exist in this chunk
        needed_cols = [k for k in mapping.keys() if k in df.columns]
        df_subset = df[needed_cols].copy()
        
        # Rename columns to match Canonical Schema kwargs
        df_subset.rename(columns=mapping, inplace=True)
        
        # Ensure subject_id and timestamp exist
        if "subject_id" not in df_subset.columns or "timestamp" not in df_subset.columns:
            return records
            
        # Convert types where necessary
        if not pd.api.types.is_datetime64_any_dtype(df_subset["timestamp"]):
            df_subset["timestamp"] = pd.to_datetime(df_subset["timestamp"])
            
        df_subset["subject_id"] = df_subset["subject_id"].astype(str)
        
        # Replace NaNs with None for the dataclass
        import numpy as np
        df_subset = df_subset.replace({np.nan: None})
        
        # Convert to records
        valid_fields = set(CanonicalRecord.__dataclass_fields__.keys())
        for dict_row in df_subset.to_dict('records'):
            valid_keys = {k: v for k, v in dict_row.items() if k in valid_fields}
            records.append(CanonicalRecord(**valid_keys))
            
        return records
