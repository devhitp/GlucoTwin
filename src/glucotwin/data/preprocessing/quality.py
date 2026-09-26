import pandas as pd

class DataQuality:
    @staticmethod
    def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes invalid physiological values and duplicates.
        Returns a cleaned copy of the DataFrame.
        """
        if df.empty:
            return df
            
        cleaned = df.copy()
        
        # Sort chronologically just in case
        cleaned = cleaned.sort_index()
        
        # Drop duplicates by timestamp
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # Filter invalid physiological ranges
        if 'glucose' in cleaned.columns:
            cleaned.loc[cleaned['glucose'] < 0, 'glucose'] = None
            
        if 'carbohydrates' in cleaned.columns:
            cleaned.loc[cleaned['carbohydrates'] < 0, 'carbohydrates'] = None
            
        if 'basal_insulin' in cleaned.columns:
            cleaned.loc[cleaned['basal_insulin'] < 0, 'basal_insulin'] = None
            
        if 'bolus_insulin' in cleaned.columns:
            cleaned.loc[cleaned['bolus_insulin'] < 0, 'bolus_insulin'] = None
            
        return cleaned
