import pandas as pd
import numpy as np

class ErrorAnalysis:
    @staticmethod
    def analyze_errors(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Generates an error table containing FPs and FNs with context features.
        """
        if df.empty:
            return pd.DataFrame()
            
        errors = df.copy()
        errors['true_label'] = y_true
        errors['predicted_label'] = y_pred
        
        # Keep only errors
        mask_fp = (errors['true_label'] == 0) & (errors['predicted_label'] == 1)
        mask_fn = (errors['true_label'] == 1) & (errors['predicted_label'] == 0)
        
        errors['error_type'] = pd.Series(dtype='object')
        errors.loc[mask_fp, 'error_type'] = 'FP'
        errors.loc[mask_fn, 'error_type'] = 'FN'
        
        errors = errors.dropna(subset=['error_type'])
        
        # Select contextual columns to retain
        keep_cols = ['error_type', 'true_label', 'predicted_label']
        for c in ['glucose_current', 'glucose_roc_5m', 'insulin_bolus_last_30m', 'carbs_last_30m', 'is_night_clock_based']:
            if c in errors.columns:
                keep_cols.append(c)
                
        return errors[keep_cols]
