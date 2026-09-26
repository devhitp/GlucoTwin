import pandas as pd
import numpy as np

class PersistenceBaseline:
    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Simple deterministic heuristic:
        If current glucose is already near threshold (e.g., < threshold + 10) 
        and ROC is negative, predict higher risk.
        Returns a pseudo-probability between 0 and 1.
        """
        probs = np.zeros(len(X))
        
        if 'glucose_current' not in X.columns or 'glucose_roc_5m' not in X.columns:
            return probs
            
        current = X['glucose_current']
        roc = X['glucose_roc_5m']
        
        # Risk factors
        # 1. Already below threshold: high risk of staying below
        idx_below = current < self.threshold
        probs[idx_below] = 0.9
        
        # 2. Near threshold (within 15 mg/dL) and dropping fast
        idx_near_dropping = (current >= self.threshold) & (current < self.threshold + 15) & (roc < -1.0)
        probs[idx_near_dropping] = 0.7
        
        # 3. Near threshold but dropping slowly
        idx_near_slow = (current >= self.threshold) & (current < self.threshold + 15) & (roc >= -1.0) & (roc < 0)
        probs[idx_near_slow] = 0.4
        
        return probs
