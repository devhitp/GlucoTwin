import numpy as np
from sklearn.metrics import brier_score_loss

class CalibrationEvaluation:
    @staticmethod
    def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return brier_score_loss(y_true, y_prob)
