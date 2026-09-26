import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .evaluation import Evaluation

class ThresholdAnalysis:
    @staticmethod
    def evaluate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
        results = []
        for t in thresholds:
            metrics = Evaluation.calculate_metrics(y_true, y_prob, t)
            if metrics["precision"] == "unavailable":
                continue
            
            row = {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"]
            }
            results.append(row)
            
        return pd.DataFrame(results)
