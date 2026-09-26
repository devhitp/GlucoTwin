import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
from typing import Dict, Any

class Evaluation:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Check if single class
        if len(np.unique(y_true)) < 2:
            return {
                "precision": "unavailable",
                "recall": "unavailable",
                "f1": "unavailable",
                "roc_auc": "unavailable",
                "pr_auc": "unavailable",
                "balanced_accuracy": "unavailable",
                "specificity": "unavailable",
                "note": "Single class in ground truth."
            }
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "specificity": specificity,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
