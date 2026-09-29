"""
Extended evaluation with Brier score and calibration for Sprint 7.
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, balanced_accuracy_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> Dict[str, Any]:
    """
    Full metric suite including Brier score. Handles single-class partitions.
    threshold is selected on validation set — never on test set.
    """
    unique = np.unique(y_true)
    if len(unique) < 2:
        return {
            "label": label,
            "note": "Single class in partition — metrics unavailable",
            "prevalence": float(y_true.mean()),
            "n_samples": len(y_true),
        }

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    brier = brier_score_loss(y_true, y_prob)

    return {
        "label": label,
        "n_samples": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold_on_val(y_val: np.ndarray, y_prob_val: np.ndarray) -> float:
    """
    Choose operating threshold that maximises F1 on validation set.
    NEVER called on test data.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 50):
        y_pred = (y_prob_val >= t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return float(best_t)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Reliability diagram data."""
    if len(np.unique(y_true)) < 2:
        return {"note": "Single class — calibration unavailable"}
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "fraction_positives": frac_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
    }
