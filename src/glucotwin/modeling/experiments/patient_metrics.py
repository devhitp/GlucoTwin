"""
Per-patient evaluation metrics for Sprint 7.5.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, brier_score_loss


def calculate_per_patient_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    patient_ids: pd.Series
) -> Dict[str, Any]:
    """
    Computes per-patient evaluation metrics.
    """
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': (y_prob >= threshold).astype(int)
    })
    
    results = {}
    for p_id, group in df.groupby('patient_id'):
        y_t = group['y_true'].values
        y_p = group['y_pred'].values
        y_prob_vals = group['y_prob'].values
        
        pos_count = y_t.sum()
        neg_count = len(y_t) - pos_count
        
        res = {
            'windows': len(y_t),
            'positive_count': int(pos_count),
            'negative_count': int(neg_count)
        }
        
        if pos_count > 0 and neg_count > 0:
            res['roc_auc'] = float(roc_auc_score(y_t, y_prob_vals))
            res['pr_auc'] = float(average_precision_score(y_t, y_prob_vals))
        else:
            res['roc_auc'] = "NA (single class)"
            res['pr_auc'] = "NA (single class)"
            
        if pos_count > 0:
            res['recall'] = float(recall_score(y_t, y_p))
        else:
            res['recall'] = "NA"
            
        if (y_p == 1).sum() > 0:
            res['precision'] = float(precision_score(y_t, y_p, zero_division=0))
        else:
            res['precision'] = 0.0
            
        res['f1'] = float(f1_score(y_t, y_p, zero_division=0))
        res['brier_score'] = float(brier_score_loss(y_t, y_prob_vals))
        
        results[str(p_id)] = res
        
    return results
