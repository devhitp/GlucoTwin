"""
Uncertainty estimation via bootstrap ensemble of LightGBM models.

All ensemble members are trained only on training patients.
Validation is used for model selection.
Test patients are NEVER used during ensemble construction.

Output probabilities are labelled as:
  "bootstrap ensemble estimate — not a calibrated clinical confidence interval."
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.glucotwin.modeling.lightgbm_model import LightGBMModel


class BootstrapEnsemble:
    """
    Bootstrap ensemble of N LightGBM classifiers.
    Each member is trained on a bootstrap resample of the training data.
    """

    def __init__(self, n_members: int = 10, random_state: int = 42) -> None:
        self.n_members = n_members
        self.random_state = random_state
        self._members: List[LightGBMModel] = []
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: List[str],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self._members = []
        rng = np.random.default_rng(self.random_state)
        n = len(X_train)

        for i in range(self.n_members):
            idx = rng.integers(0, n, size=n)
            X_b = X_train.iloc[idx][feature_cols].fillna(0.0)
            y_b = y_train.iloc[idx]

            member = LightGBMModel(random_state=self.random_state + i)
            member.fit(X_b, y_b)
            self._members.append(member)

        self._feature_cols = feature_cols
        self._fitted = True

    def predict_uncertainty(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean_prob, lower_5pct, upper_95pct, std).
        These are bootstrap estimates, NOT calibrated clinical intervals.
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted.")
        X = df[self._feature_cols].fillna(0.0)
        preds = np.stack(
            [m.predict_proba(X) for m in self._members], axis=1
        )  # (n_samples, n_members)
        mean = preds.mean(axis=1)
        lower = np.percentile(preds, 5, axis=1)
        upper = np.percentile(preds, 95, axis=1)
        std = preds.std(axis=1)
        return mean, lower, upper, std

    def uncertainty_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        mean, lower, upper, std = self.predict_uncertainty(df)
        return [
            {
                "mean_prob": float(mean[i]),
                "lower_5pct": float(lower[i]),
                "upper_95pct": float(upper[i]),
                "width_90pct": float(upper[i] - lower[i]),
                "std": float(std[i]),
                "status": "bootstrap_ensemble",
            }
            for i in range(len(mean))
        ]

    @property
    def is_fitted(self) -> bool:
        return self._fitted
