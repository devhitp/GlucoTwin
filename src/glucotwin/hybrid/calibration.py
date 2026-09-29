"""
Probability calibration for hybrid model outputs.

Calibration is ONLY fitted on validation patients.
It is NEVER applied to or fitted on held-out test patients.

Outputs are labelled "calibrated model probabilities under this evaluation protocol."
They are NOT clinical confidence intervals.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class IsotonicCalibrator:
    """
    Isotonic regression calibration wrapper.
    Suitable when validation set is large enough (>= 1000 samples).
    """
    def __init__(self) -> None:
        self._calibrator = None
        self._fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(probs, y_true)
        self._fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Calibrator not fitted.")
        return self._calibrator.predict(probs).astype(float)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class PlattCalibrator:
    """
    Platt (logistic) calibration — preferred when validation set is small.
    """
    def __init__(self) -> None:
        self._calibrator = None
        self._fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        self._calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._calibrator.fit(probs.reshape(-1, 1), y_true)
        self._fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Calibrator not fitted.")
        return self._calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def select_calibrator(n_val: int):
    """Choose calibrator based on validation set size."""
    if n_val >= 1000:
        return IsotonicCalibrator()
    return PlattCalibrator()


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))
