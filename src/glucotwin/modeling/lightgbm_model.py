"""
LightGBM classifier wrapper for GlucoTwin.

Uses eval_X/eval_y API (LightGBM >= 4.0) with early stopping callback.
Falls back gracefully if LightGBM is not installed.
"""
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class LightGBMModel:
    def __init__(self, random_state: int = 42):
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        self.is_trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """
        Fit the model with optional validation-based early stopping.

        Uses the modern LightGBM >= 4.0 eval_X/eval_y keyword API.
        Early stopping is applied on validation AUC if a validation set
        is provided; otherwise plain fit is used.
        """
        if X_val is not None and y_val is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            # LightGBM >= 4.0: use eval_X / eval_y instead of eval_set
            try:
                self.model.fit(
                    X, y,
                    eval_X=X_val, eval_y=y_val,
                    callbacks=callbacks,
                )
            except TypeError:
                # Older API fallback
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                )
        else:
            self.model.fit(X, y)

        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of the positive class."""
        if not self.is_trained:
            raise ValueError("Model must be trained before predict_proba is called.")
        return self.model.predict_proba(X)[:, 1]
