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
            raise ImportError("LightGBM is not installed.")
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )
        self.is_trained = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fits the model."""
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        # Modern LightGBM uses callbacks for early stopping
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=10, verbose=False))
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None
        )
        self.is_trained = True
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)[:, 1]
