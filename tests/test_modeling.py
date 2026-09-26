import pytest
import numpy as np
import pandas as pd
from src.glucotwin.modeling.datasets import FeatureSanitizer
from src.glucotwin.modeling.baselines import PersistenceBaseline
from src.glucotwin.modeling.lightgbm_model import LightGBMModel
from src.glucotwin.modeling.evaluation import Evaluation
from src.glucotwin.modeling.thresholds import ThresholdAnalysis
from src.glucotwin.modeling.calibration import CalibrationEvaluation
from tests.fixtures.synthetic.modeling_generator import generate_synthetic_modeling_fixture

@pytest.fixture
def synthetic_df():
    return generate_synthetic_modeling_fixture(rows=200)

def test_feature_sanitization(synthetic_df):
    synthetic_df['patient_id'] = '1'
    X, y = FeatureSanitizer.sanitize(synthetic_df, 'future_hypoglycemia_30m')
    
    assert 'future_hypoglycemia_30m' not in X.columns
    assert 'future_hypoglycemia_60m' not in X.columns
    assert 'patient_id' not in X.columns
    assert len(X) == len(y)

def test_persistence_baseline(synthetic_df):
    baseline = PersistenceBaseline(threshold=70.0)
    probs = baseline.predict_proba(synthetic_df)
    
    assert len(probs) == len(synthetic_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

def test_lightgbm_training_and_evaluation(synthetic_df):
    X, y = FeatureSanitizer.sanitize(synthetic_df, 'future_hypoglycemia_30m')
    
    model = LightGBMModel(random_state=42)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    
    metrics = Evaluation.calculate_metrics(y, probs, threshold=0.5)
    assert "roc_auc" in metrics

def test_threshold_analysis(synthetic_df):
    X, y = FeatureSanitizer.sanitize(synthetic_df, 'future_hypoglycemia_30m')
    probs = np.random.uniform(0, 1, len(y))
    
    df_thresh = ThresholdAnalysis.evaluate_thresholds(y, probs, [0.1, 0.5, 0.9])
    assert len(df_thresh) == 3
    assert 'f1' in df_thresh.columns

def test_calibration(synthetic_df):
    X, y = FeatureSanitizer.sanitize(synthetic_df, 'future_hypoglycemia_30m')
    probs = np.random.uniform(0, 1, len(y))
    
    brier = CalibrationEvaluation.calculate_brier_score(y, probs)
    assert not np.isnan(brier)

def test_modeling_leakage_regression(synthetic_df):
    """
    Verifies that changing a future value does not change the training features,
    but does change the target.
    """
    df1 = synthetic_df.copy()
    df2 = synthetic_df.copy()
    
    # Anchor row
    idx = 100
    anchor_time = df1.index[idx]
    
    # Change future glucose in df2
    future_idx = 105 # +25 mins
    future_time = df2.index[future_idx]
    
    # Make sure we don't accidentally modify the feature explicitly, just the label
    # In a full pipeline, we'd change raw records. Here we simulate the pipeline output changing.
    df2.loc[future_time, 'glucose_current'] = 30.0 
    
    # Recompute labels for df2 as the pipeline would
    df2['future_hypoglycemia_30m'] = (df2['glucose_current'].shift(-6).rolling(6, min_periods=1).min() < 70).astype(float)
    
    X1, y1 = FeatureSanitizer.sanitize(df1, 'future_hypoglycemia_30m')
    X2, y2 = FeatureSanitizer.sanitize(df2, 'future_hypoglycemia_30m')
    
    # The features at anchor time should be EXACTLY identical
    pd.testing.assert_series_equal(X1.loc[anchor_time], X2.loc[anchor_time])
    
    # But the label for anchor time could be different
    # Actually, df1 label might be 0, df2 label might be 1 because we forced a drop
    assert y2.loc[anchor_time] == 1.0 or y1.loc[anchor_time] == y2.loc[anchor_time]
