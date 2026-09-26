import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.glucotwin.data.schema import SynchronizedRecord
from src.glucotwin.data.preprocessing.pipeline import PreprocessingPipeline
from src.glucotwin.data.preprocessing.splits import DataSplitter
from src.glucotwin.data.preprocessing.cgm import CGMFeatures
from src.glucotwin.data.preprocessing.labels import Labels

@pytest.fixture
def sample_records():
    records = []
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    for i in range(20):
        # 5 minute intervals, total 100 minutes
        ts = start_time + timedelta(minutes=5 * i)
        glucose = 100.0 - i * 2 # goes down from 100 to 62
        rec = SynchronizedRecord(
            patient_id="1",
            timestamp=ts,
            glucose=glucose,
            basal_insulin=0.5,
            bolus_insulin=1.0 if i == 2 else 0.0,
            carbohydrates=20.0 if i == 5 else 0.0,
            heart_rate=70.0,
            eda=0.1,
            skin_temperature=36.5,
            accelerometer_x=0.0,
            accelerometer_y=1.0,
            accelerometer_z=0.0
        )
        records.append(rec)
    return records

def test_pipeline_basic(sample_records):
    df = PreprocessingPipeline.process_patient(sample_records)
    assert not df.empty
    
    # Check basic features
    assert 'glucose_rolling_mean_30m' in df.columns
    assert 'insulin_bolus_last_30m' in df.columns
    assert 'carbs_last_30m' in df.columns
    
def test_leakage_regression(sample_records):
    """
    CRITICAL REGRESSION TEST FOR LEAKAGE
    Modifying future glucose must NOT change current features,
    but MUST change the future labels.
    """
    # 1. Base pipeline
    df_base = PreprocessingPipeline.process_patient(sample_records)
    
    # Anchor point at i=5 (T = 08:25)
    anchor_time = datetime(2026, 1, 1, 8, 25, 0)
    base_features = df_base.loc[anchor_time].copy()
    
    # 2. Modify future glucose at i=10 (T = 08:50, which is +25m)
    modified_records = []
    for r in sample_records:
        if r.timestamp == datetime(2026, 1, 1, 8, 50, 0):
            r.glucose = 40.0 # Force severe hypoglycemia
        modified_records.append(r)
        
    df_mod = PreprocessingPipeline.process_patient(modified_records)
    mod_features = df_mod.loc[anchor_time].copy()
    
    # Assert features are unchanged
    feature_cols = [c for c in df_base.columns if not c.startswith('future_')]
    for c in feature_cols:
        if pd.notna(base_features[c]) or pd.notna(mod_features[c]):
            assert base_features[c] == mod_features[c], f"Leakage detected in feature {c}"
            
    # Assert label changed!
    assert base_features['future_severe_hypoglycemia_30m'] == 0.0
    assert mod_features['future_severe_hypoglycemia_30m'] == 1.0

def test_cgm_sorting_and_duplicates():
    records = [
        SynchronizedRecord("1", datetime(2026, 1, 1, 8, 5, 0), 110, None, None, None, None, None, None, None, None, None),
        SynchronizedRecord("1", datetime(2026, 1, 1, 8, 0, 0), 100, None, None, None, None, None, None, None, None, None),
        SynchronizedRecord("1", datetime(2026, 1, 1, 8, 0, 0), 105, None, None, None, None, None, None, None, None, None) # Duplicate
    ]
    df = PreprocessingPipeline.process_patient(records)
    # 2 unique timestamps
    assert len(df) == 2
    assert df.iloc[0]['glucose'] == 100.0

def test_cgm_negative_invalid():
    records = [
        SynchronizedRecord("1", datetime(2026, 1, 1, 8, 0, 0), -10, None, None, None, None, None, None, None, None, None)
    ]
    df = PreprocessingPipeline.process_patient(records)
    assert pd.isna(df.iloc[0]['glucose'])
    
def test_splits(sample_records):
    df = PreprocessingPipeline.process_patient(sample_records)
    train, test = DataSplitter.chronological_split(df, 0.8)
    assert len(train) > 0
    assert len(test) > 0
    assert train.index.max() < test.index.min()
