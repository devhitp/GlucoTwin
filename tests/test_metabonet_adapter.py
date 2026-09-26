import pandas as pd
from datetime import datetime
from src.glucotwin.data.adapters.metabonet import MetaboNetAdapter, CanonicalRecord

def test_metabonet_adapter_mapping():
    mapping = MetaboNetAdapter.map_columns()
    assert mapping["id"] == "subject_id"
    assert mapping["date"] == "timestamp"
    assert mapping["CGM"] == "glucose"
    assert mapping["basal"] == "basal"
    assert mapping["bolus"] == "bolus"
    assert mapping["carbs"] == "carbs"
    assert mapping["steps"] == "steps"

def test_metabonet_adapter_conversion():
    df = pd.DataFrame({
        "id": ["p1", "p2"],
        "date": ["2026-01-01 12:00:00", "2026-01-01 12:05:00"],
        "CGM": [105.0, None],
        "basal": [0.5, None],
        "bolus": [None, 2.0],
        "carbs": [15.0, None],
        "steps": [None, 100],
        "extra_col": ["ignore", "this"]
    })
    
    records = MetaboNetAdapter.convert_row_group(df)
    
    assert len(records) == 2
    
    # Record 1
    assert records[0].subject_id == "p1"
    assert isinstance(records[0].timestamp, pd.Timestamp)
    assert records[0].glucose == 105.0
    assert records[0].basal == 0.5
    assert records[0].bolus is None
    assert records[0].carbs == 15.0
    assert records[0].steps is None
    
    # Record 2
    assert records[1].subject_id == "p2"
    assert records[1].glucose is None
    assert records[1].basal is None
    assert records[1].bolus == 2.0
    assert records[1].carbs is None
    assert records[1].steps == 100

def test_metabonet_adapter_missing_required():
    # Missing subject_id
    df = pd.DataFrame({
        "date": ["2026-01-01 12:00:00"],
        "CGM": [105.0]
    })
    records = MetaboNetAdapter.convert_row_group(df)
    assert len(records) == 0
