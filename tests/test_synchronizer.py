import pytest
from src.glucotwin.data.synchronizer import DataSynchronizer
from src.glucotwin.data.loader import DatasetLoader
from pathlib import Path

def test_synchronize():
    loader = DatasetLoader(str(Path(__file__).parent / "fixtures" / "synthetic"))
    records = loader.load_patient("559")
    synced = DataSynchronizer.synchronize(records)
    
    assert len(synced) == 3
    assert synced[0].glucose == 120.0
    assert synced[0].basal_insulin == 0.5
    assert synced[0].heart_rate == 70.0
    
    assert synced[1].glucose == 125.0
    assert synced[1].bolus_insulin == 2.0
    assert synced[1].carbohydrates == 30.0
