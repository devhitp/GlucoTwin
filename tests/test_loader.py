import pytest
from src.glucotwin.data.loader import DatasetLoader
from pathlib import Path

def test_loader_discover():
    loader = DatasetLoader(str(Path(__file__).parent / "fixtures" / "synthetic"))
    patients = loader.discover_patients()
    assert "559" in patients

def test_loader_load():
    loader = DatasetLoader(str(Path(__file__).parent / "fixtures" / "synthetic"))
    records = loader.load_patient("559")
    assert len(records["cgm"]) == 3
    assert len(records["insulin"]) == 2
    assert len(records["meal"]) == 1
    assert len(records["wearable"]) == 1
