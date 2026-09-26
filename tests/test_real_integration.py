import pytest
from pathlib import Path
from src.glucotwin.data.loader import DatasetLoader

def test_dataset_discovery_missing():
    loader = DatasetLoader(Path("nonexistent_path"))
    assert loader.discover_patients() == []

def test_dataset_discovery_valid(tmp_path):
    # Create fake XML files to simulate valid dataset discovery
    (tmp_path / "101.xml").touch()
    (tmp_path / "102.xml").touch()
    (tmp_path / "not_xml.txt").touch()
    
    loader = DatasetLoader(tmp_path)
    patients = loader.discover_patients()
    assert patients == ["101", "102"]

def test_loader_missing_patient():
    loader = DatasetLoader(Path("data/raw"))
    with pytest.raises(FileNotFoundError):
        loader.load_patient("nonexistent_patient_id")
