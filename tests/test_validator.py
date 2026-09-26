import pytest
from src.glucotwin.data.validator import DataValidator
from src.glucotwin.data.schema import CGMRecord
from datetime import datetime

def test_validator_valid():
    records = {
        "cgm": [CGMRecord("1", datetime(2026, 1, 1), 100)]
    }
    report = DataValidator.validate_records(records)
    assert report["valid"]

def test_validator_invalid_negative():
    records = {
        "cgm": [CGMRecord("1", datetime(2026, 1, 1), -10)]
    }
    report = DataValidator.validate_records(records)
    assert not report["valid"]
    assert len(report["errors"]) == 1

def test_validator_chronological():
    records = {
        "cgm": [
            CGMRecord("1", datetime(2026, 1, 2), 100),
            CGMRecord("1", datetime(2026, 1, 1), 110)
        ]
    }
    report = DataValidator.validate_records(records)
    assert not report["valid"]
    assert "Non-chronological" in report["errors"][0]
