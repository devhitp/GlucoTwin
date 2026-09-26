import pytest
from datetime import datetime
from src.glucotwin.data.schema import CGMRecord

def test_cgm_record():
    record = CGMRecord(patient_id="559", timestamp=datetime(2026, 1, 1), glucose=120.0)
    assert record.patient_id == "559"
    assert record.glucose == 120.0
