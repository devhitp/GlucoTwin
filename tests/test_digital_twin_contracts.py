import pytest
from src.glucotwin.config.clinical import get_hypo_threshold, get_glucose_unit, THRESHOLD_STATUS

def test_unknown_cgm_unit_preserved():
    assert get_glucose_unit() == "unknown", "CGM unit must remain unconfirmed per MetaboNet contract"

def test_provisional_threshold_status_preserved():
    assert THRESHOLD_STATUS == "provisional", "Threshold must be flagged as provisional"

def test_hypo_threshold_value():
    assert get_hypo_threshold() == 70.0, "Engineering validation assumes 70.0 temporarily"

def test_no_clinical_claims():
    assert "clinical" not in THRESHOLD_STATUS.lower()
    assert "validate" not in THRESHOLD_STATUS.lower()
