"""
Tests for the GlucoTwin dashboard and end-to-end demo logic.
"""
import pytest
from src.glucotwin.demo.fixture import generate_demo_records, get_demo_summary
from src.glucotwin.demo.app_logic import (
    build_twin_from_records, 
    run_demo_inference, 
    run_whatif_simulation
)
from src.glucotwin.counterfactual.scenarios import CounterfactualScenario


def test_demo_fixture_generation():
    """Verify the deterministic demo fixture produces valid records."""
    records = generate_demo_records(n_hours=2.0)
    assert len(records) == 24  # 2 hours * 12 records/hour
    summary = get_demo_summary(records)
    assert summary["n_records"] == 24
    assert summary["subject_id"] == "demo_research_subject_01"


def test_dashboard_end_to_end_inference():
    """Verify the end-to-end inference pipeline runs on demo data."""
    records = generate_demo_records(n_hours=4.0)
    result = run_demo_inference(records)

    assert "twin_state" in result
    assert "trajectory_30m" in result
    assert "trajectory_60m" in result
    assert "data_quality" in result
    assert result["data_quality"]["cgm_availability_pct"] == 100.0
    
    # Verify the twin forecast points are offset by correct horizons
    # 30m forecast = 6 points at 5-min intervals
    assert len(result["trajectory_30m"]) == 6
    assert len(result["trajectory_60m"]) == 12


def test_dashboard_whatif_simulation():
    """Verify the What-If simulation executes correctly and baseline != counterfactual."""
    records = generate_demo_records(n_hours=4.0)
    scenario = CounterfactualScenario.meal_perturbation(carbs=20.0, offset_minutes=5)
    
    res = run_whatif_simulation(records, scenario)
    
    assert "baseline" in res
    assert "counterfactual" in res
    assert len(res["baseline"]) > 0
    assert len(res["baseline"]) == len(res["counterfactual"])
    
    # Delta should be non-zero since a meal was added
    has_diff = any(d != 0.0 for _, d in res["delta"])
    assert has_diff, "Counterfactual trajectory must diverge from baseline"


def test_invalid_whatif_scenario_rejected():
    """Verify negative carbs/bolus are safely rejected."""
    with pytest.raises(ValueError, match="requires meal_carbs >= 0"):
        CounterfactualScenario.meal_perturbation(carbs=-10.0, offset_minutes=5).validate()
        
    with pytest.raises(ValueError, match="requires insulin_bolus >= 0"):
        CounterfactualScenario.insulin_timing(bolus=-2.0, offset_minutes=5).validate()
