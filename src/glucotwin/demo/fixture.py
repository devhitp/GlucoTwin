"""
Demo fixture for GlucoTwin dashboard.

Generates a deterministic, reproducible, patient-free research demo sequence.
Safe to commit. No raw patient data. No personal identifiers.
Based on physiologically plausible glucose/insulin/meal dynamics.
"""
from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.glucotwin.data.schema import SynchronizedRecord

DEMO_SEED = 2026
DEMO_PATIENT_ID = "demo_research_subject_01"
T0 = datetime(2026, 1, 15, 20, 0, 0)  # 8 PM — nocturnal demo window


def generate_demo_records(n_hours: float = 6.0, seed: int = DEMO_SEED) -> List[SynchronizedRecord]:
    """
    Generate a deterministic, synthetic, patient-free research demo sequence.

    Simulates a Type 1 Diabetes-like scenario:
    - Stable glucose in early evening
    - Evening meal (carbohydrate pulse at T+30m)
    - Insulin bolus shortly after (T+40m)
    - Glucose rises, then falls
    - Nocturnal glucose dip at T+3h (hypoglycemia-risk window)

    Returns:
        List[SynchronizedRecord] — 5-minute CGM records (~n_hours * 12 records)
    """
    rng = np.random.default_rng(seed)
    records = []

    n_steps = int(n_hours * 60 / 5)
    t = T0

    # -- Synthetic glucose trajectory --
    # Phase 1: stable baseline ~135 (dataset units, provisional)
    # Phase 2: meal-driven rise to ~190
    # Phase 3: insulin action drives fall through ~80 (hypoglycemia risk zone)
    # Phase 4: slow recovery
    baseline = 135.0
    glucose = baseline

    # Pre-computed trajectory parameters
    meal_carbs_event = 40.0          # units (carbohydrate)
    meal_at_step = 6                 # T+30m
    bolus_at_step = 8                # T+40m
    bolus_amount = 4.0               # units (insulin bolus)
    basal_rate = 0.8                 # units/hr

    insulin_state = 0.0
    insulin_action = 0.0
    meal_state = 0.0

    for i in range(n_steps):
        # Insulin/meal inputs
        new_carbs = meal_carbs_event if i == meal_at_step else 0.0
        new_bolus = bolus_amount if i == bolus_at_step else 0.0
        basal_5m = basal_rate * (5.0 / 60.0)

        # State dynamics (matches TwinDynamics coefficients)
        meal_decay = 0.05 * meal_state
        meal_state = max(0.0, meal_state - meal_decay + new_carbs)

        ins_input = new_bolus + basal_5m
        ins_decay = 0.1 * insulin_state
        insulin_state = max(0.0, insulin_state - ins_decay + ins_input)

        action_decay = 0.1 * insulin_action
        action_input = 0.1 * insulin_state
        insulin_action = max(0.0, insulin_action - action_decay + action_input)

        # Glucose dynamics
        drift = 0.05 * (baseline - glucose)
        meal_effect = 2.0 * meal_decay
        ins_effect = 1.0 * insulin_action

        glucose = max(45.0, glucose + drift + meal_effect - ins_effect)

        # Add small noise
        glucose_obs = glucose + rng.normal(0, 2.0)
        glucose_obs = max(40.0, min(glucose_obs, 400.0))

        records.append(SynchronizedRecord(
            patient_id=DEMO_PATIENT_ID,
            timestamp=t,
            glucose=round(glucose_obs, 1),
            basal_insulin=round(basal_rate, 2),
            bolus_insulin=round(new_bolus, 2),
            carbohydrates=round(new_carbs, 1),
            heart_rate=None,
            eda=None,
            skin_temperature=None,
            accelerometer_x=None,
            accelerometer_y=None,
            accelerometer_z=None,
        ))

        t += timedelta(minutes=5)

    return records


def get_demo_summary(records: List[SynchronizedRecord]) -> dict:
    """Return a summary dict for display in the dashboard."""
    glucose_vals = [r.glucose for r in records if r.glucose is not None]
    return {
        "subject_id": DEMO_PATIENT_ID,
        "n_records": len(records),
        "duration_hours": len(records) * 5 / 60,
        "glucose_min": round(min(glucose_vals), 1),
        "glucose_max": round(max(glucose_vals), 1),
        "glucose_mean": round(sum(glucose_vals) / len(glucose_vals), 1),
        "start_time": str(records[0].timestamp),
        "end_time": str(records[-1].timestamp),
        "note": "Synthetic research demo — not patient data.",
    }
