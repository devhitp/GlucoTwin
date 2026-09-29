"""
Sprint 10 counterfactual smoke test — real data.
Uses a minimal number of subjects. Saves only aggregate artifacts.
Raw patient trajectories are NOT saved.
"""
import math
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.glucotwin.modeling.experiments.metabonet_bridge import iter_subjects_from_parquet
from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest
from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinParameters
from src.glucotwin.twin.observations import TwinObservation
from src.glucotwin.data.schema import SynchronizedRecord
from src.glucotwin.counterfactual import (
    CounterfactualScenario, CounterfactualSimulator
)
import numpy as np

PARQUET_PATH = "data/raw/metabonet_public.parquet"
ARTIFACTS_DIR = "artifacts/local"
N_SUBJECTS = 5  # Minimal smoke test


def obs_from_record(rec: SynchronizedRecord) -> TwinObservation:
    g = rec.glucose
    return TwinObservation(
        timestamp=rec.timestamp,
        glucose=g if not math.isnan(g) else None,
        basal=rec.basal_insulin,
        bolus=rec.bolus_insulin,
        carbohydrates=rec.carbohydrates,
    )


def main():
    print("=" * 56)
    print("Sprint 10 — Counterfactual Smoke Test (Real Data)")
    print("Exploratory model simulation only. Not medical advice.")
    print("=" * 56)

    manifest = build_cohort_manifest(PARQUET_PATH)
    subjects = manifest["core_subjects"][:N_SUBJECTS]

    params = TwinParameters.default_population_params()
    sim = CounterfactualSimulator(params)

    successful = 0
    rejected = 0
    t0 = time.time()

    for subj_id, records in iter_subjects_from_parquet(PARQUET_PATH, subject_ids=subjects):
        try:
            observations = sorted(
                [obs_from_record(r) for r in records],
                key=lambda o: o.timestamp
            )
            valid = [o for o in observations if not o.is_glucose_missing()]
            if len(valid) < 10:
                rejected += 1
                continue

            # Initialize and run forward to get a state
            engine = TwinEngine(TwinInitializer.initialize(valid[:10], params), params)
            for obs in valid[10:30]:
                engine.update(obs)

            state = engine.current_state

            # Baseline
            baseline_res = sim.simulate(state, CounterfactualScenario.baseline(), 60)

            # Meal perturbation
            meal_res = sim.simulate(
                state,
                CounterfactualScenario.meal_perturbation(50.0, 10.0),
                60
            )

            # Insulin timing
            ins_res = sim.simulate(
                state,
                CounterfactualScenario.insulin_timing(2.0, 10.0),
                60
            )

            # Verify numerical stability
            for g in baseline_res.baseline_glucose + meal_res.counterfactual_glucose + ins_res.counterfactual_glucose:
                assert not math.isnan(g) and not math.isinf(g)

            # Aggregate only: max delta, direction of effect
            meal_max_delta = max(meal_res.delta_glucose)
            ins_max_delta  = min(ins_res.delta_glucose)

            print(f"  Subject {subj_id}: baseline_g0={state.glucose:.1f}  "
                  f"meal_max_delta=+{meal_max_delta:.2f}  "
                  f"ins_max_delta={ins_max_delta:.2f}")
            successful += 1

        except Exception as e:
            print(f"  Subject {subj_id}: REJECTED — {e}")
            rejected += 1

    elapsed = time.time() - t0

    # Save aggregate artifact only (no raw trajectories)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    artifact = {
        "stage": "smoke",
        "subjects_attempted": N_SUBJECTS,
        "successful": successful,
        "rejected": rejected,
        "runtime_seconds": round(elapsed, 2),
        "note": "Exploratory model simulation only. Not medical advice.",
    }
    out = os.path.join(ARTIFACTS_DIR, "counterfactual_smoke.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nResult: {successful}/{N_SUBJECTS} successful  "
          f"Rejected: {rejected}  Runtime: {elapsed:.1f}s")
    print(f"Artifact: {out}")
    print("=" * 56)


if __name__ == "__main__":
    main()
