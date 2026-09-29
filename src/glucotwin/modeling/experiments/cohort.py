"""
Cohort qualification for Sprint 7.

Determines eligible subjects from the MetaboNet Parquet using
DuckDB (memory-safe, no full-table load) and saves a deterministic
cohort manifest.
"""
import json
import os
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Any

# Minimum eligibility thresholds — documented here, not hard-coded elsewhere
MIN_DURATION_DAYS = 14
MIN_CGM_ROWS = 1000      # ~3.5 days of 5-min readings
MIN_INSULIN_ROWS = 1     # any insulin record
MIN_CARB_ROWS = 1        # any carb record


def build_cohort_manifest(parquet_path: str) -> Dict[str, Any]:
    """
    Queries the Parquet using DuckDB and returns a dict:
      {
        "all_subjects": [...],
        "core_subjects": [...],
        "stats": {...}
      }
    Never loads full 154M rows. Never includes patient timelines.
    """
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT
            id::VARCHAR AS subject_id,
            min(date) AS first_ts,
            max(date) AS last_ts,
            date_diff('hour', min(date), max(date)) AS duration_hours,
            count(CGM)    AS cgm_rows,
            count(insulin) AS ins_rows,
            count(carbs)   AS carb_rows
        FROM '{parquet_path}'
        GROUP BY id
    """).df()

    all_subjects = df["subject_id"].tolist()

    # Core eligibility
    core_mask = (
        (df["duration_hours"] >= MIN_DURATION_DAYS * 24) &
        (df["cgm_rows"] >= MIN_CGM_ROWS) &
        (df["ins_rows"] >= MIN_INSULIN_ROWS) &
        (df["carb_rows"] >= MIN_CARB_ROWS)
    )
    core_df = df[core_mask]
    core_subjects = core_df["subject_id"].tolist()

    stats = {
        "total_subjects": len(all_subjects),
        "core_subjects": len(core_subjects),
        "min_duration_days": float(df["duration_hours"].min() / 24),
        "median_duration_days": float(np.median(df["duration_hours"]) / 24),
        "max_duration_days": float(df["duration_hours"].max() / 24),
        "core_min_duration_days": float(core_df["duration_hours"].min() / 24) if not core_df.empty else None,
        "core_median_duration_days": float(np.median(core_df["duration_hours"]) / 24) if not core_df.empty else None,
        "core_median_cgm_rows": float(np.median(core_df["cgm_rows"])) if not core_df.empty else None,
    }

    return {
        "all_subjects": all_subjects,
        "core_subjects": core_subjects,
        "stats": stats,
    }


def save_manifest(manifest: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save only aggregate stats + pseudonymised subject list (local IDs only)
    with open(output_path, "w") as f:
        json.dump({
            "stats": manifest["stats"],
            "core_subject_count": len(manifest["core_subjects"]),
            # subject list kept local; not included in committed docs
        }, f, indent=2)


def load_core_subjects(parquet_path: str) -> list:
    """Convenience: returns just the list of core-eligible subject IDs."""
    manifest = build_cohort_manifest(parquet_path)
    return manifest["core_subjects"]
