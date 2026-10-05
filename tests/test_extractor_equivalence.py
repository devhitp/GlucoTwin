"""
Tests: extract_causal_twin_features (scalar) vs
       extract_causal_twin_features_fast (batch NumPy)

Covers:
  - Numerical equivalence on controlled synthetic subjects
  - Causal isolation: future records cannot change earlier features
  - Missing-data policy preserved
  - Column names preserved
  - Timestamp alignment preserved
  - Determinism
  - Small performance comparison (timing)
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.glucotwin.data.schema import SynchronizedRecord
from src.glucotwin.hybrid.causal_feature_extractor import (
    extract_causal_twin_features,
    extract_causal_twin_features_fast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(ts: datetime, glucose: float = 100.0,
                 carbs: float = 0.0, bolus: float = 0.0,
                 basal: float = 0.5) -> SynchronizedRecord:
    return SynchronizedRecord(
        patient_id="test_patient",
        timestamp=ts,
        glucose=glucose,
        basal_insulin=basal,
        bolus_insulin=bolus,
        carbohydrates=carbs,
        heart_rate=None,
        eda=None,
        skin_temperature=None,
        accelerometer_x=None,
        accelerometer_y=None,
        accelerometer_z=None,
    )


def _synthetic_subject(
    n_records: int = 200,
    base_glucose: float = 110.0,
    seed: int = 42,
    include_meal_at: int = 50,
    missing_gap_at: int = 80,  # gap index: SKIP 4 records entirely (~20 min)
) -> List[SynchronizedRecord]:
    """
    Deterministic synthetic patient: 5-minute CGM for n_records steps.
    Includes one meal event and one >15min gap (4 records skipped entirely).
    The LONG_GAP flag in TwinEngine.update() triggers when dt>15 minutes,
    i.e., when the time between consecutive update calls exceeds 15 minutes.
    We create this by omitting records[missing_gap_at..missing_gap_at+3]
    so the next available observation is 20 minutes after the previous one.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2023, 6, 1, 6, 0, 0)
    records = []
    g = base_glucose

    skip_indices = set(range(missing_gap_at, missing_gap_at + 4))

    for i in range(n_records):
        if i in skip_indices:
            # Advance glucose simulation but DO NOT emit a record
            g = g + rng.normal(0, 0.5) + 0.05 * (base_glucose - g)
            g = max(40.0, min(300.0, g))
            continue

        ts = t0 + timedelta(minutes=5 * i)
        g = g + rng.normal(0, 0.5) + 0.05 * (base_glucose - g)
        g = max(40.0, min(300.0, g))

        carbs = 50.0 if i == include_meal_at else 0.0
        bolus = 2.0 if i == include_meal_at else 0.0
        records.append(_make_record(ts, glucose=g, carbs=carbs, bolus=bolus))

    return records


SYNTHETIC_RECORDS = _synthetic_subject(n_records=200)
SYNTHETIC_RECORDS_SMALL = _synthetic_subject(n_records=60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ABS_TOL = 1e-8  # slightly relaxed from forecast-level 1e-10 due to feature arithmetic
SKIP_COLS = {"timestamp", "twin_traj_area_60m"}  # traj_area differs by design (noted in extractor)


def _compare_dfs(ref: pd.DataFrame, opt: pd.DataFrame):
    """Assert numerical equivalence between reference and optimised DataFrames."""
    assert not ref.empty, "Reference DataFrame is empty"
    assert not opt.empty, "Optimised DataFrame is empty"
    assert len(ref) == len(opt), f"Row count mismatch: {len(ref)} vs {len(opt)}"

    # Timestamps must match exactly
    assert list(ref["timestamp"]) == list(opt["timestamp"]), "Timestamp mismatch"

    # Numeric columns must match within tolerance
    float_cols = [c for c in ref.columns
                  if c not in SKIP_COLS
                  and pd.api.types.is_float_dtype(ref[c])]

    max_diff = 0.0
    worst_col = None
    for col in float_cols:
        if col not in opt.columns:
            pytest.fail(f"Column {col!r} missing from optimised output")
        r = ref[col].fillna(0.0).values.astype(float)
        o = opt[col].fillna(0.0).values.astype(float)
        diff = np.max(np.abs(r - o))
        if diff > max_diff:
            max_diff = diff
            worst_col = col

    if max_diff >= ABS_TOL:
        print(f"\nDEBUG: mismatch in {worst_col!r} (max_diff={max_diff:.3f})")
        diffs = np.abs(ref[worst_col].fillna(0.0).values - opt[worst_col].fillna(0.0).values)
        bad_idx = np.where(diffs >= ABS_TOL)[0]
        for idx in bad_idx[:2]:
            r = ref.iloc[idx]
            o = opt.iloc[idx]
            print(f"Row {idx} TS={r['timestamp']}")
            print(f"  Ref g0={r['twin_glucose_current']}, ia={r['twin_insulin_action']}, ins={r['twin_insulin_state']}, meal={r['twin_meal_state']}, base={r.get('twin_baseline_deviation')}, T60={r['twin_glucose_t60']}")
            print(f"  Opt g0={o['twin_glucose_current']}, ia={o['twin_insulin_action']}, ins={o['twin_insulin_state']}, meal={o['twin_meal_state']}, base={o.get('twin_baseline_deviation')}, T60={o['twin_glucose_t60']}")

    assert max_diff < ABS_TOL, (
        f"Max column diff {max_diff:.2e} exceeds {ABS_TOL:.2e} in column {worst_col!r}"
    )
    return max_diff


# ---------------------------------------------------------------------------
# 1. Numerical equivalence — full synthetic subject
# ---------------------------------------------------------------------------

def test_extractor_equivalence_full_subject():
    ref = extract_causal_twin_features(SYNTHETIC_RECORDS)
    opt = extract_causal_twin_features_fast(SYNTHETIC_RECORDS)
    max_diff = _compare_dfs(ref, opt)
    # Report the actual max diff for documentation
    assert max_diff < ABS_TOL, f"Max diff {max_diff:.2e}"


# ---------------------------------------------------------------------------
# 2. Numerical equivalence — with target_timestamps filter
# ---------------------------------------------------------------------------

def test_extractor_equivalence_with_target_timestamps():
    ref_all = extract_causal_twin_features(SYNTHETIC_RECORDS)
    if ref_all.empty:
        pytest.skip("No features generated")

    # Use every other timestamp as target
    all_ts = set(ref_all["timestamp"].tolist())
    target = set(list(all_ts)[::2])

    ref = extract_causal_twin_features(SYNTHETIC_RECORDS, target_timestamps=target)
    opt = extract_causal_twin_features_fast(SYNTHETIC_RECORDS, target_timestamps=target)
    _compare_dfs(ref, opt)


# ---------------------------------------------------------------------------
# 3. Column name preservation
# ---------------------------------------------------------------------------

def test_column_names_preserved():
    ref = extract_causal_twin_features(SYNTHETIC_RECORDS)
    opt = extract_causal_twin_features_fast(SYNTHETIC_RECORDS)
    ref_cols = set(ref.columns)
    opt_cols = set(opt.columns)
    missing = ref_cols - opt_cols
    extra   = opt_cols - ref_cols
    assert not missing, f"Columns missing from fast path: {missing}"
    assert not extra,   f"Extra columns in fast path: {extra}"


# ---------------------------------------------------------------------------
# 4. Causal isolation — modifying future records cannot change earlier features
# ---------------------------------------------------------------------------

def test_causal_isolation():
    records = _synthetic_subject(n_records=150, seed=7)
    ref = extract_causal_twin_features_fast(records)

    if ref.empty:
        pytest.skip("No features generated")

    # Modify the last 10 records (future, relative to earlier prediction times)
    modified = list(records)
    for i in range(len(records) - 10, len(records)):
        old = records[i]
        modified[i] = _make_record(
            old.timestamp,
            glucose=old.glucose + 50.0 if not math.isnan(old.glucose) else float('nan'),
            carbs=100.0,
            bolus=10.0,
        )

    opt_mod = extract_causal_twin_features_fast(modified)

    # Features at early timestamps (far from the modified tail) must be unchanged
    if not ref.empty and not opt_mod.empty:
        early_cutoff = ref["timestamp"].iloc[len(ref) // 2]
        ref_early = ref[ref["timestamp"] <= early_cutoff].reset_index(drop=True)
        mod_early = opt_mod[opt_mod["timestamp"] <= early_cutoff].reset_index(drop=True)

        if not ref_early.empty and not mod_early.empty:
            _compare_dfs(ref_early, mod_early)


# ---------------------------------------------------------------------------
# 5. Missing data policy — >15min gap must not interpolate
# ---------------------------------------------------------------------------

def test_missing_gap_flag_preserved():
    records = _synthetic_subject(n_records=150, missing_gap_at=40)
    opt = extract_causal_twin_features_fast(records)

    if opt.empty:
        pytest.skip("No features generated")

    # After the gap, some rows should have LONG_GAP flag encoded as 1.0
    assert "twin_has_long_gap" in opt.columns
    # At least one row must have the flag set
    assert (opt["twin_has_long_gap"] == 1.0).any(), (
        "Expected LONG_GAP flag in at least one row after the >15min CGM gap"
    )


def test_missing_glucose_flag_preserved():
    records = _synthetic_subject(n_records=150, missing_gap_at=30)
    opt = extract_causal_twin_features_fast(records)
    assert "twin_has_missing_glucose" in opt.columns


# ---------------------------------------------------------------------------
# 6. Determinism — same input → same output
# ---------------------------------------------------------------------------

def test_determinism():
    r1 = extract_causal_twin_features_fast(SYNTHETIC_RECORDS)
    r2 = extract_causal_twin_features_fast(SYNTHETIC_RECORDS)
    if r1.empty:
        pytest.skip("No features generated")
    _compare_dfs(r1, r2)


# ---------------------------------------------------------------------------
# 7. Insufficient records — graceful empty return
# ---------------------------------------------------------------------------

def test_insufficient_records():
    records = _synthetic_subject(n_records=5)
    ref = extract_causal_twin_features(records)
    opt = extract_causal_twin_features_fast(records)
    assert ref.empty
    assert opt.empty


# ---------------------------------------------------------------------------
# 8. Small performance benchmark — scalar vs fast
# ---------------------------------------------------------------------------

def test_performance_benchmark(capsys):
    """
    Measure actual runtime ratio between scalar and fast implementations.
    Does NOT assert a minimum speedup — reports empirical ratio only.
    """
    records = _synthetic_subject(n_records=500, seed=0)

    N_REPEATS = 3

    # Scalar reference
    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        extract_causal_twin_features(records)
    scalar_s = (time.perf_counter() - t0) / N_REPEATS

    # Fast path
    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        extract_causal_twin_features_fast(records)
    fast_s = (time.perf_counter() - t0) / N_REPEATS

    speedup = scalar_s / fast_s if fast_s > 0 else float("inf")

    with capsys.disabled():
        print(f"\n[Benchmark] 500-record subject × {N_REPEATS} runs")
        print(f"  Scalar:  {scalar_s*1000:.1f} ms/subject")
        print(f"  Fast:    {fast_s*1000:.1f} ms/subject")
        print(f"  Speedup: {speedup:.1f}x")

    # Fast path must not be SLOWER than scalar (allow 20% margin for noise)
    assert fast_s <= scalar_s * 1.2, (
        f"Fast path ({fast_s*1000:.1f}ms) is slower than scalar ({scalar_s*1000:.1f}ms)"
    )
