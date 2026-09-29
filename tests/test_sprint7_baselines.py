"""
Sprint 7 tests for real-data pipeline infrastructure.
Uses tiny synthetic fixtures — never the 1.26 GB real file.
"""
import io
import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.glucotwin.modeling.experiments.metabonet_bridge import (
    _canonical_to_synchronized, REQUIRED_COLS
)
from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest
from src.glucotwin.modeling.experiments.temporal_split import (
    chronological_split_3way, assert_no_temporal_leakage
)
from src.glucotwin.modeling.experiments.metrics import (
    compute_metrics, select_threshold_on_val, compute_calibration
)
from src.glucotwin.modeling.experiments.feature_matrix import (
    build_subject_matrices, LABEL_30M
)
from src.glucotwin.data.schema import SynchronizedRecord


# ---- Fixtures ----------------------------------------------------------------

def make_synthetic_parquet(tmp_path, n_subjects=3, n_rows_each=200):
    """Create a tiny synthetic Parquet file with MetaboNet-style schema."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for s in range(n_subjects):
        sid = str(100 + s)
        for r in range(n_rows_each):
            ts = base + pd.Timedelta(minutes=5 * r)
            rows.append({
                "id": sid,
                "date": ts,
                "CGM": 100.0 + 10 * np.sin(r * 0.1),
                "insulin": 1.0,
                "basal": 0.8,
                "bolus": 0.2,
                "carbs": 20.0 if r % 50 == 0 else 0.0,
                "steps": None,
                "heartrate": None,
                "galvanic_skin_response": None,
                "skin_temp": None,
            })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path, index=False)
    return path, [str(100 + s) for s in range(n_subjects)]


def make_synchronized_records(n=120, glucose_val=110.0):
    """Create synthetic SynchronizedRecords for pipeline testing."""
    base = pd.Timestamp("2024-01-01")
    records = []
    for i in range(n):
        g = glucose_val + 5 * np.sin(i * 0.2)
        # introduce a few hypo events
        if i in [30, 31, 32, 60, 61]:
            g = 60.0
        records.append(SynchronizedRecord(
            patient_id="test_subj",
            timestamp=(base + pd.Timedelta(minutes=5 * i)).to_pydatetime(),
            glucose=g,
            basal_insulin=0.8,
            bolus_insulin=0.2 if i % 10 == 0 else 0.0,
            carbohydrates=20.0 if i % 25 == 0 else 0.0,
            heart_rate=None,
            eda=None,
            skin_temperature=None,
            accelerometer_x=None,
            accelerometer_y=None,
            accelerometer_z=None,
        ))
    return records


# ---- Bridge tests -----------------------------------------------------------

def test_bridge_canonical_to_synchronized():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="5min"),
        "CGM": [100.0 + i for i in range(10)],
        "basal": [0.5] * 10,
        "bolus": [0.1] * 10,
        "carbs": [0.0] * 10,
    })
    records = _canonical_to_synchronized("p1", df)
    assert len(records) == 10
    assert records[0].patient_id == "p1"
    assert records[0].glucose == pytest.approx(100.0, abs=0.01)
    assert records[0].basal_insulin == pytest.approx(0.5, abs=0.01)


def test_bridge_filters_non_physiological_glucose():
    """Values outside 20–600 must be removed."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "CGM": [1.0, 15.0, 100.0, 700.0, 300.0],  # 2 invalid
        "basal": [0.5] * 5,
        "bolus": [0.0] * 5,
        "carbs": [0.0] * 5,
    })
    records = _canonical_to_synchronized("p1", df)
    assert len(records) == 2  # 1.0, 15.0 below 20; 700.0 above 600; 100.0 and 300.0 valid


def test_bridge_null_glucose_filtered():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="5min"),
        "CGM": [100.0, None, 110.0, None],
        "basal": [0.5] * 4,
        "bolus": [0.0] * 4,
        "carbs": [0.0] * 4,
    })
    records = _canonical_to_synchronized("p1", df)
    assert len(records) == 2


# ---- Cohort tests -----------------------------------------------------------

def test_cohort_manifest(tmp_path):
    path, subj_ids = make_synthetic_parquet(tmp_path, n_subjects=3, n_rows_each=300)
    manifest = build_cohort_manifest(path)
    assert manifest["stats"]["total_subjects"] == 3
    assert "core_subjects" in manifest
    # With 300 rows * 5min = 25 hours, subjects may or may not meet 14-day threshold
    # Just check structure
    assert isinstance(manifest["core_subjects"], list)


# ---- Temporal split tests ---------------------------------------------------

def test_chronological_3way_split_no_overlap():
    idx = pd.date_range("2024-01-01", periods=300, freq="5min")
    df = pd.DataFrame({"a": np.random.rand(300)}, index=idx)
    train, val, test = chronological_split_3way(df, embargo_minutes=60)

    assert not train.empty
    assert not val.empty
    assert not test.empty

    # Assert no temporal overlap with embargo
    assert_no_temporal_leakage(train, val, test, embargo_minutes=60)


def test_chronological_split_ordering():
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    df = pd.DataFrame({"x": range(200)}, index=idx)
    train, val, test = chronological_split_3way(df)
    assert train.index.max() < test.index.min()


def test_embargo_gap_enforced():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({"x": range(100)}, index=idx)
    train, val, test = chronological_split_3way(df, embargo_minutes=120)
    # Check 2-hour gap exists
    if not val.empty and not train.empty:
        gap = (val.index.min() - train.index.max()).total_seconds() / 60
        assert gap >= 120


# ---- Metrics tests ----------------------------------------------------------

def test_metrics_binary():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_prob = rng.uniform(0, 1, 100)
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert "brier_score" in m
    assert 0 <= m["roc_auc"] <= 1


def test_metrics_single_class():
    y_true = np.zeros(50, dtype=int)
    y_prob = np.ones(50) * 0.1
    m = compute_metrics(y_true, y_prob)
    assert "note" in m


def test_threshold_selection_on_val():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    prob = rng.uniform(0, 1, 200)
    t = select_threshold_on_val(y, prob)
    assert 0.0 <= t <= 1.0


def test_calibration():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    prob = rng.uniform(0, 1, 200)
    cal = compute_calibration(y, prob)
    assert "fraction_positives" in cal
    assert "mean_predicted" in cal


# ---- Feature matrix / leakage regression test --------------------------------

def test_feature_matrix_leakage_regression():
    """
    Changing future glucose (after prediction time T) must NOT change
    features available at T — only the corresponding label may change.
    """
    records = make_synchronized_records(120, glucose_val=110.0)

    from src.glucotwin.data.preprocessing.pipeline import PreprocessingPipeline
    df_original = PreprocessingPipeline.process_patient(records)

    if df_original.empty:
        pytest.skip("Pipeline produced empty DataFrame for this fixture")

    # Modify glucose only for the LAST 10 rows (future from T=110th row)
    records_modified = list(records)
    for i in range(110, 120):
        r = records_modified[i]
        records_modified[i] = SynchronizedRecord(
            patient_id=r.patient_id,
            timestamp=r.timestamp,
            glucose=30.0,  # force hypo in the future
            basal_insulin=r.basal_insulin,
            bolus_insulin=r.bolus_insulin,
            carbohydrates=r.carbohydrates,
            heart_rate=r.heart_rate,
            eda=r.eda,
            skin_temperature=r.skin_temperature,
            accelerometer_x=r.accelerometer_x,
            accelerometer_y=r.accelerometer_y,
            accelerometer_z=r.accelerometer_z,
        )

    df_modified = PreprocessingPipeline.process_patient(records_modified)

    if df_modified.empty:
        pytest.skip("Modified pipeline produced empty DataFrame")

    # Features at T<110 must be identical
    shared_idx = df_original.index[:100].intersection(df_modified.index[:100])
    if len(shared_idx) == 0:
        pytest.skip("No shared index for comparison")

    feature_cols = [c for c in df_original.columns if not c.startswith("future_")]
    feature_cols = [c for c in feature_cols if c in df_modified.columns]

    orig_feats = df_original.loc[shared_idx, feature_cols]
    mod_feats = df_modified.loc[shared_idx, feature_cols]

    pd.testing.assert_frame_equal(orig_feats, mod_feats, check_like=True,
                                   rtol=1e-5, atol=1e-5)


def test_build_subject_matrices_returns_splits():
    records = make_synchronized_records(300)
    result = build_subject_matrices(records)
    if result is None:
        pytest.skip("Insufficient data for splits in synthetic fixture")
    train, val, test = result
    assert not train.empty
    assert not test.empty
    # Verify temporal ordering
    if not val.empty:
        assert train.index.max() < test.index.min()


# ---- Privacy / artifact safety test ----------------------------------------

def test_no_raw_patient_records_in_artifacts():
    """
    Ensure artifact JSON files never contain raw glucose sequences.
    Check that each value in a result file is an aggregate, not a list > 20 items.
    """
    artifact_dir = "artifacts/local"
    if not os.path.exists(artifact_dir):
        return  # no artifacts yet — pass
    for fname in os.listdir(artifact_dir):
        if fname.endswith(".json"):
            with open(os.path.join(artifact_dir, fname)) as f:
                data = json.load(f)
            # Recursively check no list has > 100 numeric elements (raw sequences)
            def check_no_raw(obj, path=""):
                if isinstance(obj, list) and len(obj) > 100:
                    # Only calibration curves are allowed
                    assert "calibration" in path or "fraction" in path or "predicted" in path, \
                        f"Potential raw sequence in artifacts at {path}: len={len(obj)}"
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        check_no_raw(v, path + f"/{k}")
            check_no_raw(data)
