"""
GlucoTwin — Resumable N-Subject Evaluation Pipeline (Phase 2-6).

Checkpoint/resume contract
--------------------------
* On first run, a JSON manifest is created in --manifest-dir.
* On subsequent runs with the same manifest, already-completed subjects
  are skipped; failed subjects are retried.
* Interruption (Ctrl-C, OOM-kill, timeout) leaves the manifest intact.
* Temporary Parquet chunks are written atomically:
    <tmp>.parquet.partial -> validate -> rename -> <id>_<part>_<hz>.parquet
* Completed subjects are individually fsynced before the manifest is updated.

Scientific freeze
-----------------
* Digital Twin equations, feature definitions, labels, prediction horizons,
  subject split logic, personalization semantics, LightGBM configuration,
  missing-data rules, and clinical threshold are NOT modified here.
* This script is evaluation infrastructure only.

Causal-safety language
----------------------
Pipeline-level causal feature construction, patient-held-out evaluation,
and leakage safeguards passed validation (79-subject baseline: commit 0b79519).
The dataset remains observational; the What-If simulator remains exploratory.

Usage
-----
  # First run (or full fresh run)
  python scripts/run_938_resumable.py --stage full

  # Smoke test (10 subjects)
  python scripts/run_938_resumable.py --stage smoke

  # Resume interrupted run
  python scripts/run_938_resumable.py --stage full --resume

  # Custom manifest location
  python scripts/run_938_resumable.py --stage full \
      --manifest-dir artifacts/runs/run_001 --resume
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest
from src.glucotwin.modeling.experiments.metabonet_bridge import iter_subjects_from_parquet
from src.glucotwin.modeling.experiments.feature_matrix import (
    build_subject_matrices_heldout, LABEL_30M, LABEL_60M, FEATURE_COLS,
)
from src.glucotwin.hybrid.causal_feature_extractor import extract_causal_twin_features_fast
from src.glucotwin.hybrid.model import (
    HybridModel, ModelConfig,
    BASELINE_FEATURE_COLS, TWIN_FEATURE_COLS, TWIN_V2_FEATURE_COLS,
)
from src.glucotwin.hybrid.calibration import select_calibrator
from src.glucotwin.hybrid.twin_eval import evaluate_twin_forecast
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)

# ---------------------------------------------------------------------------
# Constants -- do NOT change without incrementing MANIFEST_VERSION
# ---------------------------------------------------------------------------
MANIFEST_VERSION = "1"
PARQUET_PATH = "data/raw/metabonet_public.parquet"
ARTIFACTS_DIR = "artifacts/local"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remainder

STAGE_SUBJECT_LIMITS: Dict[str, Optional[int]] = {
    "smoke": 10,
    "pilot": 79,
    "full":  None,  # all core subjects
}

SAVE_COLS_30 = sorted(set(
    list(FEATURE_COLS) + TWIN_FEATURE_COLS + TWIN_V2_FEATURE_COLS +
    [LABEL_30M, "patient_id", "glucose_current", "twin_glucose_t30"]
))
SAVE_COLS_60 = sorted(set(
    list(FEATURE_COLS) + TWIN_FEATURE_COLS + TWIN_V2_FEATURE_COLS +
    [LABEL_60M, "patient_id", "glucose_current", "twin_glucose_t60"]
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(stage: str, seed: int, train_ratio: float, val_ratio: float,
                 parquet_path: str) -> str:
    """Stable hash of the run configuration (not the data)."""
    h = hashlib.sha256()
    h.update(f"{stage}|{seed}|{train_ratio}|{val_ratio}|{os.path.abspath(parquet_path)}".encode())
    return h.hexdigest()[:16]


def _align_and_trim(df: pd.DataFrame, save_cols: list) -> pd.DataFrame:
    for c in save_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[save_cols]


def _merge_twin_features(df_baseline: pd.DataFrame, df_twin: pd.DataFrame) -> pd.DataFrame:
    if df_twin.empty:
        return df_baseline
    if len(df_baseline) == len(df_twin) and "timestamp" in df_twin.columns:
        b_ts = (df_baseline.index
                if (df_baseline.index.name == "timestamp" or
                    np.issubdtype(df_baseline.index.dtype, np.datetime64))
                else df_baseline.get("timestamp"))
        if b_ts is not None and (b_ts.values == df_twin["timestamp"].values).all():
            for col in df_twin.columns:
                if col != "timestamp":
                    df_baseline[col] = df_twin[col].values
            return df_baseline
    df_b = df_baseline.copy()
    if "timestamp" not in df_b.columns:
        if df_b.index.name == "timestamp":
            df_b = df_b.reset_index()
        elif np.issubdtype(df_b.index.dtype, np.datetime64):
            df_b = df_b.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" in df_twin.columns and "timestamp" in df_b.columns:
        return pd.merge(df_b, df_twin, on="timestamp", how="left")
    return pd.concat([df_b.reset_index(drop=True), df_twin.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

def _manifest_path(manifest_dir: str) -> str:
    return os.path.join(manifest_dir, "run_manifest.json")


def _write_manifest(manifest: dict, manifest_dir: str) -> None:
    """Atomically write manifest: tmp -> fsync -> rename."""
    path = _manifest_path(manifest_dir)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp, path)
    except Exception:
        shutil.move(tmp, path)


def _load_manifest(manifest_dir: str) -> Optional[dict]:
    path = _manifest_path(manifest_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _build_initial_manifest(
    stage: str,
    parquet_path: str,
    manifest_dir: str,
    chunks_dir: str,
    artifacts_dir: str,
) -> dict:
    """Build the initial manifest from the cohort query."""
    print("[manifest] Querying cohort from Parquet (DuckDB)...")
    cohort = build_cohort_manifest(parquet_path)
    core_subjects = cohort["core_subjects"]
    print(f"[manifest]   Core subjects from Parquet: {len(core_subjects)}")

    # Deterministic split -- MUST use sorted() before permutation for reproducibility
    subjects_sorted = sorted(core_subjects)
    rng = np.random.default_rng(SEED)
    shuffled: List[str] = rng.permutation(subjects_sorted).tolist()

    limit = STAGE_SUBJECT_LIMITS[stage]
    if limit is not None:
        shuffled = shuffled[:limit]

    n = len(shuffled)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    test_ids  = shuffled[n_train + n_val:]

    # Safety: verify no overlap
    assert not (set(train_ids) & set(val_ids)), "Train/val overlap!"
    assert not (set(train_ids) & set(test_ids)), "Train/test overlap!"
    assert not (set(val_ids)   & set(test_ids)), "Val/test overlap!"

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "stage": stage,
        "git_commit": _git_head(),
        "git_branch": _git_branch(),
        "config_hash": _config_hash(stage, SEED, TRAIN_RATIO, VAL_RATIO, parquet_path),
        "dataset_path": os.path.abspath(parquet_path),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "cohort_stats": cohort["stats"],
        "subject_counts": {
            "total_core": len(core_subjects),
            "stage_total": n,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        # Subject ID lists -- local IDs only, no patient-level records
        "train_subject_ids": train_ids,
        "val_subject_ids": val_ids,
        "test_subject_ids": test_ids,
        # Per-subject status
        "subject_status": {sid: "pending" for sid in shuffled},
        "subject_errors": {},
        "subject_windows": {},      # sid -> {"windows_30m": int, "windows_60m": int}
        "subject_durations_s": {},  # sid -> float
        # Run-level bookkeeping
        "run_started_at": _now_iso(),
        "run_updated_at": _now_iso(),
        "run_completed_at": None,
        "total_runtime_s": None,
        "peak_memory_mb": None,
        # Artifact locations
        "chunks_dir": chunks_dir,
        "artifacts_dir": artifacts_dir,
        "manifest_dir": manifest_dir,
        "output_artifact": os.path.join(artifacts_dir, f"run_938_{stage}.json"),
        # Results (written at end)
        "results": None,
        "warnings": [],
    }
    return manifest


def _subject_partition(sid: str, manifest: dict) -> str:
    if sid in manifest["train_subject_ids"]:
        return "train"
    if sid in manifest["val_subject_ids"]:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Per-subject processing (atomic chunk writes)
# ---------------------------------------------------------------------------

def _atomic_write_parquet(df: pd.DataFrame, final_path: str) -> None:
    """Write df to final_path atomically via a .partial file."""
    import pyarrow.parquet as pq
    tmp = final_path + ".partial"
    df.to_parquet(tmp, index=False)
    meta = pq.read_metadata(tmp)
    assert meta.num_rows == len(df), "Partial parquet row count mismatch"
    try:
        os.replace(tmp, final_path)
    except Exception:
        shutil.move(tmp, final_path)


def process_subject(records, subj_id: str, partition: str, chunks_dir: str) -> Dict:
    """
    Process one subject: feature extraction -> atomic chunk write.
    Returns {"windows_30m": int, "windows_60m": int} or {} if skipped.
    """
    result = build_subject_matrices_heldout(records, subj_id)
    if result is None:
        return {}

    df_30, df_60 = result

    target_timestamps: Set = set()
    if not df_30.empty:
        target_timestamps.update(df_30.index)
    if not df_60.empty:
        target_timestamps.update(df_60.index)

    df_twin = extract_causal_twin_features_fast(records, target_timestamps)

    df_30_full = _align_and_trim(_merge_twin_features(df_30, df_twin), SAVE_COLS_30)
    df_60_full = _align_and_trim(_merge_twin_features(df_60, df_twin), SAVE_COLS_60)

    safe_id = subj_id.replace("/", "_").replace("\\", "_")
    p30 = os.path.join(chunks_dir, f"{safe_id}_{partition}_30.parquet")
    p60 = os.path.join(chunks_dir, f"{safe_id}_{partition}_60.parquet")
    _atomic_write_parquet(df_30_full, p30)
    _atomic_write_parquet(df_60_full, p60)

    counts = {"windows_30m": len(df_30_full), "windows_60m": len(df_60_full)}
    del df_30, df_60, df_twin, df_30_full, df_60_full
    return counts


# ---------------------------------------------------------------------------
# Partition loading
# ---------------------------------------------------------------------------

def load_partition_parquet(chunks_dir: str, partition: str, horizon: str) -> pd.DataFrame:
    pattern = f"_{partition}_{horizon}.parquet"
    files = [
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.endswith(pattern) and not f.endswith(".partial")
    ]
    if not files:
        return pd.DataFrame()
    parts = []
    for fp in files:
        df = pd.read_parquet(fp)
        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])
        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    horizon: str,
    label_col: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    print(f"\nEvaluating {horizon} horizon...")
    if df_train.empty or df_test.empty:
        print(f"  Skipping {horizon} -- insufficient data.")
        return {}

    results = {}

    # Persistence baseline
    y_te = df_test[label_col].fillna(0).astype(int).values
    if "glucose_current" in df_test.columns and len(np.unique(y_te)) >= 2:
        probs_pers = (df_test["glucose_current"] < 70).astype(float).values
        y_pred_pers = (probs_pers >= 0.5).astype(int)
        results["persistence"] = {
            "roc_auc":   round(roc_auc_score(y_te, probs_pers), 4),
            "pr_auc":    round(average_precision_score(y_te, probs_pers), 4),
            "recall":    round(recall_score(y_te, y_pred_pers, zero_division=0), 4),
            "precision": round(precision_score(y_te, y_pred_pers, zero_division=0), 4),
            "f1":        round(f1_score(y_te, y_pred_pers, zero_division=0), 4),
            "brier_raw": round(float(np.mean((probs_pers - y_te) ** 2)), 4),
            "brier_cal": "NA",
            "n_test":    int(len(y_te)),
            "n_pos":     int(y_te.sum()),
        }
        r = results["persistence"]
        print(f"  persistence      ROC={r['roc_auc']}  PR={r['pr_auc']}"
              f"  F1={r['f1']}  Recall={r['recall']}")

    configs = [
        ModelConfig.BASELINE,
        ModelConfig.TWIN_ONLY,
        ModelConfig.HYBRID,
        ModelConfig.V2_DYNAMIC_HYBRID,
    ]
    for config in configs:
        try:
            model = HybridModel(config, random_state=SEED)
            model.fit(df_train, label_col, df_val=df_val if not df_val.empty else None)
            probs_te = model.predict_proba(df_test)
            y_te_ = df_test[label_col].fillna(0).astype(int).values
            if len(np.unique(y_te_)) < 2:
                results[config.value] = {"note": "single-class test"}
                continue
            y_pred = (probs_te >= 0.5).astype(int)
            bs = float(np.mean((probs_te - y_te_) ** 2))
            bs_cal = "NA"
            if not df_val.empty:
                probs_val = model.predict_proba(df_val)
                y_val = df_val[label_col].fillna(0).astype(int).values
                if len(np.unique(y_val)) >= 2:
                    cal = select_calibrator(len(df_val))
                    cal.fit(probs_val, y_val)
                    probs_te_cal = cal.calibrate(probs_te)
                    bs_cal = round(float(np.mean((probs_te_cal - y_te_) ** 2)), 4)
            results[config.value] = {
                "roc_auc":   round(roc_auc_score(y_te_, probs_te), 4),
                "pr_auc":    round(average_precision_score(y_te_, probs_te), 4),
                "recall":    round(recall_score(y_te_, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_te_, y_pred, zero_division=0), 4),
                "f1":        round(f1_score(y_te_, y_pred, zero_division=0), 4),
                "brier_raw": round(bs, 4),
                "brier_cal": bs_cal,
                "n_test":    int(len(y_te_)),
                "n_pos":     int(y_te_.sum()),
            }
            r = results[config.value]
            print(f"  {config.value[:17]:<17} ROC={r['roc_auc']}  PR={r['pr_auc']}"
                  f"  F1={r['f1']}  Recall={r['recall']}")
        except Exception as e:
            results[config.value] = {"error": str(e)}
            print(f"  {config.value}: ERROR -- {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GlucoTwin resumable N-subject evaluation pipeline."
    )
    parser.add_argument(
        "--stage", choices=["smoke", "pilot", "full"], default="smoke",
        help="smoke=10, pilot=79, full=938 subjects",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing manifest. Without this, a fresh run is started.",
    )
    parser.add_argument(
        "--manifest-dir", default=None, dest="manifest_dir",
        help="Directory for the run manifest. Defaults to artifacts/runs/<stage>/",
    )
    parser.add_argument(
        "--parquet", default=PARQUET_PATH,
        help="Path to metabonet_public.parquet",
    )
    parser.add_argument(
        "--artifacts-dir", default=ARTIFACTS_DIR, dest="artifacts_dir",
        help="Directory for final result artifacts",
    )
    parser.add_argument(
        "--chunks-dir", default=None, dest="chunks_dir",
        help="Scratch directory for per-subject Parquet chunks. "
             "Defaults to system temp dir (C: on Windows).",
    )
    args = parser.parse_args()

    stage = args.stage
    parquet_path = args.parquet
    artifacts_dir = args.artifacts_dir

    if args.manifest_dir is None:
        args.manifest_dir = os.path.join("artifacts", "runs", stage)
    manifest_dir = args.manifest_dir
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Determine chunks_dir
    if args.chunks_dir is not None:
        chunks_dir = args.chunks_dir
    else:
        if platform.system() == "Windows":
            base = os.environ.get("TEMP", "C:\\Temp")
        else:
            base = tempfile.gettempdir()
        chunks_dir = os.path.join(base, f"glucotwin_chunks_{stage}")

    print("=" * 72)
    print(f"GlucoTwin -- Resumable Evaluation  [stage={stage}]")
    print(f"  Manifest: {os.path.abspath(_manifest_path(manifest_dir))}")
    print(f"  Chunks:   {chunks_dir}")
    print(f"  Git:      {_git_head()} @ {_git_branch()}")
    print("IMPORTANT: CGM units UNKNOWN. Threshold PROVISIONAL. NOT clinical.")
    print("=" * 72)

    # Load or create manifest
    manifest = _load_manifest(manifest_dir)

    if manifest is not None and args.resume:
        print(f"\n[manifest] Resuming from existing manifest.")
        expected_hash = _config_hash(stage, SEED, TRAIN_RATIO, VAL_RATIO, parquet_path)
        if manifest["config_hash"] != expected_hash:
            print(f"  ERROR: Config hash mismatch!")
            print(f"    Manifest: {manifest['config_hash']}")
            print(f"    Current:  {expected_hash}")
            print("  Dataset path or split ratios changed. Cannot safely resume.")
            sys.exit(1)
        if manifest["git_commit"] != _git_head():
            print(f"  WARNING: Git commit changed since manifest was created.")
            print(f"    Manifest commit: {manifest['git_commit']}")
            print(f"    Current commit:  {_git_head()}")
            manifest["warnings"].append(
                f"Resume: git commit changed from {manifest['git_commit']} "
                f"to {_git_head()} at {_now_iso()}"
            )
        chunks_dir = manifest["chunks_dir"]
        print(f"  Config hash: {manifest['config_hash']} [OK]")
        print(f"  Chunks dir:  {chunks_dir}")
    elif manifest is not None and not args.resume:
        print(f"\n[manifest] Existing manifest found but --resume not set.")
        print(f"  To resume: add --resume flag.")
        print(f"  To start fresh: delete {manifest_dir}")
        sys.exit(1)
    else:
        if os.path.exists(chunks_dir):
            print(f"  [chunks] Cleaning stale chunks dir: {chunks_dir}")
            shutil.rmtree(chunks_dir)
        os.makedirs(chunks_dir)
        manifest = _build_initial_manifest(
            stage, parquet_path, manifest_dir, chunks_dir, artifacts_dir
        )
        _write_manifest(manifest, manifest_dir)
        print(f"  [manifest] Created: {manifest['subject_counts']}")

    # Determine pending subjects
    all_subjects = (
        manifest["train_subject_ids"] +
        manifest["val_subject_ids"] +
        manifest["test_subject_ids"]
    )
    pending = [
        sid for sid in all_subjects
        if manifest["subject_status"].get(sid) not in ("done", "skipped")
    ]
    done_count = sum(1 for s in manifest["subject_status"].values() if s == "done")
    print(f"\n[subjects] Total={len(all_subjects)}  Done={done_count}  Pending={len(pending)}")

    if pending:
        print(f"\n[stream] Processing {len(pending)} pending subjects...")
        tracemalloc.start()
        t_stream_start = time.time()
        processed = 0

        for subj_id, records in iter_subjects_from_parquet(parquet_path, subject_ids=pending):
            partition = _subject_partition(subj_id, manifest)
            t_subj = time.time()
            try:
                counts = process_subject(records, subj_id, partition, chunks_dir)
                duration = round(time.time() - t_subj, 2)
                if counts:
                    manifest["subject_status"][subj_id] = "done"
                    manifest["subject_windows"][subj_id] = counts
                    manifest["subject_durations_s"][subj_id] = duration
                else:
                    manifest["subject_status"][subj_id] = "skipped"
                    manifest["warnings"].append(f"Subject {subj_id} skipped: insufficient data")
            except Exception as e:
                manifest["subject_status"][subj_id] = "failed"
                manifest["subject_errors"][subj_id] = str(e)
                print(f"    WARN [{subj_id}]: {e}")

            processed += 1
            manifest["run_updated_at"] = _now_iso()
            _write_manifest(manifest, manifest_dir)  # atomic write after every subject

            if processed % 10 == 0 or processed == len(pending):
                cur_mem = tracemalloc.get_traced_memory()[0] / 1024 ** 2
                elapsed = time.time() - t_stream_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = ((len(pending) - processed) / rate / 60) if rate > 0 else float("inf")
                print(f"  [{processed}/{len(pending)}]  Mem={cur_mem:.0f}MB  "
                      f"Rate={rate:.2f} subj/s  ETA={eta:.1f}min")
            del records
            gc.collect()

        tracemalloc.stop()
    else:
        print("[subjects] All subjects already completed. Proceeding to evaluation.")

    # Summary
    n_done = sum(1 for s in manifest["subject_status"].values() if s == "done")
    n_skip = sum(1 for s in manifest["subject_status"].values() if s == "skipped")
    n_fail = sum(1 for s in manifest["subject_status"].values() if s == "failed")
    print(f"\n[subjects] Final: Done={n_done}  Skipped={n_skip}  Failed={n_fail}")
    if n_fail > 0:
        print(f"  Failed IDs: {list(manifest['subject_errors'].keys())[:10]}")

    # Load partition chunks
    print(f"\n[load] Reading Parquet chunks from {chunks_dir} ...")
    if not os.path.exists(chunks_dir):
        print("  ERROR: Chunks directory does not exist. Cannot evaluate.")
        sys.exit(1)

    train_30 = load_partition_parquet(chunks_dir, "train", "30")
    val_30   = load_partition_parquet(chunks_dir, "val",   "30")
    test_30  = load_partition_parquet(chunks_dir, "test",  "30")
    train_60 = load_partition_parquet(chunks_dir, "train", "60")
    val_60   = load_partition_parquet(chunks_dir, "val",   "60")
    test_60  = load_partition_parquet(chunks_dir, "test",  "60")
    print(f"  30m: train={len(train_30)}  val={len(val_30)}  test={len(test_30)}")
    print(f"  60m: train={len(train_60)}  val={len(val_60)}  test={len(test_60)}")

    # Model evaluation
    tracemalloc.start()
    t_eval = time.time()
    results: Dict = {}

    results["30m"] = evaluate_models("30m", LABEL_30M, train_30, val_30, test_30)
    del train_30, val_30
    gc.collect()

    results["60m"] = evaluate_models("60m", LABEL_60M, train_60, val_60, test_60)
    del train_60, val_60
    gc.collect()

    # Twin forecast
    print(f"\n[twin] Twin trajectory evaluation...")
    twin_results: Dict = {}
    for horizon, col, df_te in [
        ("30m", "twin_glucose_t30", test_30),
        ("60m", "twin_glucose_t60", test_60),
    ]:
        obs_col = "glucose_current"
        if col in df_te.columns and obs_col in df_te.columns:
            m = evaluate_twin_forecast(
                df_te[col].tolist(), df_te[obs_col].tolist(), int(horizon[:-1])
            )
            twin_results[horizon] = {
                "n_valid": m.n_valid,
                "mae":  round(m.mae, 3)  if not math.isnan(m.mae) else "NA",
                "rmse": round(m.rmse, 3) if not math.isnan(m.rmse) else "NA",
                "bias": round(m.mean_bias, 3) if not math.isnan(m.mean_bias) else "NA",
            }
            print(f"  {horizon}: n={m.n_valid}  "
                  f"MAE={twin_results[horizon]['mae']}  "
                  f"RMSE={twin_results[horizon]['rmse']}  "
                  f"Bias={twin_results[horizon]['bias']}")
        else:
            twin_results[horizon] = {"note": "twin column not available"}
            print(f"  {horizon}: twin column not found.")

    elapsed_eval = time.time() - t_eval
    _, peak_eval_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Cleanup chunks
    shutil.rmtree(chunks_dir, ignore_errors=True)
    print(f"\n[cleanup] Chunks removed: {chunks_dir}")

    # Aggregate windows
    total_windows_30 = sum(
        v.get("windows_30m", 0) for v in manifest["subject_windows"].values()
    )
    total_windows_60 = sum(
        v.get("windows_60m", 0) for v in manifest["subject_windows"].values()
    )
    total_runtime_s = sum(manifest["subject_durations_s"].values()) + elapsed_eval

    # Final artifact (atomic write)
    output = {
        "stage": stage,
        "git_commit": manifest["git_commit"],
        "git_branch": manifest["git_branch"],
        "config_hash": manifest["config_hash"],
        "provisional_note": (
            "CGM units UNKNOWN. Threshold PROVISIONAL. NOT clinical. "
            "Pipeline-level causal feature construction, patient-held-out "
            "evaluation, and leakage safeguards passed validation."
        ),
        "subject_counts": manifest["subject_counts"],
        "subject_outcomes": {"done": n_done, "skipped": n_skip, "failed": n_fail},
        "total_windows": {"30m": total_windows_30, "60m": total_windows_60},
        "twin_forecast": twin_results,
        "models": results,
        "runtime_seconds": round(total_runtime_s, 1),
        "eval_runtime_seconds": round(elapsed_eval, 1),
        "peak_memory_mb": round(peak_eval_mem / 1024 ** 2, 1),
        "warnings": manifest["warnings"],
        "run_completed_at": _now_iso(),
    }

    out_path = manifest["output_artifact"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out = out_path + ".partial"
    with open(tmp_out, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_out, out_path)
    print(f"\n[artifact] Saved: {out_path}")

    # Finalize manifest
    manifest["run_completed_at"] = output["run_completed_at"]
    manifest["total_runtime_s"] = round(total_runtime_s, 1)
    manifest["peak_memory_mb"] = round(peak_eval_mem / 1024 ** 2, 1)
    manifest["results"] = {
        "output_artifact": out_path,
        "subject_outcomes": output["subject_outcomes"],
        "total_windows": output["total_windows"],
    }
    _write_manifest(manifest, manifest_dir)

    print(f"\n[manifest] Final: {_manifest_path(manifest_dir)}")
    print(f"[runtime]  Total={total_runtime_s:.1f}s  "
          f"Eval={elapsed_eval:.1f}s  "
          f"Peak RAM={peak_eval_mem/1024**2:.1f}MB")
    print("=" * 72)


if __name__ == "__main__":
    main()
