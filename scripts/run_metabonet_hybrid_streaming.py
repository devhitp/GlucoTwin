"""
Sprint 11 — Real MetaboNet Hybrid Evaluation (Streaming + Parquet chunks).

Memory-safe processing:
- Streams subjects in batches via PyArrow dataset filter.
- Writes intermediate features to Parquet (compressed) on C: drive temp dir.
- Evaluates LightGBM models after all subjects are processed.
- Avoids massive Pandas concatenation and D: drive exhaustion.
"""
import argparse
import gc
import json
import math
import os
import sys
import time
import tempfile
import tracemalloc
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest
from src.glucotwin.modeling.experiments.metabonet_bridge import iter_subjects_from_parquet
from src.glucotwin.modeling.experiments.feature_matrix import (
    build_subject_matrices_heldout, LABEL_30M, LABEL_60M, FEATURE_COLS
)
from src.glucotwin.hybrid.causal_feature_extractor import extract_causal_twin_features_fast
from src.glucotwin.hybrid.model import HybridModel, ModelConfig, BASELINE_FEATURE_COLS, TWIN_FEATURE_COLS, TWIN_V2_FEATURE_COLS
from src.glucotwin.hybrid.calibration import select_calibrator
from src.glucotwin.hybrid.twin_eval import evaluate_twin_forecast
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

PARQUET_PATH = "data/raw/metabonet_public.parquet"
ARTIFACTS_DIR = "artifacts/local"
SEED = 42

STAGE_LIMITS = {
    "smoke": 15,
    "pilot": 79,
    "full": 938,
}

# Minimal columns to persist — only those required for LightGBM fit and twin evaluation
SAVE_COLS_30 = sorted(set(
    [c for c in FEATURE_COLS] +
    TWIN_FEATURE_COLS + TWIN_V2_FEATURE_COLS +
    [LABEL_30M, "patient_id", "glucose_current", "twin_glucose_t30"]
))
SAVE_COLS_60 = sorted(set(
    [c for c in FEATURE_COLS] +
    TWIN_FEATURE_COLS + TWIN_V2_FEATURE_COLS +
    [LABEL_60M, "patient_id", "glucose_current", "twin_glucose_t60"]
))


def merge_twin_features(df_baseline: pd.DataFrame, df_twin: pd.DataFrame) -> pd.DataFrame:
    if df_twin.empty:
        return df_baseline
        
    # Check if we can fast-path using direct numpy assignment
    if len(df_baseline) == len(df_twin) and "timestamp" in df_twin.columns:
        b_ts = df_baseline.index if df_baseline.index.name == "timestamp" or np.issubdtype(df_baseline.index.dtype, np.datetime64) else df_baseline.get("timestamp")
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
        merged = pd.merge(df_b, df_twin, on="timestamp", how="left")
    else:
        merged = pd.concat([df_b.reset_index(drop=True), df_twin.reset_index(drop=True)], axis=1)
    return merged


def _align_and_trim(df: pd.DataFrame, save_cols: list) -> pd.DataFrame:
    """Add missing save_cols as NaN and return only save_cols."""
    for c in save_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[save_cols]


def process_subject(records, subj_id: str, partition: str, chunks_dir: str):
    """Process one subject, write a per-subject Parquet file to chunks_dir."""
    result = build_subject_matrices_heldout(records, subj_id)
    if result is None:
        return

    df_30, df_60 = result

    target_timestamps = set()
    if not df_30.empty:
        target_timestamps.update(df_30.index)
    if not df_60.empty:
        target_timestamps.update(df_60.index)

    df_twin = extract_causal_twin_features_fast(records, target_timestamps)

    df_30_full = _align_and_trim(merge_twin_features(df_30, df_twin), SAVE_COLS_30)
    df_60_full = _align_and_trim(merge_twin_features(df_60, df_twin), SAVE_COLS_60)

    safe_id = subj_id.replace("/", "_").replace("\\", "_")
    df_30_full.to_parquet(os.path.join(chunks_dir, f"{safe_id}_{partition}_30.parquet"), index=False)
    df_60_full.to_parquet(os.path.join(chunks_dir, f"{safe_id}_{partition}_60.parquet"), index=False)

    del df_30, df_60, df_twin, df_30_full, df_60_full


def load_partition_parquet(chunks_dir: str, partition: str, horizon: str) -> pd.DataFrame:
    """Load all per-subject Parquet files, downcast to float32 to save memory."""
    pattern = f"_{partition}_{horizon}.parquet"
    files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(pattern)]
    if not files:
        return pd.DataFrame()
    
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])
        
        # Downcast floats to float32
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)
            
        parts.append(df)
        
    df_full = pd.concat(parts, ignore_index=True)
    del parts
    return df_full


def evaluate_models(horizon: str, label_col: str, df_train: pd.DataFrame,
                    df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    print(f"\nEvaluating {horizon} horizon...")
    if df_train.empty or df_test.empty:
        print(f"  Skipping {horizon} - insufficient data.")
        return {}

    results = {}

    # 1. Persistence Baseline
    y_te_pers = df_test[label_col].fillna(0).astype(int).values
    if "glucose_current" in df_test.columns and len(np.unique(y_te_pers)) >= 2:
        # Persistence predicts positive if current glucose is < 70
        probs_pers = (df_test["glucose_current"] < 70).astype(float).values
        y_pred_pers = (probs_pers >= 0.5).astype(int)
        bs_pers = float(np.mean((probs_pers - y_te_pers) ** 2))
        
        results["persistence"] = {
            "roc_auc": round(roc_auc_score(y_te_pers, probs_pers), 4),
            "pr_auc": round(average_precision_score(y_te_pers, probs_pers), 4),
            "recall": round(recall_score(y_te_pers, y_pred_pers, zero_division=0), 4),
            "precision": round(precision_score(y_te_pers, y_pred_pers, zero_division=0), 4),
            "f1": round(f1_score(y_te_pers, y_pred_pers, zero_division=0), 4),
            "brier_raw": round(bs_pers, 4),
            "brier_cal": "NA",
            "n_test": int(len(y_te_pers)),
            "n_pos": int(y_te_pers.sum()),
        }
        r = results["persistence"]
        print(f"  persistence  ROC={r['roc_auc']}  PR={r['pr_auc']}  "
              f"F1={r['f1']}  Recall={r['recall']}")

    configs = [
        ModelConfig.BASELINE, 
        ModelConfig.TWIN_ONLY, 
        ModelConfig.HYBRID,
        ModelConfig.V2_DYNAMIC_HYBRID
    ]
    for config in configs:
        try:
            model = HybridModel(config, random_state=SEED)
            model.fit(df_train, label_col, df_val=df_val if not df_val.empty else None)

            probs_te = model.predict_proba(df_test)
            y_te = df_test[label_col].fillna(0).astype(int).values

            if len(np.unique(y_te)) < 2:
                results[config.value] = {"note": "single-class test"}
                continue

            y_pred = (probs_te >= 0.5).astype(int)
            bs = float(np.mean((probs_te - y_te) ** 2))
            bs_cal = "NA"

            if not df_val.empty:
                probs_val = model.predict_proba(df_val)
                y_val = df_val[label_col].fillna(0).astype(int).values
                if len(np.unique(y_val)) >= 2:
                    cal = select_calibrator(len(df_val))
                    cal.fit(probs_val, y_val)
                    probs_te_cal = cal.calibrate(probs_te)
                    bs_cal = round(float(np.mean((probs_te_cal - y_te) ** 2)), 4)

            results[config.value] = {
                "roc_auc": round(roc_auc_score(y_te, probs_te), 4),
                "pr_auc": round(average_precision_score(y_te, probs_te), 4),
                "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_te, y_pred, zero_division=0), 4),
                "brier_raw": round(bs, 4),
                "brier_cal": bs_cal,
                "n_test": int(len(y_te)),
                "n_pos": int(y_te.sum()),
            }
            r = results[config.value]
            # Ensure proper formatting for alignment
            conf_str = config.value[:17].ljust(17)
            print(f"  {conf_str} ROC={r['roc_auc']}  PR={r['pr_auc']}  "
                  f"F1={r['f1']}  Recall={r['recall']}")

        except Exception as e:
            results[config.value] = {"error": str(e)}
            print(f"  {config.value}: ERROR - {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    args = parser.parse_args()
    stage = args.stage

    print("=" * 68)
    print(f"Sprint 11 — Streaming Evaluation  [stage={stage}]")
    print("IMPORTANT: CGM units UNKNOWN. Threshold PROVISIONAL. NOT clinical.")
    print("=" * 68)

    tracemalloc.start()
    t0 = time.time()

    # Use system temp dir (C: drive) to avoid D: exhaustion
    chunks_dir = os.path.join(tempfile.gettempdir(), f"glucotwin_chunks_{stage}")
    if os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir)
    print(f"  Chunk dir: {chunks_dir}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Subject split
    manifest = build_cohort_manifest(PARQUET_PATH)
    core_subjects = manifest["core_subjects"]
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(core_subjects).tolist()

    limit = STAGE_LIMITS[stage]
    if limit is not None:
        shuffled = shuffled[:limit]

    n = len(shuffled)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    train_subj = set(shuffled[:n_train])
    val_subj = set(shuffled[n_train:n_train + n_val])
    test_subj = set(shuffled[n_train + n_val:])

    print(f"\n[1] Subjects: total={n}  train={len(train_subj)}  "
          f"val={len(val_subj)}  test={len(test_subj)}")

    # Stream & process
    print(f"\n[2] Streaming subjects...")
    processed = 0
    for subj_id, records in iter_subjects_from_parquet(PARQUET_PATH, subject_ids=shuffled):
        partition = ("train" if subj_id in train_subj else
                     "val" if subj_id in val_subj else "test")
        try:
            process_subject(records, subj_id, partition, chunks_dir)
        except Exception as e:
            print(f"    WARN: subject {subj_id} failed: {e}")
        processed += 1
        if processed % 10 == 0:
            cur_mem = tracemalloc.get_traced_memory()[0] / 1024**2
            print(f"  Processed {processed}/{n}. Mem: {cur_mem:.1f}MB")
        del records
        gc.collect()

    # Load partitions from Parquet chunks
    print(f"\n[3] Loading partitions...")
    train_30 = load_partition_parquet(chunks_dir, "train", "30")
    val_30 = load_partition_parquet(chunks_dir, "val", "30")
    test_30 = load_partition_parquet(chunks_dir, "test", "30")
    train_60 = load_partition_parquet(chunks_dir, "train", "60")
    val_60 = load_partition_parquet(chunks_dir, "val", "60")
    test_60 = load_partition_parquet(chunks_dir, "test", "60")
    print(f"  30m: train={len(train_30)} val={len(val_30)} test={len(test_30)}")
    print(f"  60m: train={len(train_60)} val={len(val_60)} test={len(test_60)}")

    # Model evaluation
    print(f"\n[4] Training & Evaluating models...")
    results = {}
    
    # 30m models
    results["30m"] = evaluate_models("30m", LABEL_30M, train_30, val_30, test_30)
    # Clear 30m train/val to save memory
    del train_30
    del val_30
    gc.collect()

    # 60m models
    results["60m"] = evaluate_models("60m", LABEL_60M, train_60, val_60, test_60)
    # Clear 60m train/val to save memory
    del train_60
    del val_60
    gc.collect()

    # Twin forecast evaluation
    print(f"\n[5] Twin trajectory evaluation...")
    twin_results = {}
    for horizon, col, df_te in [
        ("30m", "twin_glucose_t30", test_30),
        ("60m", "twin_glucose_t60", test_60),
    ]:
        obs_col = "glucose_current"
        if col in df_te.columns and obs_col in df_te.columns:
            m = evaluate_twin_forecast(
                df_te[col].tolist(),
                df_te[obs_col].tolist(),
                int(horizon[:-1])
            )
            twin_results[horizon] = {
                "n_valid": m.n_valid,
                "mae": round(m.mae, 3) if not math.isnan(m.mae) else "NA",
                "rmse": round(m.rmse, 3) if not math.isnan(m.rmse) else "NA",
                "bias": round(m.mean_bias, 3) if not math.isnan(m.mean_bias) else "NA",
            }
            print(f"  {horizon}: n={m.n_valid}  MAE={twin_results[horizon]['mae']}  "
                  f"Bias={twin_results[horizon]['bias']}")
        else:
            twin_results[horizon] = {"note": "twin column not available"}
            print(f"  {horizon}: twin column not found in test set.")

    # Cleanup chunks
    shutil.rmtree(chunks_dir, ignore_errors=True)

    elapsed = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[6] Runtime: {elapsed:.1f}s  Peak Mem: {peak_mem / 1024**2:.1f} MB")

    out = {
        "stage": stage,
        "provisional_note": "CGM units UNKNOWN. Threshold PROVISIONAL. NOT clinical.",
        "n_subjects": n,
        "twin_forecast": twin_results,
        "models": results,
        "runtime_seconds": round(elapsed, 1),
        "peak_memory_mb": round(peak_mem / 1024**2, 1),
    }
    out_path = os.path.join(ARTIFACTS_DIR, f"sprint11_hybrid_{stage}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Artifact: {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
