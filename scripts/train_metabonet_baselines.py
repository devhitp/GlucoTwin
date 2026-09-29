"""
GlucoTwin Sprint 7 — Real MetaboNet Baseline Training Script

Stages:
  --stage smoke   : 5 deterministic subjects (engineering validation)
  --stage pilot   : 50 subjects (performance + metric validation)
  --stage full    : all core-cohort subjects

PROVISIONAL NOTE:
  CGM units assumed mg/dL (NOT confirmed from authoritative documentation).
  All threshold-based metrics are labelled PROVISIONAL.
  Insulin and carbohydrate units remain unconfirmed — they are used as
  relative features only; no dosing-specific clinical thresholds are applied.

  Do NOT treat these results as clinically validated.

Usage:
  python scripts/train_metabonet_baselines.py --stage smoke
  python scripts/train_metabonet_baselines.py --stage pilot
  python scripts/train_metabonet_baselines.py --stage full --max-subjects 200
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

# Ensure src is on path when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest, save_manifest
from src.glucotwin.modeling.experiments.metabonet_bridge import iter_subjects_from_parquet
from src.glucotwin.modeling.experiments.feature_matrix import (
    build_subject_matrices, aggregate_splits, LABEL_30M, LABEL_60M, FEATURE_COLS
)
from src.glucotwin.modeling.experiments.metrics import (
    compute_metrics, select_threshold_on_val, compute_calibration
)
from src.glucotwin.modeling.experiments.error_analysis import error_analysis
from src.glucotwin.modeling.baselines import PersistenceBaseline
from src.glucotwin.modeling.lightgbm_model import LightGBMModel

PARQUET_PATH = "data/raw/metabonet_public.parquet"
ARTIFACTS_DIR = "artifacts/local"

LGBM_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _assert_privacy(artifacts_dir: str) -> None:
    """Verify no raw sequences in JSON artifacts."""
    if not os.path.exists(artifacts_dir):
        return
    for fname in os.listdir(artifacts_dir):
        if fname.endswith(".json"):
            with open(os.path.join(artifacts_dir, fname)) as f:
                data = json.load(f)
            def _check(obj, path=""):
                if isinstance(obj, list) and len(obj) > 100:
                    # Calibration arrays are permitted
                    assert ("calibration" in path or "fraction" in path
                            or "predicted" in path or "importance" in path), \
                        f"Potential raw sequence leak at {path}: len={len(obj)}"
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _check(v, path + f"/{k}")
            _check(data)


def run_stage(stage: str, max_subjects: int = None):
    print(f"\n{'=' * 60}")
    print(f"GlucoTwin Sprint 7 — Stage: {stage.upper()}")
    print(f"PROVISIONAL: CGM threshold at 70 mg/dL NOT confirmed from docs")
    print(f"{'=' * 60}\n")

    assert os.path.exists(PARQUET_PATH), \
        f"Parquet not found at '{PARQUET_PATH}'. Run from repository root."

    # ---- COHORT SELECTION ---------------------------------------------------
    print("Building cohort manifest (DuckDB scan — memory-safe)...")
    t0 = time.time()
    manifest = build_cohort_manifest(PARQUET_PATH)
    core_subjects = manifest["core_subjects"]
    stats = manifest["stats"]

    print(f"  All subjects:        {stats['total_subjects']:,}")
    print(f"  Core cohort:         {len(core_subjects):,}  "
          f"(>={14}d recording, >=1000 CGM rows, any insulin+carbs)")
    print(f"  Median duration:     {stats['median_duration_days']:.1f} days")
    print(f"  Core median duration:{stats.get('core_median_duration_days', 'N/A')}")
    print(f"  Core median CGM rows:{stats.get('core_median_cgm_rows', 'N/A')}")

    if stage == "smoke":
        selected = sorted(core_subjects)[:5]
        print(f"  Smoke test:          {len(selected)} subjects (deterministic: first 5 sorted IDs)")
    elif stage == "pilot":
        rng = np.random.default_rng(42)
        n_pilot = min(50, len(core_subjects))
        selected = list(rng.choice(core_subjects, size=n_pilot, replace=False))
        print(f"  Pilot:               {len(selected)} subjects (seed=42, not selected on outcome)")
    else:
        selected = core_subjects
        if max_subjects is not None and max_subjects < len(selected):
            # Deterministic sub-sample to respect resource limits
            rng = np.random.default_rng(42)
            selected = list(rng.choice(selected, size=max_subjects, replace=False))
            print(f"  Full (resource-limited): {len(selected)} / {len(core_subjects)} subjects")
        else:
            print(f"  Full cohort:         {len(selected)} subjects")

    # Save cohort manifest (aggregate stats only, no raw patient data)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    manifest_path = os.path.join(ARTIFACTS_DIR, f"sprint7_{stage}_cohort_manifest.json")
    save_manifest(manifest, manifest_path)
    print(f"  Cohort manifest:     {manifest_path}")

    # ---- DATA LOADING & PREPROCESSING ---------------------------------------
    print(f"\nProcessing {len(selected)} subjects through pipeline "
          f"(memory-safe row-group iteration)...")
    tracemalloc.start()

    trains_30, vals_30, tests_30 = [], [], []
    trains_60, vals_60, tests_60 = [], [], []
    subjects_processed = 0
    subjects_skipped = 0
    total_cgm_rows = 0

    for subj_id, records in iter_subjects_from_parquet(PARQUET_PATH, subject_ids=selected):
        total_cgm_rows += len(records)
        result = build_subject_matrices(records)
        if result is None:
            subjects_skipped += 1
            continue

        train, val, test = result

        trains_30.append(train)
        vals_30.append(val)
        tests_30.append(test)

        trains_60.append(train)
        vals_60.append(val)
        tests_60.append(test)

        subjects_processed += 1
        if subjects_processed % 10 == 0:
            _cur, _peak = tracemalloc.get_traced_memory()
            print(f"  Processed {subjects_processed}/{len(selected)} | "
                  f"Peak mem so far: {_peak / 1e6:.1f} MB")

    current_mb, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    prep_time = time.time() - t0

    print(f"\n  Subjects processed:  {subjects_processed}")
    print(f"  Subjects skipped:    {subjects_skipped}  "
          f"(insufficient data after quality filter + split)")
    print(f"  Total CGM rows read: {total_cgm_rows:,}")
    print(f"  Preprocessing time:  {prep_time:.1f}s")
    print(f"  Peak memory:         {peak_mb / 1e6:.1f} MB")

    if subjects_processed == 0:
        print("\nERROR: No subjects produced usable data. Aborting.")
        sys.exit(1)

    # ---- FEATURE MATRICES ---------------------------------------------------
    print("\nAggregating feature matrices...")
    X_tr30, y_tr30, X_val30, y_val30, X_te30, y_te30 = aggregate_splits(
        trains_30, vals_30, tests_30, LABEL_30M)
    X_tr60, y_tr60, X_val60, y_val60, X_te60, y_te60 = aggregate_splits(
        trains_60, vals_60, tests_60, LABEL_60M)

    n_features_used = X_tr30.shape[1]
    print(f"  Feature columns used: {n_features_used}/{len(FEATURE_COLS)}")
    print(f"  Train-30m: {X_tr30.shape}, pos={y_tr30.mean():.4f}")
    print(f"  Val-30m:   {X_val30.shape}, pos={y_val30.mean():.4f}")
    print(f"  Test-30m:  {X_te30.shape}, pos={y_te30.mean():.4f}")
    print(f"  Train-60m: {X_tr60.shape}, pos={y_tr60.mean():.4f}")
    print(f"  Test-60m:  {X_te60.shape}, pos={y_te60.mean():.4f}")

    # ---- POSITIVE EVENT PREVALENCE REPORTING --------------------------------
    # Count positive windows (not assumed to be independent observations)
    pos_windows_30 = int(y_te30.sum())
    pos_windows_60 = int(y_te60.sum())
    print(f"\n  [PROVISIONAL mg/dL] Positive test windows 30m: {pos_windows_30:,} "
          f"/ {len(y_te30):,} ({100*y_te30.mean():.2f}%)")
    print(f"  [PROVISIONAL mg/dL] Positive test windows 60m: {pos_windows_60:,} "
          f"/ {len(y_te60):,} ({100*y_te60.mean():.2f}%)")

    results = {
        "stage": stage,
        "PROVISIONAL_NOTE": "CGM threshold 70 mg/dL assumed mg/dL — units NOT confirmed",
        "label_30m_definition": (
            "Binary: 1 if any glucose < 70 mg/dL (PROVISIONAL) in next 30 min; "
            "NaN if no future glucose; excludes inter-split boundary rows"
        ),
        "label_60m_definition": (
            "Binary: 1 if any glucose < 70 mg/dL (PROVISIONAL) in next 60 min; "
            "NaN if no future glucose; excludes inter-split boundary rows"
        ),
        "split_protocol": (
            "Within-subject chronological: 60% train / 20% val / 20% test; "
            "60-minute embargo gap at each boundary; aggregated across subjects"
        ),
        "subjects_selected": len(selected),
        "subjects_processed": subjects_processed,
        "subjects_skipped": subjects_skipped,
        "total_cgm_rows_loaded": total_cgm_rows,
        "n_features_used": n_features_used,
        "peak_memory_mb": round(peak_mb / 1e6, 1),
        "preprocessing_seconds": round(prep_time, 1),
        "train_30m_windows": int(len(X_tr30)),
        "val_30m_windows": int(len(X_val30)),
        "test_30m_windows": int(len(X_te30)),
        "train_30m_prevalence": float(y_tr30.mean()),
        "val_30m_prevalence": float(y_val30.mean()),
        "test_30m_prevalence": float(y_te30.mean()),
        "train_60m_windows": int(len(X_tr60)),
        "val_60m_windows": int(len(X_val60)),
        "test_60m_windows": int(len(X_te60)),
        "train_60m_prevalence": float(y_tr60.mean()),
        "val_60m_prevalence": float(y_val60.mean()),
        "test_60m_prevalence": float(y_te60.mean()),
    }

    # ---- PERSISTENCE BASELINE -----------------------------------------------
    print("\nEvaluating persistence baseline...")
    persistence = PersistenceBaseline(threshold=70.0)

    prob_val30 = persistence.predict_proba(X_val30)
    thr_p30 = select_threshold_on_val(y_val30.values, prob_val30)
    prob_te30 = persistence.predict_proba(X_te30)
    pred_p30 = (prob_te30 >= thr_p30).astype(int)

    prob_val60 = persistence.predict_proba(X_val60)
    thr_p60 = select_threshold_on_val(y_val60.values, prob_val60)
    prob_te60 = persistence.predict_proba(X_te60)
    pred_p60 = (prob_te60 >= thr_p60).astype(int)

    results["persistence_30m"] = compute_metrics(
        y_te30.values, prob_te30, thr_p30, "persistence_30m")
    results["persistence_60m"] = compute_metrics(
        y_te60.values, prob_te60, thr_p60, "persistence_60m")
    results["error_analysis_persistence_30m"] = error_analysis(
        X_te30, y_te30.values, pred_p30, "persistence_30m")
    results["error_analysis_persistence_60m"] = error_analysis(
        X_te60, y_te60.values, pred_p60, "persistence_60m")

    print(f"  Persistence 30m | "
          f"ROC-AUC={results['persistence_30m'].get('roc_auc', 'N/A'):.3f} | "
          f"PR-AUC={results['persistence_30m'].get('pr_auc', 'N/A'):.3f} | "
          f"thr={thr_p30:.2f}")
    print(f"  Persistence 60m | "
          f"ROC-AUC={results['persistence_60m'].get('roc_auc', 'N/A'):.3f} | "
          f"PR-AUC={results['persistence_60m'].get('pr_auc', 'N/A'):.3f} | "
          f"thr={thr_p60:.2f}")

    # ---- LIGHTGBM -----------------------------------------------------------
    print("\nTraining LightGBM (30m target)...")
    t_lgbm = time.time()
    lgbm_params = {k: v for k, v in LGBM_CONFIG.items() if k != "random_state"}

    lgbm30 = LightGBMModel(random_state=42)
    lgbm30.model.set_params(**lgbm_params)
    lgbm30.fit(X_tr30, y_tr30, X_val30, y_val30)

    prob_lgbm_val30 = lgbm30.predict_proba(X_val30)
    thr_l30 = select_threshold_on_val(y_val30.values, prob_lgbm_val30)
    prob_lgbm_te30 = lgbm30.predict_proba(X_te30)
    pred_l30 = (prob_lgbm_te30 >= thr_l30).astype(int)

    results["lgbm_30m"] = compute_metrics(
        y_te30.values, prob_lgbm_te30, thr_l30, "lgbm_30m")
    results["calibration_lgbm_30m"] = compute_calibration(y_te30.values, prob_lgbm_te30)
    results["error_analysis_lgbm_30m"] = error_analysis(
        X_te30, y_te30.values, pred_l30, "lgbm_30m")

    print(f"  LightGBM 30m | "
          f"ROC-AUC={results['lgbm_30m'].get('roc_auc', 'N/A'):.3f} | "
          f"PR-AUC={results['lgbm_30m'].get('pr_auc', 'N/A'):.3f} | "
          f"thr={thr_l30:.2f}")

    print("\nTraining LightGBM (60m target)...")
    lgbm60 = LightGBMModel(random_state=42)
    lgbm60.model.set_params(**lgbm_params)
    lgbm60.fit(X_tr60, y_tr60, X_val60, y_val60)

    prob_lgbm_val60 = lgbm60.predict_proba(X_val60)
    thr_l60 = select_threshold_on_val(y_val60.values, prob_lgbm_val60)
    prob_lgbm_te60 = lgbm60.predict_proba(X_te60)
    pred_l60 = (prob_lgbm_te60 >= thr_l60).astype(int)

    results["lgbm_60m"] = compute_metrics(
        y_te60.values, prob_lgbm_te60, thr_l60, "lgbm_60m")
    results["calibration_lgbm_60m"] = compute_calibration(y_te60.values, prob_lgbm_te60)
    results["error_analysis_lgbm_60m"] = error_analysis(
        X_te60, y_te60.values, pred_l60, "lgbm_60m")

    lgbm_time = time.time() - t_lgbm
    results["lgbm_training_seconds"] = round(lgbm_time, 1)

    print(f"  LightGBM 60m | "
          f"ROC-AUC={results['lgbm_60m'].get('roc_auc', 'N/A'):.3f} | "
          f"PR-AUC={results['lgbm_60m'].get('pr_auc', 'N/A'):.3f} | "
          f"thr={thr_l60:.2f}")
    print(f"  LightGBM training time: {lgbm_time:.1f}s")

    # ---- FEATURE IMPORTANCE -------------------------------------------------
    print("\nFeature importance (LightGBM 30m)...")
    fi_df = pd.DataFrame({
        "feature": X_tr30.columns.tolist(),
        "importance": lgbm30.model.feature_importances_,
    }).sort_values("importance", ascending=False)
    results["feature_importance_30m"] = fi_df.to_dict(orient="records")
    print("  Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.0f}")

    fi_df60 = pd.DataFrame({
        "feature": X_tr60.columns.tolist(),
        "importance": lgbm60.model.feature_importances_,
    }).sort_values("importance", ascending=False)
    results["feature_importance_60m"] = fi_df60.to_dict(orient="records")

    # ---- SAVE ARTIFACTS -----------------------------------------------------
    out_path = os.path.join(ARTIFACTS_DIR, f"sprint7_{stage}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Privacy assertion — no raw patient sequences in artifact
    _assert_privacy(ARTIFACTS_DIR)
    print("Privacy check: PASSED (no raw patient sequences in artifacts)")

    # ---- SUMMARY REPORT -----------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SPRINT 7 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Stage:                {stage}")
    print(f"Subjects:             {subjects_processed} processed / {subjects_skipped} skipped")
    print(f"Peak memory:          {peak_mb / 1e6:.1f} MB")
    print(f"Feature columns used: {n_features_used}")
    print(f"Test windows 30m:     {len(X_te30):,}")
    print(f"Test windows 60m:     {len(X_te60):,}")
    print(f"Prevalence 30m:       {y_te30.mean():.4f} [PROVISIONAL mg/dL]")
    print(f"Prevalence 60m:       {y_te60.mean():.4f} [PROVISIONAL mg/dL]")
    print()

    def _fmt(m, key):
        v = m.get(key, "N/A")
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    print(f"{'Model':<20} {'Target':<6} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'F1':>8} {'Recall':>8} {'Prec':>8} {'Brier':>8}")
    print("-" * 78)
    for target, label_key, m in [
        ("Persistence", "30m", results["persistence_30m"]),
        ("Persistence", "60m", results["persistence_60m"]),
        ("LightGBM", "30m", results["lgbm_30m"]),
        ("LightGBM", "60m", results["lgbm_60m"]),
    ]:
        print(f"  {target:<18} {label_key:<6}"
              + f" {_fmt(m, 'roc_auc'):>8} {_fmt(m, 'pr_auc'):>8}"
              + f" {_fmt(m, 'f1'):>8} {_fmt(m, 'recall'):>8}"
              + f" {_fmt(m, 'precision'):>8} {_fmt(m, 'brier_score'):>8}")

    print(f"\n[IMPORTANT] Results are {stage}-stage only.")
    print("[IMPORTANT] Within-subject chronological evaluation — NOT unseen-patient.")
    print("[IMPORTANT] CGM units PROVISIONAL (assumed mg/dL).")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GlucoTwin Sprint 7 baseline training")
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke",
                        help="Execution stage")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Resource cap: max subjects for full stage")
    args = parser.parse_args()
    run_stage(args.stage, max_subjects=args.max_subjects)
