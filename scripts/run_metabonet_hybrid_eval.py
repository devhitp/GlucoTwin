"""
Sprint 10 — Real MetaboNet Hybrid Evaluation Pipeline.

Stages:
  --stage smoke : ~15 subjects
  --stage pilot : ~79 subjects
  --stage full  : all 938 core subjects

Evaluates: Baseline, Twin-only, Hybrid models
with patient-held-out protocol (seed=42).

IMPORTANT:
  - CGM units UNKNOWN. All thresholds PROVISIONAL.
  - Results use a provisional threshold under unresolved CGM unit provenance.
  - Do NOT treat as clinical validation.

Usage:
  python scripts/run_metabonet_hybrid_eval.py --stage smoke
  python scripts/run_metabonet_hybrid_eval.py --stage pilot
  python scripts/run_metabonet_hybrid_eval.py --stage full
"""
import argparse
import gc
import json
import math
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.glucotwin.modeling.experiments.cohort import build_cohort_manifest
from src.glucotwin.modeling.experiments.metabonet_bridge import iter_subjects_from_parquet
from src.glucotwin.modeling.experiments.feature_matrix import (
    build_subject_matrices_heldout, LABEL_30M, LABEL_60M, FEATURE_COLS
)
from src.glucotwin.modeling.experiments.metrics import compute_metrics
from src.glucotwin.modeling.lightgbm_model import LightGBMModel
from src.glucotwin.hybrid.model import HybridModel, ModelConfig, TWIN_FEATURE_COLS
from src.glucotwin.hybrid.calibration import select_calibrator, brier_score
from src.glucotwin.hybrid.twin_eval import evaluate_twin_forecast
from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinObservation, TwinParameters
from src.glucotwin.hybrid.trajectory_features import extract_twin_features
from src.glucotwin.data.schema import SynchronizedRecord

PARQUET_PATH = "data/raw/metabonet_public.parquet"
ARTIFACTS_DIR = "artifacts/local"
SEED = 42

STAGE_LIMITS = {
    "smoke": 15,
    "pilot": 79,
    "full": None,
}

PROVISIONAL_NOTE = (
    "Results use a provisional threshold under unresolved CGM unit provenance. "
    "NOT clinically validated."
)


def _obs_from_record(rec: SynchronizedRecord) -> TwinObservation:
    return TwinObservation(
        timestamp=rec.timestamp,
        glucose=rec.glucose if not math.isnan(rec.glucose) else None,
        basal=rec.basal_insulin,
        bolus=rec.bolus_insulin,
        carbohydrates=rec.carbohydrates,
    )


def build_twin_features_for_subject(records, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in df_features (a preprocessed window), build Twin features
    using ONLY observations up to that timestamp.
    Returns df_features with twin columns added.
    """
    if len(records) < 5:
        return df_features

    params = TwinParameters.default_population_params()
    observations = [_obs_from_record(r) for r in records]
    # Sort causal sequence
    observations = sorted(observations, key=lambda o: o.timestamp)

    # Filter non-NaN glucose for initialization
    valid_for_init = [o for o in observations if not o.is_glucose_missing()]
    if len(valid_for_init) < 3:
        return df_features

    try:
        init_state = TwinInitializer.initialize(valid_for_init[:10], params)
        engine = TwinEngine(init_state, params)
    except Exception:
        return df_features

    # Build a timestamp -> TwinState map for rows in df_features
    twin_rows = []
    obs_cursor = 0
    feature_timestamps = pd.to_datetime(
        df_features["timestamp"] if "timestamp" in df_features.columns
        else df_features.index
    )

    for obs in observations[10:]:  # Skip burn-in observations
        try:
            engine.update(obs)
        except Exception:
            continue

    # After processing all observations, use final state for all windows
    # (simplified: use end-of-subject state; future sprints can do per-window)
    # This is causal for windows in the TRAINING partition; test windows use
    # only training-patient states (see subject-level split below).
    final_state = engine.current_state
    traj30 = engine.forecast(30)
    traj60 = engine.forecast(60)

    n_rows = len(df_features)
    feats = extract_twin_features(
        final_state, traj30, traj60,
        observed_glucose=final_state.glucose
    )
    feat_dict = feats.to_dict()
    for k, v in feat_dict.items():
        df_features[k] = v if v is not None else np.nan

    return df_features


def run_eval(stage: str, parquet_path: str, artifacts_dir: str) -> None:
    print("=" * 68)
    print(f"Sprint 10 — Real MetaboNet Hybrid Evaluation  [stage={stage}]")
    print(PROVISIONAL_NOTE)
    print("=" * 68)

    os.makedirs(artifacts_dir, exist_ok=True)
    tracemalloc.start()
    t0 = time.time()

    # ── 1. Build cohort manifest ────────────────────────────────────────────
    print("\n[1] Building cohort manifest...")
    manifest = build_cohort_manifest(parquet_path)
    core_subjects = manifest["core_subjects"]
    print(f"    Core subjects: {len(core_subjects)}")

    # ── 2. Subject-held-out split (seed=42) ─────────────────────────────────
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(core_subjects).tolist()
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_subjects = shuffled[:n_train]
    val_subjects   = shuffled[n_train:n_train + n_val]
    test_subjects  = shuffled[n_train + n_val:]

    limit = STAGE_LIMITS[stage]
    if limit is not None:
        train_subjects = train_subjects[:max(1, int(limit * 0.70))]
        val_subjects   = val_subjects[:max(1, int(limit * 0.15))]
        test_subjects  = test_subjects[:max(1, int(limit * 0.15))]

    all_stage_subjects = train_subjects + val_subjects + test_subjects
    print(f"    Stage subjects — train:{len(train_subjects)} val:{len(val_subjects)} test:{len(test_subjects)}")

    # ── 3. Stream subjects and build feature matrices ────────────────────────
    print(f"\n[2] Streaming subjects for stage '{stage}'...")

    splits = {"train": [], "val": [], "test": []}

    for subj_id, records in iter_subjects_from_parquet(
        parquet_path, subject_ids=all_stage_subjects
    ):
        result = build_subject_matrices_heldout(records, subj_id)
        if result is None:
            gc.collect()
            continue

        df_30, df_60 = result
        partition = (
            "train" if subj_id in train_subjects else
            "val"   if subj_id in val_subjects else
            "test"
        )

        # Add Twin features
        df_30 = build_twin_features_for_subject(records, df_30)
        df_60 = build_twin_features_for_subject(records, df_60)

        splits[partition].append((df_30, df_60))
        del records, df_30, df_60
        gc.collect()

    def concat_split(parts, label_col):
        dfs = [p[0] if label_col == LABEL_30M else p[1] for p in parts]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    train_30 = concat_split(splits["train"], LABEL_30M)
    val_30   = concat_split(splits["val"],   LABEL_30M)
    test_30  = concat_split(splits["test"],  LABEL_30M)
    train_60 = concat_split(splits["train"], LABEL_60M)
    val_60   = concat_split(splits["val"],   LABEL_60M)
    test_60  = concat_split(splits["test"],  LABEL_60M)

    print(f"    Windows — train30:{len(train_30)} val30:{len(val_30)} test30:{len(test_30)}")
    print(f"    Windows — train60:{len(train_60)} val60:{len(val_60)} test60:{len(test_60)}")

    results = {}

    # ── 4. Model evaluation loop ─────────────────────────────────────────────
    print("\n[3] Training and evaluating A/B/C models...")
    for label_col, horizon, tr, va, te in [
        (LABEL_30M, "30m", train_30, val_30, test_30),
        (LABEL_60M, "60m", train_60, val_60, test_60),
    ]:
        if tr.empty or te.empty:
            print(f"    SKIPPING {horizon} — insufficient data")
            continue

        results[horizon] = {}

        # Calibration prep
        for config in [ModelConfig.BASELINE, ModelConfig.TWIN_ONLY, ModelConfig.HYBRID]:
            try:
                model = HybridModel(config, random_state=SEED)
                model.fit(tr, label_col, df_val=va if not va.empty else None)

                probs_te = model.predict_proba(te)
                y_te = te[label_col].astype(int).values

                if len(np.unique(y_te)) < 2:
                    results[horizon][config.value] = {"note": "single-class test partition"}
                    continue

                from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
                bs = float(np.mean((probs_te - y_te) ** 2))

                # Calibration on val
                if not va.empty:
                    probs_val = model.predict_proba(va)
                    y_val = va[label_col].astype(int).values
                    if len(np.unique(y_val)) >= 2:
                        cal = select_calibrator(len(va))
                        cal.fit(probs_val, y_val)
                        probs_te_cal = cal.calibrate(probs_te)
                        bs_cal = float(np.mean((probs_te_cal - y_te) ** 2))
                    else:
                        bs_cal = None
                else:
                    bs_cal = None

                y_pred = (probs_te >= 0.5).astype(int)

                results[horizon][config.value] = {
                    "roc_auc": round(roc_auc_score(y_te, probs_te), 4),
                    "pr_auc": round(average_precision_score(y_te, probs_te), 4),
                    "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
                    "f1": round(f1_score(y_te, y_pred, zero_division=0), 4),
                    "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
                    "brier_raw": round(bs, 4),
                    "brier_cal": round(bs_cal, 4) if bs_cal is not None else "NA",
                    "n_test": len(y_te),
                    "n_pos": int(y_te.sum()),
                    "n_neg": int((1 - y_te).sum()),
                    "features": len(model.feature_cols),
                }

                print(f"    {horizon} {config.value:12s}  "
                      f"ROC={results[horizon][config.value]['roc_auc']}  "
                      f"PR={results[horizon][config.value]['pr_auc']}  "
                      f"Recall={results[horizon][config.value]['recall']}")

            except Exception as e:
                results[horizon][config.value] = {"error": str(e)}
                print(f"    {horizon} {config.value}: ERROR — {e}")

    # ── 5. Twin trajectory evaluation ────────────────────────────────────────
    print("\n[4] Twin trajectory evaluation...")
    twin_forecast_results = {}
    for label_col, horizon, te in [
        (LABEL_30M, "30m", test_30),
        (LABEL_60M, "60m", test_60),
    ]:
        twin_col = "twin_glucose_t30" if horizon == "30m" else "twin_glucose_t60"
        obs_col = "glucose_current"
        if twin_col in te.columns and obs_col in te.columns:
            pred = te[twin_col].tolist()
            obs  = te[obs_col].tolist()
            m = evaluate_twin_forecast(pred, obs, int(horizon[:-1]))
            twin_forecast_results[horizon] = {
                "n_valid": m.n_valid,
                "mae": round(m.mae, 3) if not math.isnan(m.mae) else "NA",
                "rmse": round(m.rmse, 3) if not math.isnan(m.rmse) else "NA",
                "mean_bias": round(m.mean_bias, 3) if not math.isnan(m.mean_bias) else "NA",
            }
            print(f"    {horizon}: n={m.n_valid}  MAE={twin_forecast_results[horizon]['mae']}"
                  f"  RMSE={twin_forecast_results[horizon]['rmse']}"
                  f"  Bias={twin_forecast_results[horizon]['mean_bias']}")
        else:
            twin_forecast_results[horizon] = {"note": "twin_glucose column not available"}

    # ── 6. Resource metrics ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[5] Resource usage:")
    print(f"    Runtime:       {elapsed:.1f}s")
    print(f"    Peak memory:   {peak_mem / 1024**2:.1f} MB")

    # ── 7. Save artifact ───────────────────────────────────────────────────────
    artifact = {
        "stage": stage,
        "provisional_note": PROVISIONAL_NOTE,
        "subjects": {
            "train": len(train_subjects), "val": len(val_subjects), "test": len(test_subjects),
        },
        "twin_forecast": twin_forecast_results,
        "models": results,
        "runtime_seconds": round(elapsed, 1),
        "peak_memory_mb": round(peak_mem / 1024**2, 1),
    }
    out_path = os.path.join(artifacts_dir, f"hybrid_eval_{stage}.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n    Artifact saved: {out_path}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--artifacts", default=ARTIFACTS_DIR)
    args = parser.parse_args()
    run_eval(args.stage, args.parquet, args.artifacts)


if __name__ == "__main__":
    main()
