"""
Sprint 9 synthetic ablation experiment.
Runs A/B/C configurations on synthetic data and prints a summary report.
This is NOT a clinical evaluation. All results use synthetic data only.
"""
import time
import numpy as np
import pandas as pd
import psutil
from datetime import datetime

from src.glucotwin.hybrid.model import HybridModel, ModelConfig
from src.glucotwin.hybrid.calibration import select_calibrator, brier_score
from src.glucotwin.hybrid.uncertainty import BootstrapEnsemble
from src.glucotwin.hybrid.twin_eval import evaluate_twin_forecast

SEED = 42


def make_dataset(n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    glucose = rng.uniform(60, 200, n)
    df = pd.DataFrame({
        "glucose_current": glucose,
        "glucose_roc_5m": rng.uniform(-3, 3, n),
        "glucose_roc_15m": rng.uniform(-2, 2, n),
        "glucose_roc_30m": rng.uniform(-1, 1, n),
        "glucose_mean_30m": rng.uniform(80, 180, n),
        "glucose_std_30m": rng.uniform(5, 30, n),
        "glucose_min_30m": rng.uniform(60, 120, n),
        "glucose_max_30m": rng.uniform(120, 200, n),
        "insulin_recent_sum": rng.uniform(0, 10, n),
        "carbs_recent_sum": rng.uniform(0, 80, n),
        "time_since_meal": rng.uniform(0, 300, n),
        "time_since_bolus": rng.uniform(0, 300, n),
        "patient_glucose_baseline": rng.uniform(90, 140, n),
        "hour_of_day": rng.integers(0, 24, n).astype(float),
        "steps_available": rng.choice([0.0, 1.0], n),
    })
    # Twin features with some signal (twin glucose close to actual)
    df["twin_glucose_current"] = glucose + rng.normal(0, 3, n)
    df["twin_insulin_action"] = rng.uniform(0, 2, n)
    df["twin_insulin_state"] = rng.uniform(0, 3, n)
    df["twin_meal_state"] = rng.uniform(0, 20, n)
    df["twin_context_state"] = rng.uniform(0, 1, n)
    df["twin_glucose_t5"] = glucose + rng.uniform(-5, 5, n)
    df["twin_glucose_t15"] = glucose + rng.uniform(-10, 10, n)
    df["twin_glucose_t30"] = glucose + rng.uniform(-15, 15, n)
    df["twin_glucose_t60"] = glucose + rng.uniform(-20, 20, n)
    df["twin_slope_30m"] = (df["twin_glucose_t30"] - df["twin_glucose_current"]) / 30.0
    df["twin_slope_60m"] = (df["twin_glucose_t60"] - df["twin_glucose_current"]) / 60.0
    df["twin_delta_15m"] = df["twin_glucose_t15"] - df["twin_glucose_current"]
    df["twin_delta_30m"] = df["twin_glucose_t30"] - df["twin_glucose_current"]
    df["twin_delta_60m"] = df["twin_glucose_t60"] - df["twin_glucose_current"]
    df["twin_residual_current"] = glucose - df["twin_glucose_current"]
    df["twin_has_long_gap"] = 0.0
    df["twin_has_missing_glucose"] = 0.0
    df["twin_uncertainty_calibrated"] = 0.0

    # Labels: simple rule based on glucose + twin trajectory
    risk_30 = (glucose < 80) | (df["twin_glucose_t30"] < 80)
    risk_60 = (glucose < 90) | (df["twin_glucose_t60"] < 90)
    df["future_hypoglycemia_30m"] = risk_30.astype(int)
    df["future_hypoglycemia_60m"] = risk_60.astype(int)
    return df


def run_ablation():
    t0 = time.time()
    mem0 = psutil.Process().memory_info().rss / (1024**2)

    print("=" * 64)
    print("Sprint 9 — Synthetic Ablation Experiment")
    print("NOTE: Synthetic data only. NOT a clinical evaluation.")
    print("Results use a provisional threshold under unresolved CGM unit provenance.")
    print("=" * 64)

    df = make_dataset(1000)
    n_train, n_val = 700, 150
    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_train + n_val]
    df_test  = df.iloc[n_train + n_val:]

    targets = [
        ("future_hypoglycemia_30m", "30m"),
        ("future_hypoglycemia_60m", "60m"),
    ]

    results = {}
    for target_col, horizon in targets:
        results[horizon] = {}
        for config in [ModelConfig.BASELINE, ModelConfig.TWIN_ONLY, ModelConfig.HYBRID]:
            model = HybridModel(config, random_state=SEED)
            model.fit(df_train, target_col, df_val=df_val)

            probs_val = model.predict_proba(df_val)
            y_val = df_val[target_col].values

            # Calibration (on val only)
            cal = select_calibrator(len(df_val))
            cal.fit(probs_val, y_val)

            probs_test = model.predict_proba(df_test)
            probs_test_cal = cal.calibrate(probs_test)
            y_test = df_test[target_col].values

            result = model.evaluate(df_test, target_col, horizon)
            bs_raw = brier_score(probs_test, y_test)
            bs_cal = brier_score(probs_test_cal, y_test)

            results[horizon][config.value] = {
                "roc_auc": result.metrics.get("roc_auc"),
                "pr_auc": result.metrics.get("pr_auc"),
                "recall": result.metrics.get("recall"),
                "f1": result.metrics.get("f1"),
                "brier_raw": round(bs_raw, 4),
                "brier_cal": round(bs_cal, 4),
                "n_features": len(model.feature_cols),
            }

    # Uncertainty estimate (hybrid, 30m only)
    feature_cols = [c for c in df_train.columns if c.startswith("twin_") or c in [
        "glucose_current", "glucose_roc_5m"]]
    ens = BootstrapEnsemble(n_members=10, random_state=SEED)
    ens.fit(df_train, df_train["future_hypoglycemia_30m"], feature_cols)
    mean_p, lo, hi, std = ens.predict_uncertainty(df_test)

    # Twin trajectory eval (synthetic)
    pred30 = list(df_test["twin_glucose_t30"].values[:50])
    obs30  = list(df_test["glucose_current"].values[:50])
    twin_metrics_30 = evaluate_twin_forecast(pred30, obs30, 30)

    pred60 = list(df_test["twin_glucose_t60"].values[:50])
    obs60  = list(df_test["glucose_current"].values[:50])
    twin_metrics_60 = evaluate_twin_forecast(pred60, obs60, 60)

    runtime = time.time() - t0
    mem1 = psutil.Process().memory_info().rss / (1024**2)

    print("\n── 30m Results ────────────────────────────────────────────")
    for config_name, m in results["30m"].items():
        print(f"  {config_name:12s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"Brier(raw)={m['brier_raw']}  Brier(cal)={m['brier_cal']}  "
              f"Features={m['n_features']}")

    print("\n── 60m Results ────────────────────────────────────────────")
    for config_name, m in results["60m"].items():
        print(f"  {config_name:12s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"Brier(raw)={m['brier_raw']}  Brier(cal)={m['brier_cal']}  "
              f"Features={m['n_features']}")

    print("\n── Twin Trajectory Evaluation (Synthetic) ─────────────────")
    print(f"  30m: n={twin_metrics_30.n_valid}  MAE={twin_metrics_30.mae:.2f}  "
          f"RMSE={twin_metrics_30.rmse:.2f}  Bias={twin_metrics_30.mean_bias:.2f}")
    print(f"  60m: n={twin_metrics_60.n_valid}  MAE={twin_metrics_60.mae:.2f}  "
          f"RMSE={twin_metrics_60.rmse:.2f}  Bias={twin_metrics_60.mean_bias:.2f}")
    print(f"  Note: {twin_metrics_30.note}")

    print("\n── Bootstrap Uncertainty (Hybrid 30m) ──────────────────────")
    print(f"  Mean prob range: [{mean_p.min():.3f}, {mean_p.max():.3f}]")
    print(f"  Mean 90% width: {(hi - lo).mean():.3f}")
    print(f"  Status: bootstrap_ensemble (not calibrated clinical CI)")

    print("\n── Resource Usage ──────────────────────────────────────────")
    print(f"  Runtime:        {runtime:.2f}s")
    print(f"  Peak memory:    {mem1:.1f} MB")
    print(f"  Train windows:  {n_train}")
    print(f"  Val windows:    {n_val}")
    print(f"  Test windows:   {len(df_test)}")
    print("=" * 64)


if __name__ == "__main__":
    run_ablation()
