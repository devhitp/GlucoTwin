import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glucotwin.modeling.datasets import FeatureSanitizer
from src.glucotwin.modeling.baselines import PersistenceBaseline
from src.glucotwin.modeling.lightgbm_model import LightGBMModel
from src.glucotwin.modeling.evaluation import Evaluation
from src.glucotwin.modeling.calibration import CalibrationEvaluation
from src.glucotwin.modeling.thresholds import ThresholdAnalysis
from src.glucotwin.modeling.feature_importance import FeatureImportance
from src.glucotwin.modeling.errors import ErrorAnalysis
from tests.fixtures.synthetic.modeling_generator import generate_synthetic_modeling_fixture

def main():
    print("========================================")
    print("GLUCOTWIN EVALUATION REPORT")
    print("========================================")
    print("SYNTHETIC FIXTURE RESULT — NOT REAL OHIO T1DM PERFORMANCE\n")
    
    # Re-run pipeline to get test set
    df = generate_synthetic_modeling_fixture(rows=2000)
    target = 'future_hypoglycemia_30m'
    X, y = FeatureSanitizer.sanitize(df, target)
    
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
    
    # Train
    model = LightGBMModel(random_state=42)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Persistence Baseline
    from src.glucotwin.config.clinical import get_hypo_threshold
    baseline = PersistenceBaseline(threshold=get_hypo_threshold())
    baseline_probs = baseline.predict_proba(X_test)
    base_metrics = Evaluation.calculate_metrics(y_test, baseline_probs, threshold=0.5)
    
    print("--- Persistence Baseline ---")
    print(f"ROC-AUC: {base_metrics.get('roc_auc', 'unavailable')}")
    print(f"PR-AUC: {base_metrics.get('pr_auc', 'unavailable')}")
    print(f"F1: {base_metrics.get('f1', 'unavailable')}")
    
    # ML Baseline
    lgb_probs = model.predict_proba(X_test)
    lgb_metrics = Evaluation.calculate_metrics(y_test, lgb_probs, threshold=0.5)
    
    print("\n--- LightGBM Model ---")
    print(f"ROC-AUC: {lgb_metrics.get('roc_auc', 'unavailable')}")
    print(f"PR-AUC: {lgb_metrics.get('pr_auc', 'unavailable')}")
    print(f"F1: {lgb_metrics.get('f1', 'unavailable')}")
    print(f"Balanced Accuracy: {lgb_metrics.get('balanced_accuracy', 'unavailable')}")
    
    # Calibration
    brier = CalibrationEvaluation.calculate_brier_score(y_test, lgb_probs)
    print(f"\nCalibration (Brier Score): {brier:.4f}")
    
    # Threshold Analysis
    print("\n--- Threshold Analysis (Validation Set) ---")
    val_probs = model.predict_proba(X_val)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    df_thresh = ThresholdAnalysis.evaluate_thresholds(y_val, val_probs, thresholds)
    print(df_thresh.to_string(index=False))
    
    # Feature Importance
    print("\n--- Top Features ---")
    fi = FeatureImportance.get_importance(model, list(X.columns))
    print(fi.head().to_string(index=False))
    
    # Error Analysis
    print("\n--- Error Analysis ---")
    errors = ErrorAnalysis.analyze_errors(X_test, y_test, (lgb_probs >= 0.5).astype(int))
    print(f"False Positives: {len(errors[errors['error_type'] == 'FP'])}")
    print(f"False Negatives: {len(errors[errors['error_type'] == 'FN'])}")

if __name__ == "__main__":
    main()
