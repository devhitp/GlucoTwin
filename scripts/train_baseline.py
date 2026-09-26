import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glucotwin.modeling.datasets import FeatureSanitizer
from src.glucotwin.modeling.baselines import PersistenceBaseline
from src.glucotwin.modeling.lightgbm_model import LightGBMModel
from src.glucotwin.data.preprocessing.splits import DataSplitter
from tests.fixtures.synthetic.modeling_generator import generate_synthetic_modeling_fixture

def main():
    print("========================================")
    print("GLUCOTWIN MODEL TRAINING PIPELINE")
    print("========================================")
    
    # Load data
    df = generate_synthetic_modeling_fixture(rows=2000)
    print("Loaded SYNTHETIC MODELING FIXTURE — NOT REAL OHIO T1DM")
    
    target = 'future_hypoglycemia_30m'
    X, y = FeatureSanitizer.sanitize(df, target)
    
    # Chronological Split (Train 60%, Val 20%, Test 20%)
    train_ratio = 0.6
    val_ratio = 0.2
    
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
    
    print(f"Data Split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"Target Positive Rate: {y.mean():.4f}")
    
    # Train LightGBM
    print("\nTraining LightGBM baseline...")
    model = LightGBMModel(random_state=42)
    model.fit(X_train, y_train, X_val, y_val)
    print("Training complete.")
    
    # Save artifacts/metadata
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    metadata = {
        "target": target,
        "features": list(X.columns),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "positive_rate": float(y.mean())
    }
    
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved model metadata to {artifacts_dir / 'model_metadata.json'}")

if __name__ == "__main__":
    main()
