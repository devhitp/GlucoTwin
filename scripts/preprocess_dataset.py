import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glucotwin.config import GLUCOTWIN_DATA_PATH
from src.glucotwin.data.loader import DatasetLoader
from src.glucotwin.data.synchronizer import DataSynchronizer
from src.glucotwin.data.preprocessing.pipeline import PreprocessingPipeline

def main():
    print("========================================")
    print("GLUCOTWIN DATASET PREPROCESSING REPORT")
    print("========================================")
    
    loader = DatasetLoader(GLUCOTWIN_DATA_PATH)
    patients = loader.discover_patients()
    
    using_synthetic = False
    if not patients:
        print("REAL DATASET NOT AVAILABLE — SYNTHETIC VALIDATION ONLY")
        loader = DatasetLoader(str(Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic"))
        patients = loader.discover_patients()
        using_synthetic = True
    
    print(f"Patients: {len(patients)}")
    
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pid in patients:
        print(f"\nPatient: {pid}")
        try:
            records_dict = loader.load_patient(pid)
            cgm_raw = len(records_dict.get('cgm', []))
            
            records = DataSynchronizer.synchronize(records_dict)
            print(f"  Synchronized Timestamps: {len(records)}")
            
            df = PreprocessingPipeline.process_patient(records)
            
            if not df.empty:
                print(f"  Timeline start: {df.index.min()}")
                print(f"  Timeline end: {df.index.min()}")
                
                print(f"  CGM Records (raw): {cgm_raw}")
                if 'glucose_missing' in df.columns:
                    print(f"  Missing CGM points after sync: {df['glucose_missing'].sum()}")
                    
                print(f"  Insulin Records: {len(records_dict.get('insulin', []))}")
                print(f"  Meal Records: {len(records_dict.get('meal', []))}")
                wear_raw = len(records_dict.get('wearable', []))
                print(f"  Wearable Coverage: {wear_raw}")
                
                # Windows
                windows = PreprocessingPipeline.extract_windows(df, lookback_m=60, horizon_m=60)
                
                print(f"  Usable feature rows: {len(df)}")
                print(f"  Usable temporal windows: {len(windows)}")
                
                if len(df) > 0:
                    dropped_total = len(df) - len(windows)
                    print(f"  Rows dropped (insufficient history/horizon/unknown labels): {dropped_total}")
                
                # Output to parquet/csv
                out_path = out_dir / f"{pid}_processed.csv"
                windows.to_csv(out_path)
                print(f"  Saved to {out_path}")
            
        except Exception as e:
            print(f"  Error processing patient: {e}")

if __name__ == "__main__":
    main()
