import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glucotwin.config import GLUCOTWIN_DATA_PATH
from src.glucotwin.data.loader import DatasetLoader
from src.glucotwin.data.validator import DataValidator
from src.glucotwin.data.synchronizer import DataSynchronizer

def main():
    print("========================================")
    print("GLUCOTWIN DATASET REPORT")
    print("========================================")
    
    loader = DatasetLoader(GLUCOTWIN_DATA_PATH)
    patients = loader.discover_patients()
    
    if not patients:
        print("REAL DATASET NOT FOUND")
        print("Falling back to synthetic fixture for report demonstration.")
        loader = DatasetLoader(str(Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic"))
        patients = loader.discover_patients()
    
    print(f"Patients: {len(patients)}")
    
    for pid in patients:
        print(f"\nPatient: {pid}")
        try:
            records = loader.load_patient(pid)
            
            val_report = DataValidator.validate_records(records)
            if not val_report["valid"]:
                print("  VALIDATION FAILED:")
                for e in val_report["errors"]:
                    print(f"    - {e}")
            
            cgm_count = len(records.get('cgm', []))
            ins_count = len(records.get('insulin', []))
            meal_count = len(records.get('meal', []))
            wear_count = len(records.get('wearable', []))
            
            print(f"  CGM Records: {cgm_count}")
            print(f"  Insulin Records: {ins_count}")
            print(f"  Meal Records: {meal_count}")
            print(f"  Wearable Records: {wear_count}")
            
            if cgm_count > 0:
                cgm_start = records['cgm'][0].timestamp
                cgm_end = records['cgm'][-1].timestamp
                print(f"  Date range: {cgm_start} -> {cgm_end}")
            
            synced = DataSynchronizer.synchronize(records)
            print(f"  Synchronized Timestamps: {len(synced)}")
            
        except Exception as e:
            print(f"  Error loading patient: {e}")

if __name__ == "__main__":
    main()
