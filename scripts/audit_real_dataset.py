import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glucotwin.data.loader import DatasetLoader
from src.glucotwin.data.parser import DataParser

def main():
    print("====================================================")
    print("GLUCOTWIN — OHIO T1DM DATASET AUDIT")
    print("====================================================")
    
    loader = DatasetLoader(Path("data/raw"))
    all_patients = loader.discover_patients()
    
    # Filter out synthetic fixtures
    real_patients = [p for p in all_patients if p != "559"]
    
    if not real_patients:
        print("Dataset detected: NO")
        print("\nREAL OHIO T1DM DATASET NOT FOUND")
        print("REAL OHIO T1DM DATASET NOT AVAILABLE — REAL-DATA VALIDATION PENDING")
        return
        
    print("Dataset detected: YES")
    print("\nPatients:")
    
    total_records = 0
    streams = {
        "CGM": False,
        "Insulin": False,
        "Meals": False,
        "Heart rate": False,
        "EDA": False,
        "Skin temperature": False,
        "Accelerometry": False,
        "Sleep/context": False
    }
    
    for patient_id in real_patients:
        try:
            parsed_records = loader.load_patient(patient_id)
            records = []
            for k in parsed_records:
                records.extend(parsed_records[k])
                
            if not records:
                continue
            
            start_time = min(r.timestamp for r in records)
            end_time = max(r.timestamp for r in records)
            print(f"{patient_id} → {start_time} → {end_time}")
            
            total_records += len(records)
            
            if parsed_records.get("cgm"): streams["CGM"] = True
            if parsed_records.get("insulin"): streams["Insulin"] = True
            if parsed_records.get("meal"): streams["Meals"] = True
            
            wearables = parsed_records.get("wearable", [])
            if wearables:
                if any(r.heart_rate is not None for r in wearables): streams["Heart rate"] = True
                if any(r.eda is not None for r in wearables): streams["EDA"] = True
                if any(r.skin_temperature is not None for r in wearables): streams["Skin temperature"] = True
                if any(r.accelerometer_x is not None for r in wearables): streams["Accelerometry"] = True
                
        except Exception as e:
            print(f"Error parsing {patient_id}: {e}")
            
    print(f"\nTotal Patients: {len(real_patients)}")
    print(f"Total Canonical Records: {total_records}")
    
    print("\nStreams:")
    for k, v in streams.items():
        print(f"{k}: {'YES' if v else 'NO'}")

if __name__ == "__main__":
    main()
