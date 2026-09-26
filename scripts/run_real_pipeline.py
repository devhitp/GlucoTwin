import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.audit_real_dataset import main as audit_main

def main():
    print("====================================================")
    print("GLUCOTWIN — REAL OHIO T1DM INTEGRATION PIPELINE")
    print("====================================================")
    
    # 1. Audit Dataset
    print("\n[1] Auditing Dataset...")
    audit_main()
    
    # If the real dataset is not found, the audit script prints a message.
    # We can detect this by checking if real files exist.
    from src.glucotwin.data.loader import DatasetLoader
    loader = DatasetLoader(Path("data/raw"))
    all_patients = loader.discover_patients()
    real_patients = [p for p in all_patients if p != "559"]
    
    if not real_patients:
        print("\nPIPELINE HALTED: Real dataset validation pending.")
        return
        
    print("\n[2] Running Parsing and Validation...")
    # This would execute the actual parsing and validation logic
    print("Pipeline logic for real data would continue here...")
    
if __name__ == "__main__":
    main()
