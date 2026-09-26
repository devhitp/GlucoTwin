from typing import List, Dict, Any

class DataValidator:
    @staticmethod
    def validate_records(records: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Validate patient integrity, timestamps, and values.
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        patient_ids = set()
        
        for key, rec_list in records.items():
            last_ts = None
            for i, r in enumerate(rec_list):
                patient_ids.add(r.patient_id)
                
                # Check timestamp ordering
                if last_ts and r.timestamp < last_ts:
                    report["errors"].append(f"Non-chronological ordering in {key} at {r.timestamp}")
                    report["valid"] = False
                elif last_ts and r.timestamp == last_ts:
                    report["warnings"].append(f"Duplicate timestamp in {key} at {r.timestamp}")
                last_ts = r.timestamp
                
                # Check specifics
                if key == 'cgm':
                    if r.glucose is not None and r.glucose < 0:
                        report["errors"].append(f"Negative glucose value at {r.timestamp}")
                        report["valid"] = False
                elif key == 'insulin':
                    if r.basal_insulin is not None and r.basal_insulin < 0:
                        report["errors"].append(f"Negative basal insulin at {r.timestamp}")
                        report["valid"] = False
                    if r.bolus_insulin is not None and r.bolus_insulin < 0:
                        report["errors"].append(f"Negative bolus insulin at {r.timestamp}")
                        report["valid"] = False
                elif key == 'meal':
                    if r.carbohydrates is not None and r.carbohydrates < 0:
                        report["errors"].append(f"Negative carbohydrates at {r.timestamp}")
                        report["valid"] = False
                        
        if len(patient_ids) > 1:
            report["errors"].append(f"Multiple patient IDs found in single record set: {patient_ids}")
            report["valid"] = False
            
        return report
