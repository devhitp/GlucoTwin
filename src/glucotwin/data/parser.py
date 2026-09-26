from datetime import datetime
from typing import Optional, List, Dict, Any
from .schema import CGMRecord, InsulinRecord, MealRecord, WearableRecord
import xml.etree.ElementTree as ET

def parse_timestamp(ts_str: str) -> datetime:
    # Typical OhioT1DM timestamp format: DD-MM-YYYY HH:MM:SS or similar
    # In synthetic data we might just use ISO format
    try:
        return datetime.fromisoformat(ts_str.replace(" ", "T"))
    except ValueError:
        # Fallback for DD-MM-YYYY HH:MM:SS
        return datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")

def parse_float(val: Optional[str]) -> Optional[float]:
    if not val or val.lower() == 'nan':
        return None
    try:
        return float(val)
    except ValueError:
        return None

class DataParser:
    @staticmethod
    def parse_synthetic_xml(xml_content: str, patient_id: str) -> Dict[str, List[Any]]:
        records: Dict[str, List[Any]] = {
            "cgm": [],
            "insulin": [],
            "meal": [],
            "wearable": []
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            # parse cgm
            for event in root.findall(".//glucose_level/event"):
                ts_str = event.get('ts')
                val_str = event.get('value')
                if ts_str:
                    records["cgm"].append(CGMRecord(
                        patient_id=patient_id,
                        timestamp=parse_timestamp(ts_str),
                        glucose=parse_float(val_str)
                    ))
            
            # parse insulin
            for event in root.findall(".//basal/event"):
                ts_str = event.get('ts')
                val_str = event.get('value')
                if ts_str:
                    records["insulin"].append(InsulinRecord(
                        patient_id=patient_id,
                        timestamp=parse_timestamp(ts_str),
                        basal_insulin=parse_float(val_str),
                        bolus_insulin=None
                    ))
            
            for event in root.findall(".//bolus/event"):
                ts_str = event.get('ts')
                val_str = event.get('dose')
                if ts_str:
                    records["insulin"].append(InsulinRecord(
                        patient_id=patient_id,
                        timestamp=parse_timestamp(ts_str),
                        basal_insulin=None,
                        bolus_insulin=parse_float(val_str)
                    ))

            # parse meals
            for event in root.findall(".//meal/event"):
                ts_str = event.get('ts')
                val_str = event.get('carbs')
                if ts_str:
                    records["meal"].append(MealRecord(
                        patient_id=patient_id,
                        timestamp=parse_timestamp(ts_str),
                        carbohydrates=parse_float(val_str)
                    ))
            
            # parse wearables
            for event in root.findall(".//wearable/event"):
                ts_str = event.get('ts')
                hr_str = event.get('hr')
                if ts_str:
                    records["wearable"].append(WearableRecord(
                        patient_id=patient_id,
                        timestamp=parse_timestamp(ts_str),
                        heart_rate=parse_float(hr_str),
                        eda=parse_float(event.get('eda')),
                        skin_temperature=parse_float(event.get('temp')),
                        accelerometer_x=parse_float(event.get('acc_x')),
                        accelerometer_y=parse_float(event.get('acc_y')),
                        accelerometer_z=parse_float(event.get('acc_z')),
                    ))
                    
        except ET.ParseError:
            pass # Invalid XML
            
        return records
