import os
from pathlib import Path
from typing import Dict, List, Any
from .parser import DataParser

class DatasetLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_patient(self, patient_id: str) -> Dict[str, List[Any]]:
        """Loads data for a specific patient."""
        file_path = self.data_path / f"{patient_id}.xml"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file for patient {patient_id} not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return DataParser.parse_synthetic_xml(content, patient_id)

    def discover_patients(self) -> List[str]:
        """Discovers available patient IDs based on XML files in the data path."""
        if not self.data_path.exists():
            return []
        
        patients = []
        for file in self.data_path.glob("*.xml"):
            patients.append(file.stem)
        return sorted(patients)
