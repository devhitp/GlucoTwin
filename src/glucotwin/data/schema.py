from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CGMRecord:
    patient_id: str
    timestamp: datetime
    glucose: Optional[float]

@dataclass
class InsulinRecord:
    patient_id: str
    timestamp: datetime
    basal_insulin: Optional[float]
    bolus_insulin: Optional[float]

@dataclass
class MealRecord:
    patient_id: str
    timestamp: datetime
    carbohydrates: Optional[float]

@dataclass
class WearableRecord:
    patient_id: str
    timestamp: datetime
    heart_rate: Optional[float]
    eda: Optional[float]
    skin_temperature: Optional[float]
    accelerometer_x: Optional[float]
    accelerometer_y: Optional[float]
    accelerometer_z: Optional[float]

@dataclass
class SynchronizedRecord:
    patient_id: str
    timestamp: datetime
    glucose: Optional[float]
    basal_insulin: Optional[float]
    bolus_insulin: Optional[float]
    carbohydrates: Optional[float]
    heart_rate: Optional[float]
    eda: Optional[float]
    skin_temperature: Optional[float]
    accelerometer_x: Optional[float]
    accelerometer_y: Optional[float]
    accelerometer_z: Optional[float]
