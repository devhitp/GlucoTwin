from typing import List, Dict, Any
import pandas as pd
from datetime import timedelta
from .schema import SynchronizedRecord

class DataSynchronizer:
    @staticmethod
    def synchronize(records: Dict[str, List[Any]], freq: str = '5min') -> List[SynchronizedRecord]:
        """
        Synchronize disparate records into a single timeline.
        Sprint 1: Basic alignment, chronological sorting, and duplicate handling.
        """
        
        all_records = []
        for key, rec_list in records.items():
            for r in rec_list:
                all_records.append((r.timestamp, key, r))
                
        if not all_records:
            return []
            
        # Chronological sort
        all_records.sort(key=lambda x: x[0])
        
        # Get patient_id
        patient_id = all_records[0][2].patient_id
        
        # Group by aligned timestamps (floor to nearest frequency)
        df_records = []
        for ts, k, r in all_records:
            df_records.append({
                'timestamp': ts.floor(freq) if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).floor(freq),
                'type': k,
                'record': r
            })
            
        df = pd.DataFrame(df_records)
        if df.empty:
            return []
            
        synced = []
        for ts, group in df.groupby('timestamp'):
            sync_rec = SynchronizedRecord(
                patient_id=patient_id,
                timestamp=ts.to_pydatetime(),
                glucose=None,
                basal_insulin=None,
                bolus_insulin=None,
                carbohydrates=None,
                heart_rate=None,
                eda=None,
                skin_temperature=None,
                accelerometer_x=None,
                accelerometer_y=None,
                accelerometer_z=None
            )
            
            for _, row in group.iterrows():
                r = row['record']
                if row['type'] == 'cgm':
                    if sync_rec.glucose is None:
                        sync_rec.glucose = r.glucose
                elif row['type'] == 'insulin':
                    if r.basal_insulin is not None and sync_rec.basal_insulin is None:
                        sync_rec.basal_insulin = r.basal_insulin
                    if r.bolus_insulin is not None and sync_rec.bolus_insulin is None:
                        sync_rec.bolus_insulin = r.bolus_insulin
                elif row['type'] == 'meal':
                    if sync_rec.carbohydrates is None:
                        sync_rec.carbohydrates = r.carbohydrates
                elif row['type'] == 'wearable':
                    if sync_rec.heart_rate is None:
                        sync_rec.heart_rate = r.heart_rate
                        sync_rec.eda = r.eda
                        sync_rec.skin_temperature = r.skin_temperature
                        sync_rec.accelerometer_x = r.accelerometer_x
                        sync_rec.accelerometer_y = r.accelerometer_y
                        sync_rec.accelerometer_z = r.accelerometer_z
                        
            synced.append(sync_rec)
            
        return synced
