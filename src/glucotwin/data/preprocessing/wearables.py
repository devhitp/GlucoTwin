import pandas as pd
import numpy as np

class WearableFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        features = df.copy()
        
        if 'heart_rate' in features.columns:
            features['hr_current'] = features['heart_rate'].ffill()
            features['hr_rolling_mean_30m'] = features['hr_current'].rolling('30min').mean()
            features['hr_rolling_std_30m'] = features['hr_current'].rolling('30min').std()
            
        if 'eda' in features.columns:
            features['eda_current'] = features['eda'].ffill()
            features['eda_rolling_mean_30m'] = features['eda_current'].rolling('30min').mean()
            
        if 'skin_temperature' in features.columns:
            features['temp_current'] = features['skin_temperature'].ffill()
            features['temp_rolling_mean_30m'] = features['temp_current'].rolling('30min').mean()
            
        has_acc = all(c in features.columns for c in ['accelerometer_x', 'accelerometer_y', 'accelerometer_z'])
        if has_acc:
            features['acc_magnitude'] = np.sqrt(
                features['accelerometer_x'].astype(float)**2 + 
                features['accelerometer_y'].astype(float)**2 + 
                features['accelerometer_z'].astype(float)**2
            )
            # Ffill missing magnitude if appropriate
            features['acc_magnitude'] = features['acc_magnitude'].ffill()
            features['acc_rolling_mean_15m'] = features['acc_magnitude'].rolling('15min').mean()
            features['acc_rolling_std_15m'] = features['acc_magnitude'].rolling('15min').std()
            
        return features
