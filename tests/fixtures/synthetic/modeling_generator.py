import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_modeling_fixture(rows: int = 1000) -> pd.DataFrame:
    """
    SYNTHETIC MODELING FIXTURE — NOT REAL OHIO T1DM
    """
    np.random.seed(42)
    
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(rows)]
    
    # Generate some synthetic glucose that oscillates and sometimes drops below 70
    time_seq = np.linspace(0, 10 * np.pi, rows)
    base_glucose = 120 + 50 * np.sin(time_seq)
    noise = np.random.normal(0, 5, rows)
    glucose = base_glucose + noise
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_current": glucose,
        "glucose_roc_5m": np.gradient(glucose) / 5.0,
        "insulin_bolus_last_30m": np.random.exponential(1.0, rows),
        "carbs_last_30m": np.random.exponential(5.0, rows),
        "is_night_clock_based": [1 if (t.hour >= 22 or t.hour < 6) else 0 for t in timestamps],
    })
    
    # Create the future labels to mimic preprocessing
    df['future_hypoglycemia_30m'] = (df['glucose_current'].shift(-6).rolling(6, min_periods=1).min() < 70).astype(float)
    df['future_hypoglycemia_60m'] = (df['glucose_current'].shift(-12).rolling(12, min_periods=1).min() < 70).astype(float)
    
    # Drop NaNs at the end
    df = df.dropna()
    df = df.set_index('timestamp')
    return df
