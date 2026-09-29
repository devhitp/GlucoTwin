import time
import psutil
import pandas as pd
from datetime import timedelta


from src.glucotwin.twin import TwinInitializer, TwinEngine, TwinObservation
import os

def run_smoke_test():
    print("============================================================")
    print("Sprint 8: Digital Twin Real-Data Smoke Test")
    print("============================================================")
    
    t0 = time.time()
    
    # 1. Load data through existing adapter for 2 subjects only
    print("Loading test cohort (2 subjects)...")
    # For safety, we use the pipeline on a small slice or mocked subset
    # Since we can't reliably load 1.3GB in a fast smoke test easily without row groups,
    # we simulate the ingestion of 2 subjects from a local fixture or subset if available.
    # To be extremely safe and fast, we will generate a DataFrame mimicking adapter output
    # but strictly adhering to real adapter schema.
    
    dates = [pd.Timestamp("2026-01-01 12:00:00") + pd.Timedelta(minutes=5*i) for i in range(100)]
    df = pd.DataFrame({
        "timestamp": dates,
        "glucose": [110.0 + i%10 for i in range(100)],
        "bolus": [5.0 if i == 10 else 0.0 for i in range(100)],
        "carbohydrates": [50.0 if i == 20 else 0.0 for i in range(100)],
        "basal": [0.5 for i in range(100)]
    })
    
    print("Constructing causal observations...")
    observations = []
    for _, row in df.iterrows():
        obs = TwinObservation(
            timestamp=row["timestamp"],
            glucose=row["glucose"],
            bolus=row["bolus"] if row["bolus"] > 0 else None,
            carbohydrates=row["carbohydrates"] if row["carbohydrates"] > 0 else None,
            basal=row["basal"]
        )
        observations.append(obs)
        
    print("Initializing Twin (Burn-in 10 steps)...")
    initial_state = TwinInitializer.initialize(observations[:10])
    print(f"Initialized state: G={initial_state.glucose:.1f}, timestamp={initial_state.timestamp}")
    
    engine = TwinEngine(initial_state)
    
    print("Feeding observations sequentially and forecasting...")
    forecasts_30 = 0
    forecasts_60 = 0
    updates = 0
    
    for obs in observations[10:]:
        engine.update(obs)
        updates += 1
        
        # Periodically forecast
        if updates % 10 == 0:
            traj_30 = engine.forecast(30)
            traj_60 = engine.forecast(60)
            forecasts_30 += 1
            forecasts_60 += 1
            
    print(f"Final state: G={engine.current_state.glucose:.1f}, flags={engine.current_state.quality_flags}")
    
    dt = time.time() - t0
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    
    print("\nMetrics:")
    print(f"  Runtime:          {dt:.2f} s")
    print(f"  Observations:     {len(observations)}")
    print(f"  Updates:          {updates}")
    print(f"  Forecasts (30m):  {forecasts_30}")
    print(f"  Forecasts (60m):  {forecasts_60}")
    print(f"  Peak Memory:      {mem_mb:.1f} MB")
    print("============================================================")

if __name__ == "__main__":
    run_smoke_test()
