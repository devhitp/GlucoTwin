import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from scripts.audit_metabonet import identify_semantic_fields

def test_semantic_fields_identification():
    columns = ["id", "date", "CGM", "insulin", "carbs", "steps"]
    semantics = identify_semantic_fields(columns)
    
    assert "id" in semantics["Identity"]["CONFIRMED"]
    assert "date" in semantics["Time"]["CONFIRMED"]
    assert "CGM" in semantics["Glucose"]["CONFIRMED"]
    assert "insulin" in semantics["Insulin"]["CONFIRMED"]
    assert "carbs" in semantics["Meals"]["CONFIRMED"]
    assert "steps" in semantics["Activity"]["CONFIRMED"]
    assert len(semantics["Wearables"]["CONFIRMED"]) == 0
    assert len(semantics["Sleep"]["CONFIRMED"]) == 0

def test_parquet_memory_safe_audit(tmp_path):
    # Create a tiny mock parquet
    df = pd.DataFrame({
        "id": ["p1", "p1", "p2"],
        "date": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:00"]),
        "CGM": [100.0, None, 120.0]
    })
    
    file_path = tmp_path / "mock.parquet"
    df.to_parquet(file_path)
    
    # Audit structure
    pf = pq.ParquetFile(file_path)
    assert pf.metadata.num_rows == 3
    assert pf.metadata.num_columns == 3
    
    # Validate chunks
    total_nulls = 0
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i).to_pandas()
        total_nulls += rg["CGM"].isna().sum()
        
    assert total_nulls == 1
