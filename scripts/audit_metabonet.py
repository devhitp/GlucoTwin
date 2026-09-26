import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
import json
import pandas as pd
import numpy as np

def identify_semantic_fields(columns):
    semantic_map = {
        "Identity": {"subject_id": [], "session_id": [], "record_id": [], "id": []},
        "Time": {"timestamp": [], "date": [], "datetime": [], "timezone": []},
        "Glucose": {"CGM": [], "glucose": [], "glucose_value": [], "glucose_timestamp": []},
        "Insulin": {"insulin": [], "basal": [], "bolus": [], "insulin_dose": [], "insulin_timestamp": []},
        "Meals": {"carbohydrate": [], "carbs": [], "meal": [], "nutrition": [], "food": []},
        "Activity": {"steps": [], "activity": [], "accelerometer": [], "exercise": []},
        "Wearables": {"heart_rate": [], "HR": [], "EDA": [], "skin_temperature": [], "temperature": []},
        "Sleep": {"sleep": [], "sleep_stage": [], "sleep_duration": []}
    }
    
    results = {}
    for group, keywords in semantic_map.items():
        results[group] = {"CONFIRMED": [], "POSSIBLE": [], "NOT FOUND": []}
        for kw in keywords:
            found = False
            for col in columns:
                if kw.lower() == col.lower():
                    results[group]["CONFIRMED"].append(col)
                    found = True
                elif kw.lower() in col.lower():
                    results[group]["POSSIBLE"].append(col)
                    found = True
            if not found:
                results[group]["NOT FOUND"].append(kw)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Audit MetaboNet Parquet Dataset")
    parser.add_argument("--input", required=True, help="Path to parquet file")
    args = parser.parse_args()

    file_path = args.input
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("====================================================")
    print("GLUCOTWIN — METABONET DATASET AUDIT")
    print("====================================================")

    # File Stats
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n[ FILE METADATA ]")
    print(f"File: {file_path}")
    print(f"Size: {file_size_mb:.2f} MB")

    try:
        pf = pq.ParquetFile(file_path)
    except Exception as e:
        print(f"Failed to open parquet file: {e}")
        return

    print(f"Total Rows: {pf.metadata.num_rows}")
    print(f"Row Groups: {pf.metadata.num_row_groups}")
    print(f"Columns: {pf.metadata.num_columns}")

    # Schema Inventory
    print("\n[ COLUMN INVENTORY ]")
    schema = pf.schema
    columns = []
    for i in range(len(schema)):
        field = schema[i]
        col_name = field.name
        columns.append(col_name)
        log_type = field.logical_type
        phys_type = field.physical_type
        print(f" - {col_name}: {phys_type} (Logical: {log_type})")

    # Semantic Fields
    print("\n[ SEMANTIC FIELD IDENTIFICATION ]")
    semantics = identify_semantic_fields(columns)
    for group, statuses in semantics.items():
        print(f"--- {group} ---")
        if statuses["CONFIRMED"]:
            print(f"  CONFIRMED: {', '.join(statuses['CONFIRMED'])}")
        if statuses["POSSIBLE"]:
            print(f"  POSSIBLE: {', '.join(statuses['POSSIBLE'])}")
        if not statuses["CONFIRMED"] and not statuses["POSSIBLE"]:
            print("  NOT FOUND")

    # Read a small sample
    print("\n[ SAFE SAMPLE INSPECTION (First 10 rows) ]")
    sample_table = pf.read_row_group(0)
    sample_df = sample_table.slice(0, 10).to_pandas()
    # Print non-sensitive summary
    for col in sample_df.columns:
        vals = sample_df[col].dropna().head(3).tolist()
        print(f" - {col}: e.g. {vals}")

    # Analyze full dataset in chunks (memory safe)
    print("\n[ FULL DATASET CHUNK ANALYSIS ]")
    
    unique_subjects = set()
    min_time = None
    max_time = None
    null_counts = {c: 0 for c in columns}
    
    subject_col = None
    if semantics["Identity"]["CONFIRMED"]:
        subject_col = semantics["Identity"]["CONFIRMED"][0]
    elif semantics["Identity"]["POSSIBLE"]:
        subject_col = semantics["Identity"]["POSSIBLE"][0]
        
    time_col = None
    if semantics["Time"]["CONFIRMED"]:
        time_col = semantics["Time"]["CONFIRMED"][0]
    elif semantics["Time"]["POSSIBLE"]:
        time_col = semantics["Time"]["POSSIBLE"][0]

    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i)
        df = rg.to_pandas()
        
        if subject_col and subject_col in df.columns:
            unique_subjects.update(df[subject_col].dropna().unique())
            
        if time_col and time_col in df.columns:
            # Convert to datetime if possible
            t_col_data = pd.to_datetime(df[time_col], errors='coerce').dropna()
            if not t_col_data.empty:
                t_min = t_col_data.min()
                t_max = t_col_data.max()
                if min_time is None or t_min < min_time: min_time = t_min
                if max_time is None or t_max > max_time: max_time = t_max
                
        # Calculate nulls
        for c in df.columns:
            null_counts[c] += df[c].isna().sum()

    print(f"Subject Col Used: {subject_col}")
    print(f"Unique Subjects: {len(unique_subjects)}")
    if unique_subjects:
        print(f"Subject ID Format Example: {list(unique_subjects)[0]}")
    
    print(f"\nTime Col Used: {time_col}")
    print(f"Earliest Timestamp: {min_time}")
    print(f"Latest Timestamp: {max_time}")
    if min_time and max_time:
        print(f"Overall Duration: {max_time - min_time}")
        
    print("\n[ MISSINGNESS ]")
    total_rows = pf.metadata.num_rows
    for c, nulls in null_counts.items():
        pct = (nulls / total_rows) * 100 if total_rows > 0 else 0
        print(f" - {c}: {nulls} nulls ({pct:.2f}%)")

if __name__ == "__main__":
    main()
