import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import argparse

def analyze(file_path):
    dataset = ds.dataset(file_path, format="parquet")
    print("====================================================")
    print("GLUCOTWIN — METABONET DATA QUALIFICATION (OPTIMIZED)")
    print("====================================================")

    # We will use scanners to fetch columns and compute stats
    
    # 1. CGM
    scanner = dataset.scanner(columns=["CGM"])
    table = scanner.to_table()
    cgm_valid = table.filter(pc.is_valid(table["CGM"]))["CGM"]
    cgm_count = len(cgm_valid)
    if cgm_count > 0:
        print(f"\n--- CGM ---")
        print(f"Count: {cgm_count}")
        print(f"Min: {pc.min(cgm_valid).as_py()}, Max: {pc.max(cgm_valid).as_py()}, Mean: {pc.mean(cgm_valid).as_py():.2f}")
        below_70 = pc.sum(pc.less(cgm_valid, 70)).as_py()
        below_54 = pc.sum(pc.less(cgm_valid, 54)).as_py()
        print(f"Below 70: {below_70} ({(below_70/cgm_count)*100:.2f}%)")
        print(f"Below 54: {below_54} ({(below_54/cgm_count)*100:.2f}%)")

    # 2. Insulin
    scanner = dataset.scanner(columns=["insulin", "basal", "bolus"])
    table = scanner.to_table()
    for col in ["insulin", "basal", "bolus"]:
        valid = table.filter(pc.is_valid(table[col]))[col]
        print(f"\n--- {col.upper()} ---")
        print(f"Count: {len(valid)}")
        print(f"Zeros: {pc.sum(pc.equal(valid, 0.0)).as_py()}")
        print(f"Negatives: {pc.sum(pc.less(valid, 0.0)).as_py()}")
        print(f"Max: {pc.max(valid).as_py()}")

    # Check relation
    valid_all = table.filter(pc.and_(pc.and_(pc.is_valid(table["insulin"]), pc.is_valid(table["basal"])), pc.is_valid(table["bolus"])))
    if len(valid_all) > 0:
        sum_bb = pc.add(valid_all["basal"], valid_all["bolus"])
        match = pc.sum(pc.equal(valid_all["insulin"], sum_bb)).as_py()
        print(f"\nInsulin == Basal + Bolus check: {match} matches out of {len(valid_all)}")

    # 3. Carbs
    scanner = dataset.scanner(columns=["carbs"])
    valid = scanner.to_table().filter(pc.is_valid(scanner.to_table()["carbs"]))["carbs"]
    count = len(valid)
    print(f"\n--- CARBS ---")
    print(f"Count: {count}")
    print(f"Zeros: {pc.sum(pc.equal(valid, 0.0)).as_py()}")
    print(f"Negatives: {pc.sum(pc.less(valid, 0.0)).as_py()}")
    print(f"Max: {pc.max(valid).as_py()}")
    
    # 4. Dates
    scanner = dataset.scanner(columns=["date"])
    valid_dates = scanner.to_table().filter(pc.is_valid(scanner.to_table()["date"]))["date"]
    years = pc.year(valid_dates)
    unique_years = pc.unique(years).to_pylist()
    print("\n--- TIMESTAMPS ---")
    print(f"Years found: {sorted(unique_years)}")
    
    # We will sample a few rows to see if 2027 is prevalent or just a few
    count_2027 = pc.sum(pc.greater_equal(years, 2027)).as_py()
    print(f"Rows >= 2027: {count_2027}")

    # 5. Subjects duration & Candidate Windows
    print("\n--- SUBJECT COVERAGE & CANDIDATE WINDOWS ---")
    print("To keep this memory safe, we will use pandas groupby on projected columns")
    # We project id, date, CGM
    scanner = dataset.scanner(columns=["id", "date", "CGM", "insulin", "carbs"])
    df = scanner.to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    
    durations = []
    cgm_counts = []
    ins_counts = []
    carb_counts = []
    
    for subj, group in df.groupby("id"):
        t_min = group["date"].min()
        t_max = group["date"].max()
        dur = (t_max - t_min).total_seconds() / 3600.0 if pd.notna(t_min) else 0
        durations.append(dur)
        cgm_counts.append(group["CGM"].notna().sum())
        ins_counts.append(group["insulin"].notna().sum())
        carb_counts.append(group["carbs"].notna().sum())
        
    gt_24h = sum(1 for d in durations if d >= 24)
    gt_7d = sum(1 for d in durations if d >= 24 * 7)
    gt_14d = sum(1 for d in durations if d >= 24 * 14)
    gt_30d = sum(1 for d in durations if d >= 24 * 30)
    
    import numpy as np
    print(f"Subjects: {len(durations)}")
    if durations:
        print(f"Min Duration: {np.min(durations):.2f} hours")
        print(f"Median Duration: {np.median(durations)/24:.2f} days")
        print(f"Max Duration: {np.max(durations)/24:.2f} days")
        print(f">= 24h: {gt_24h}, >= 7d: {gt_7d}, >= 14d: {gt_14d}, >= 30d: {gt_30d}")
        
    core = sum(1 for d, c, i, cb in zip(durations, cgm_counts, ins_counts, carb_counts) if d >= 24*14 and c > 1000 and i > 0 and cb > 0)
    print(f"Core Cohort (>=14d, has CGM/Ins/Carb): {core}")
    
    # CGM Continuity & 30m windows
    # Since rows are 5 min apart in this dataset, a candidate window for 30m needs 6 future rows.
    print(f"Candidate 30m Windows: ~{sum(cgm_counts)} (assuming continuous blocks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    analyze(args.input)
