import argparse
import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

def qualify_dataset(file_path):
    print("====================================================")
    print("GLUCOTWIN — METABONET DATA QUALIFICATION")
    print("====================================================")

    pf = pq.ParquetFile(file_path)
    print(f"Total Row Groups: {pf.num_row_groups}")

    # Global Stats
    cgm_stats = {"count": 0, "sum": 0.0, "min": float('inf'), "max": float('-inf'), "below_70": 0, "below_54": 0}
    insulin_stats = {"insulin_count": 0, "basal_count": 0, "bolus_count": 0, "insulin_zero": 0, "basal_zero": 0, "bolus_zero": 0, "negative": 0}
    carb_stats = {"count": 0, "zero": 0, "positive": 0, "negative": 0, "max": float('-inf')}
    
    # Check insulin ≈ basal + bolus
    insulin_equality_checks = {"exact_match": 0, "mismatch": 0, "total_checked": 0}

    # Subject Level Stats
    subjects = defaultdict(lambda: {"min_t": pd.Timestamp.max, "max_t": pd.Timestamp.min, "rows": 0, "cgm_count": 0, "insulin_count": 0, "carb_count": 0, "cgm_below_70": 0})
    
    # Timestamp anomalies
    year_distribution = defaultdict(int)

    # Process by row group to be memory safe
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=["id", "date", "CGM", "insulin", "basal", "bolus", "carbs"]).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        
        # Years
        for y in df["date"].dt.year.dropna().unique():
            year_distribution[y] += (df["date"].dt.year == y).sum()

        # Group by subject in this chunk
        for subj, group in df.groupby("id"):
            s = subjects[subj]
            s["rows"] += len(group)
            t_min, t_max = group["date"].min(), group["date"].max()
            if pd.notna(t_min) and t_min < s["min_t"]: s["min_t"] = t_min
            if pd.notna(t_max) and t_max > s["max_t"]: s["max_t"] = t_max
            
            s["cgm_count"] += group["CGM"].notna().sum()
            s["cgm_below_70"] += (group["CGM"] < 70).sum()
            s["insulin_count"] += group["insulin"].notna().sum()
            s["carb_count"] += group["carbs"].notna().sum()

        # CGM Stats
        cgm_valid = df["CGM"].dropna()
        if not cgm_valid.empty:
            cgm_stats["count"] += len(cgm_valid)
            cgm_stats["sum"] += cgm_valid.sum()
            cgm_stats["min"] = min(cgm_stats["min"], cgm_valid.min())
            cgm_stats["max"] = max(cgm_stats["max"], cgm_valid.max())
            cgm_stats["below_70"] += (cgm_valid < 70).sum()
            cgm_stats["below_54"] += (cgm_valid < 54).sum()
            
        # Insulin Stats
        for col in ["insulin", "basal", "bolus"]:
            valid = df[col].dropna()
            if not valid.empty:
                insulin_stats[f"{col}_count"] += len(valid)
                insulin_stats[f"{col}_zero"] += (valid == 0).sum()
                insulin_stats["negative"] += (valid < 0).sum()

        # Insulin Semantics (insulin == basal + bolus)
        ins_check = df[["insulin", "basal", "bolus"]].dropna()
        if not ins_check.empty:
            insulin_equality_checks["total_checked"] += len(ins_check)
            matches = np.isclose(ins_check["insulin"], ins_check["basal"] + ins_check["bolus"], atol=1e-5).sum()
            insulin_equality_checks["exact_match"] += matches
            insulin_equality_checks["mismatch"] += len(ins_check) - matches

        # Carb Stats
        carb_valid = df["carbs"].dropna()
        if not carb_valid.empty:
            carb_stats["count"] += len(carb_valid)
            carb_stats["zero"] += (carb_valid == 0).sum()
            carb_stats["positive"] += (carb_valid > 0).sum()
            carb_stats["negative"] += (carb_valid < 0).sum()
            carb_stats["max"] = max(carb_stats["max"], carb_valid.max())

    print("\n--- PHYSIOLOGICAL CGM AUDIT ---")
    if cgm_stats["count"] > 0:
        mean_cgm = cgm_stats["sum"] / cgm_stats["count"]
        print(f"Total Valid CGM: {cgm_stats['count']}")
        print(f"Min: {cgm_stats['min']}, Max: {cgm_stats['max']}, Mean: {mean_cgm:.2f}")
        print(f"Below 70: {cgm_stats['below_70']} ({(cgm_stats['below_70']/cgm_stats['count'])*100:.2f}%)")
        print(f"Below 54: {cgm_stats['below_54']} ({(cgm_stats['below_54']/cgm_stats['count'])*100:.2f}%)")
    
    print("\n--- INSULIN SEMANTICS ---")
    print(f"Insulin entries: {insulin_stats['insulin_count']} (Zeros: {insulin_stats['insulin_zero']})")
    print(f"Basal entries: {insulin_stats['basal_count']} (Zeros: {insulin_stats['basal_zero']})")
    print(f"Bolus entries: {insulin_stats['bolus_count']} (Zeros: {insulin_stats['bolus_zero']})")
    print(f"Negative values found: {insulin_stats['negative']}")
    print(f"Insulin ≈ Basal + Bolus check:")
    print(f"  Total rows with all three: {insulin_equality_checks['total_checked']}")
    print(f"  Exact Matches: {insulin_equality_checks['exact_match']}")
    print(f"  Mismatches: {insulin_equality_checks['mismatch']}")
    if insulin_equality_checks['total_checked'] > 0 and insulin_equality_checks['exact_match'] == insulin_equality_checks['total_checked']:
        print("  => SEMANTICS: insulin = basal + bolus")
    else:
        print("  => SEMANTICS: insulin and basal/bolus might represent different concepts or scales")

    print("\n--- CARBOHYDRATE SEMANTICS ---")
    print(f"Carb entries: {carb_stats['count']}")
    print(f"Zero values: {carb_stats['zero']}")
    print(f"Positive values: {carb_stats['positive']}")
    print(f"Negative values: {carb_stats['negative']}")
    print(f"Max value: {carb_stats['max']}")
    if carb_stats['max'] > 1000:
        print("  => SEMANTICS: Suspect cumulative tracking due to very high max.")
    else:
        print("  => SEMANTICS: Likely individual meal event amounts.")

    print("\n--- TIMESTAMP ANOMALIES ---")
    print("Year Distribution:")
    for y in sorted(year_distribution.keys()):
        print(f"  {y}: {year_distribution[y]} rows")
    if max(year_distribution.keys()) > datetime.now().year:
        print("  => WARNING: Future dates detected. Likely a timezone artifact or synthetic shift in source dataset.")

    print("\n--- SUBJECT-LEVEL TEMPORAL COVERAGE ---")
    num_subj = len(subjects)
    durations = []
    
    gt_24h, gt_7d, gt_14d, gt_30d = 0, 0, 0, 0
    core_cohort_candidates = 0

    for s, data in subjects.items():
        if pd.notna(data["min_t"]) and pd.notna(data["max_t"]):
            dur = (data["max_t"] - data["min_t"]).total_seconds() / 3600.0  # in hours
            durations.append(dur)
            if dur >= 24: gt_24h += 1
            if dur >= 24 * 7: gt_7d += 1
            if dur >= 24 * 14: gt_14d += 1
            if dur >= 24 * 30: gt_30d += 1
            
            # Core cohort estimate: >14 days, has CGM, has Insulin, has Carbs
            if dur >= 24 * 14 and data["cgm_count"] > 1000 and data["insulin_count"] > 0 and data["carb_count"] > 0:
                core_cohort_candidates += 1

    durations.sort()
    print(f"Total Subjects: {num_subj}")
    if num_subj > 0:
        print(f"Minimum Duration: {durations[0]:.2f} hours")
        print(f"Median Duration: {np.median(durations) / 24:.2f} days")
        print(f"Maximum Duration: {durations[-1] / 24:.2f} days")
        print(f"Subjects >= 24h: {gt_24h}")
        print(f"Subjects >= 7 days: {gt_7d}")
        print(f"Subjects >= 14 days: {gt_14d}")
        print(f"Subjects >= 30 days: {gt_30d}")

    print("\n--- COHORT SUMMARY ---")
    print(f"All Subjects: {num_subj}")
    print(f"Core Cohort Candidates (>=14d, has CGM, Insulin, Carbs): {core_cohort_candidates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    qualify_dataset(args.input)
