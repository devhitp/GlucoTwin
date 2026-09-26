import duckdb
import argparse

def analyze(file_path):
    print("====================================================")
    print("GLUCOTWIN — METABONET DATA QUALIFICATION (DUCKDB)")
    print("====================================================")
    
    con = duckdb.connect()
    
    print("\n--- CGM ---")
    cgm_stats = con.execute(f"""
        SELECT 
            count(CGM) as count,
            min(CGM) as min,
            max(CGM) as max,
            avg(CGM) as mean,
            sum(case when CGM < 70 then 1 else 0 end) as below_70,
            sum(case when CGM < 54 then 1 else 0 end) as below_54
        FROM '{file_path}'
        WHERE CGM IS NOT NULL
    """).fetchone()
    print(f"Count: {cgm_stats[0]}")
    print(f"Min: {cgm_stats[1]}, Max: {cgm_stats[2]}, Mean: {cgm_stats[3]:.2f}")
    if cgm_stats[0] > 0:
        print(f"Below 70: {cgm_stats[4]} ({(cgm_stats[4]/cgm_stats[0])*100:.2f}%)")
        print(f"Below 54: {cgm_stats[5]} ({(cgm_stats[5]/cgm_stats[0])*100:.2f}%)")

    print("\n--- INSULIN SEMANTICS ---")
    ins_stats = con.execute(f"""
        SELECT 
            count(insulin) as ins_count,
            sum(case when insulin = 0 then 1 else 0 end) as ins_zero,
            sum(case when insulin < 0 then 1 else 0 end) as ins_neg,
            
            count(basal) as bas_count,
            sum(case when basal = 0 then 1 else 0 end) as bas_zero,
            sum(case when basal < 0 then 1 else 0 end) as bas_neg,
            
            count(bolus) as bol_count,
            sum(case when bolus = 0 then 1 else 0 end) as bol_zero,
            sum(case when bolus < 0 then 1 else 0 end) as bol_neg
        FROM '{file_path}'
    """).fetchone()
    print(f"Insulin entries: {ins_stats[0]} (Zeros: {ins_stats[1]}, Negatives: {ins_stats[2]})")
    print(f"Basal entries: {ins_stats[3]} (Zeros: {ins_stats[4]}, Negatives: {ins_stats[5]})")
    print(f"Bolus entries: {ins_stats[6]} (Zeros: {ins_stats[7]}, Negatives: {ins_stats[8]})")
    
    ins_check = con.execute(f"""
        SELECT 
            count(*) as total_checked,
            sum(case when abs(insulin - (basal + bolus)) < 0.0001 then 1 else 0 end) as exact_match
        FROM '{file_path}'
        WHERE insulin IS NOT NULL AND basal IS NOT NULL AND bolus IS NOT NULL
    """).fetchone()
    print(f"Insulin = Basal + Bolus check:")
    print(f"  Total rows with all three: {ins_check[0]}")
    print(f"  Exact Matches: {ins_check[1]}")
    print(f"  Mismatches: {ins_check[0] - ins_check[1] if ins_check[0] else 0}")
    
    print("\n--- CARBOHYDRATE SEMANTICS ---")
    carb_stats = con.execute(f"""
        SELECT 
            count(carbs) as count,
            sum(case when carbs = 0 then 1 else 0 end) as zeros,
            sum(case when carbs > 0 then 1 else 0 end) as positive,
            sum(case when carbs < 0 then 1 else 0 end) as negatives,
            max(carbs) as max
        FROM '{file_path}'
        WHERE carbs IS NOT NULL
    """).fetchone()
    print(f"Carb entries: {carb_stats[0]}")
    print(f"Zero values: {carb_stats[1]}")
    print(f"Positive values: {carb_stats[2]}")
    print(f"Negative values: {carb_stats[3]}")
    print(f"Max value: {carb_stats[4]}")

    print("\n--- TIMESTAMP ANOMALIES ---")
    years = con.execute(f"""
        SELECT date_part('year', date) as y, count(*) 
        FROM '{file_path}' 
        WHERE date IS NOT NULL 
        GROUP BY y ORDER BY y
    """).fetchall()
    print("Year Distribution:")
    for y, c in years:
        print(f"  {int(y)}: {c} rows")

    print("\n--- SUBJECT-LEVEL TEMPORAL COVERAGE ---")
    subjs = con.execute(f"""
        SELECT 
            id,
            min(date) as min_t,
            max(date) as max_t,
            date_diff('hour', min(date), max(date)) as duration_hours,
            count(CGM) as cgm_count,
            count(insulin) as ins_count,
            count(carbs) as carb_count
        FROM '{file_path}'
        GROUP BY id
    """).df()
    
    subjs = subjs.dropna(subset=['duration_hours'])
    import numpy as np
    durs = subjs['duration_hours'].values
    
    gt_24h = (durs >= 24).sum()
    gt_7d = (durs >= 24*7).sum()
    gt_14d = (durs >= 24*14).sum()
    gt_30d = (durs >= 24*30).sum()
    
    print(f"Total Subjects: {len(subjs)}")
    print(f"Minimum Duration: {np.min(durs):.2f} hours")
    print(f"Median Duration: {np.median(durs)/24:.2f} days")
    print(f"Maximum Duration: {np.max(durs)/24:.2f} days")
    print(f"Subjects >= 24h: {gt_24h}")
    print(f"Subjects >= 7 days: {gt_7d}")
    print(f"Subjects >= 14 days: {gt_14d}")
    print(f"Subjects >= 30 days: {gt_30d}")
    
    core = ((durs >= 24*14) & (subjs['cgm_count'] > 1000) & (subjs['ins_count'] > 0) & (subjs['carb_count'] > 0)).sum()
    print("\n--- COHORT SUMMARY ---")
    print(f"Core Cohort Candidates (>=14d, has CGM, Insulin, Carbs): {core}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    analyze(args.input)
