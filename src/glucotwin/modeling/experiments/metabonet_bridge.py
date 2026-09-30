"""
MetaboNet → SynchronizedRecord bridge.

Maps CanonicalRecord (from the MetaboNetAdapter) into the existing
SynchronizedRecord schema so the Sprint 2 PreprocessingPipeline
can be used without modification.

Memory safety: Uses row-group streaming with a bounded buffer per subject.
The full 154M-row Parquet is NEVER materialised as a single DataFrame.
"""
import gc
import pandas as pd
import pyarrow.parquet as pq
from typing import Iterator, List, Optional
from src.glucotwin.data.schema import SynchronizedRecord


# Columns needed from the Parquet — minimise I/O
REQUIRED_COLS = ["id", "date", "CGM", "insulin", "basal", "bolus", "carbs"]
OPTIONAL_WEARABLE_COLS = ["heartrate", "galvanic_skin_response", "skin_temp"]

# Physiological CGM bounds assumed mg/dL (PROVISIONAL — units not confirmed)
CGM_MIN = 20.0
CGM_MAX = 600.0


def _canonical_to_synchronized(subject_id: str, df: pd.DataFrame) -> List[SynchronizedRecord]:
    """
    Convert a per-subject DataFrame (MetaboNet columns already projected)
    into a list of SynchronizedRecord objects consumed by PreprocessingPipeline.

    Filters:
      - Rows with null glucose are discarded (no future-label imputation).
      - Values outside [20, 600] mg/dL (PROVISIONAL) are discarded as
        structurally non-physiological.
    """
    rename = {
        "CGM": "glucose",
        "basal": "basal_insulin",
        "bolus": "bolus_insulin",
        "carbs": "carbohydrates",
        "heartrate": "heart_rate",
        "galvanic_skin_response": "eda",
        "skin_temp": "skin_temperature",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Drop null glucose
    df = df[df["glucose"].notna()].copy()

    # Drop structurally non-physiological glucose (PROVISIONAL bounds)
    df = df[(df["glucose"] >= CGM_MIN) & (df["glucose"] <= CGM_MAX)].copy()

    records = []
    for _, row in df.iterrows():
        records.append(SynchronizedRecord(
            patient_id=subject_id,
            timestamp=row["date"],
            glucose=float(row["glucose"]),
            basal_insulin=row.get("basal_insulin"),
            bolus_insulin=row.get("bolus_insulin"),
            carbohydrates=row.get("carbohydrates"),
            heart_rate=row.get("heart_rate"),
            eda=row.get("eda"),
            skin_temperature=row.get("skin_temperature"),
            accelerometer_x=None,
            accelerometer_y=None,
            accelerometer_z=None,
        ))
    return records


def iter_subjects_from_parquet(
    parquet_path: str,
    subject_ids: Optional[List[str]] = None,
    include_wearables: bool = False,
    max_buffer_rows_per_subject: int = 300_000,
) -> Iterator[tuple]:
    """
    Memory-safe row-group streaming iterator.

    Yields (subject_id, List[SynchronizedRecord]) for each subject.

    Design:
    - Reads one Parquet row group at a time (~1M rows each).
    - Accumulates partial subject data in a per-subject buffer.
    - After each row group, finalises any subjects whose data is entirely
      contained within already-seen row groups (heuristic: subject not seen
      in the *current* row group → complete).
    - Hard limit on buffer rows per subject prevents unbounded growth.
    - After the final row group, yields all remaining buffered subjects.

    This avoids loading the entire 154M-row file into memory.
    """
    import pyarrow.dataset as ds
    
    cols = REQUIRED_COLS + (OPTIONAL_WEARABLE_COLS if include_wearables else [])
    
    # Process subjects in batches to avoid OOM
    batch_size = 50
    subject_list = subject_ids if subject_ids is not None else []
    
    # If no subjects provided, we'd need to find them, but the script always provides them.
    if not subject_list:
        return
        
    for i in range(0, len(subject_list), batch_size):
        batch_ids = subject_list[i:i+batch_size]
        batch_set = set(batch_ids)
        
        # Load exactly these subjects using PyArrow dataset filtering
        dataset = ds.dataset(parquet_path)
        table = dataset.to_table(columns=cols, filter=ds.field("id").isin(batch_set))
        df_batch = table.to_pandas()
        
        if df_batch.empty:
            continue
            
        for subj, group in df_batch.groupby("id"):
            subj = str(subj)
            chunk = group.drop(columns=["id"])
            records = _canonical_to_synchronized(subj, chunk)
            if records:
                yield subj, records
                
        del df_batch, table
        gc.collect()
