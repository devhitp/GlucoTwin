"""
Subject-level (patient-held-out) splitting for Sprint 7.5.

Ensures that entire subjects are assigned to either Train, Validation, or Test,
preventing any patient-level leakage across splits.
"""
import numpy as np
from typing import List, Tuple, Set

def get_subject_split(
    subjects: List[str],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Deterministically assigns subjects to train, val, and test sets.
    Returns sets of subject IDs for fast membership testing.
    """
    subjects_sorted = sorted(subjects)
    rng = np.random.default_rng(seed)
    
    # Shuffle for random assignment
    shuffled = rng.permutation(subjects_sorted).tolist()
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_subs = set(shuffled[:train_end])
    val_subs = set(shuffled[train_end:val_end])
    test_subs = set(shuffled[val_end:])
    
    return train_subs, val_subs, test_subs
