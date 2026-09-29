import pytest
import pandas as pd
import numpy as np

from src.glucotwin.modeling.experiments.subject_split import get_subject_split

def test_deterministic_subject_splits():
    subjects = [f"sub_{i}" for i in range(100)]
    t1, v1, te1 = get_subject_split(subjects, seed=42)
    t2, v2, te2 = get_subject_split(subjects, seed=42)
    t3, v3, te3 = get_subject_split(subjects, seed=99)
    
    assert t1 == t2
    assert v1 == v2
    assert te1 == te2
    
    assert t1 != t3

def test_no_subject_overlap():
    subjects = [f"sub_{i}" for i in range(100)]
    t, v, te = get_subject_split(subjects, seed=42)
    
    assert len(t.intersection(v)) == 0, "Train and Val overlap"
    assert len(t.intersection(te)) == 0, "Train and Test overlap"
    assert len(v.intersection(te)) == 0, "Val and Test overlap"
    assert len(t) + len(v) + len(te) == 100
