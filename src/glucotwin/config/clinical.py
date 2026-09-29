"""
Clinical Threshold Configuration Contract

This module centralized all physiological and clinical threshold assumptions.
"""

# MetaboNet dataset units are strictly unconfirmed.
GLUCOSE_UNIT = "unknown"

# 70 is used as a research/engineering assumption for pipeline validation.
HYPO_THRESHOLD = 70.0

# Explicit designation that these values cannot be interpreted clinically.
THRESHOLD_STATUS = "provisional"

def get_hypo_threshold() -> float:
    """Returns the provisional hypoglycemia threshold for engineering evaluation."""
    return HYPO_THRESHOLD

def get_glucose_unit() -> str:
    """Returns the unit status of the current glucose dataset."""
    return GLUCOSE_UNIT
