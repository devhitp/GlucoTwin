# MetaboNet Timestamp Handling Contract

## Purpose
This contract defines how temporal data is allowed to be utilized within the GlucoTwin repository, accounting for the unverified provenance and observed anomalies in the MetaboNet dataset.

## Physiological Time vs Calendar Time
There is a strict boundary between:
- **A. Physiological time:** Time elapsed between events within a single patient's continuous data stream.
- **B. Calendar time:** The alignment of a timestamp to real-world global events (e.g., year 2026, holidays, daylight savings, day of week).

**GlucoTwin is ONLY allowed to utilize Physiological Time.**

## Core Rules
1. **Preserve original timestamps:** Do not alter, modify, or rewrite the original timestamp strings/objects parsed from the dataset.
2. **Never silently shift timestamps:** Do not attempt to "fix" dates extending into 2025–2027 by shifting them backwards.
3. **Never silently delete anomalous dates:** Rows with future dates must be retained and processed as normal, provided they are physiologically valid sequences.
4. **No calendar event interpretations:** Never claim the dates represent real-world calendar dates unless provenance is later confirmed.
5. **Elapsed-time calculations:** May use timestamp differences (e.g., `minutes_since_meal`) as long as the timestamps are structurally valid and monotonic.
6. **Date-derived features:** Features such as `hour_of_day`, `day_of_week`, `month`, and `is_weekend` must be clearly documented as containing unverified provenance assumptions (e.g., the device clock might not have been synchronized to local time).
7. **Validation behavior:** If a timestamp is malformed or non-monotonic inside a subject stream, apply the standard pipeline validation (e.g., flag or drop the specific row), but do not try to impute the correct calendar date.
