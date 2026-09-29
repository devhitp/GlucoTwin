# MetaboNet Provenance Record

**Dataset Name**: MetaboNet Public Dataset
**Local File**: `data/raw/metabonet_public.parquet`

## Dimensions
- **Total rows**: 154,842,077
- **Row groups**: 149
- **Columns**: 37
- **Unique subjects**: 1,291

## Units Documentation Status
* **CGM (glucose)**: UNKNOWN. There is no explicit schema metadata, column metadata, or locally available documentation confirming the unit (e.g. mg/dL vs mmol/L). Values appear numerically consistent with mg/dL but this is strictly an inference.
* **Insulin (basal, bolus)**: UNKNOWN.
* **Carbohydrates**: UNKNOWN.

## Unresolved Questions
* What is the authoritative source and definition of the features?
* How were the disparate subjects aggregated?
* Are subject IDs unique across the constituent datasets, or can the same physical patient appear multiple times?

## Important Notice
**CGM unit is unresolved; hypoglycemia thresholds are provisional and must not be interpreted clinically.** All models evaluated on this dataset using a 70 mg/dL assumption are for engineering validation of the Digital Twin infrastructure only.
