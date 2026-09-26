# MetaboNet Dataset Audit

## File Metadata (OBSERVED FROM LOCAL FILE)
- **File**: `metabonet_public.parquet`
- **Size**: 1294.04 MB
- **Total Rows**: 154,842,077
- **Row Groups**: 149
- **Columns**: 37

## Schema (OBSERVED FROM LOCAL FILE)
- `CGM`: DOUBLE
- `age`: DOUBLE
- `air_temp`: DOUBLE
- `basal`: DOUBLE
- `bolus`: DOUBLE
- `calories_burned`: DOUBLE
- `carbs`: DOUBLE
- `cgm_device`: BYTE_ARRAY (Logical: String)
- `misc_notes`: BYTE_ARRAY (Logical: String)
- `date`: INT64 (Logical: Timestamp)
- `galvanic_skin_response`: DOUBLE
- `heartrate`: DOUBLE
- `height`: DOUBLE
- `id`: BYTE_ARRAY (Logical: String)
- `insulin`: DOUBLE
- `insulin_delivery_algorithm`: BYTE_ARRAY (Logical: String)
- `insulin_delivery_device`: BYTE_ARRAY (Logical: String)
- `insulin_delivery_modality`: BYTE_ARRAY (Logical: String)
- `insulin_type_basal`: BYTE_ARRAY (Logical: String)
- `insulin_type_bolus`: BYTE_ARRAY (Logical: String)
- `is_pregnant`: BOOLEAN
- `is_test`: BOOLEAN
- `meal_label`: BYTE_ARRAY (Logical: String)
- `skin_temp`: DOUBLE
- `source_file`: BYTE_ARRAY (Logical: String)
- `steps`: DOUBLE
- `weight`: DOUBLE
- `workout_duration`: DOUBLE
- `workout_intensity`: DOUBLE
- `workout_label`: BYTE_ARRAY (Logical: String)
- `treatment_group`: BYTE_ARRAY (Logical: String)
- `randomization_date`: INT64 (Logical: Timestamp)
- `extension_date`: INT64 (Logical: Timestamp)
- `gender`: BYTE_ARRAY (Logical: String)
- `age_of_diagnosis`: DOUBLE
- `ethnicity`: BYTE_ARRAY (Logical: String)
- `subject_split_across_traintest`: BOOLEAN

## Subjects (OBSERVED FROM LOCAL FILE)
- **Unique Subjects**: 1,291
- **Format**: Numeric String (e.g., '596')

## Time Coverage (OBSERVED FROM LOCAL FILE)
- **Earliest Timestamp**: 2011-11-17 21:00:00
- **Latest Timestamp**: 2027-07-14 15:30:00
- **Overall Duration**: ~5717 days

## Sampling Granularity (OBSERVED FROM LOCAL FILE)
- The dataset is pre-aligned into fixed 5-minute intervals (e.g. `09:20:00`, `09:25:00`, `09:30:00`), which is structurally identical to the GlucoTwin pipeline requirements.

## Missingness (OBSERVED FROM LOCAL FILE)
- `CGM`: 15.48% nulls
- `date`: 0.00% nulls
- `id`: 0.00% nulls
- `insulin`: 9.20% nulls
- `basal`: 11.93% nulls
- `bolus`: 11.15% nulls
- `carbs`: 33.21% nulls
- `steps`: 99.38% nulls
- `heartrate`: 99.33% nulls
- `galvanic_skin_response`: 99.90% nulls
- `skin_temp`: 99.90% nulls

## Units (INFERENCE)
- **Units**: UNIT NOT CONFIRMED. The columns `CGM`, `basal`, `bolus`, `carbs` contain numeric values but no explicit units are designated in the schema.

## GlucoTwin Compatibility
| GlucoTwin Requirement | Availability | Confidence |
| --- | --- | --- |
| Subject ID | SUPPORTED | High |
| Timestamp | SUPPORTED | High |
| CGM | SUPPORTED | High |
| Insulin | SUPPORTED | High |
| Meals/carbs | SUPPORTED | High |
| Activity | PARTIALLY SUPPORTED (Steps exist, but 99% null) | Low |
| Heart rate | PARTIALLY SUPPORTED (99% null) | Low |
| EDA (Skin response) | PARTIALLY SUPPORTED (99% null) | Low |
| Skin temperature | PARTIALLY SUPPORTED (99% null) | Low |
| Sleep/context | NOT SUPPORTED | High |

### Critical GlucoTwin Needs
1. **Personalized glucose baseline**: SUPPORTED. Subject ID and history are available.
2. **30-minute labels**: SUPPORTED. 5-minute sampling allows forecasting.
3. **60-minute labels**: SUPPORTED.
4. **Insulin-response features**: SUPPORTED. Basal/Bolus are present.
5. **Meal/carbohydrate features**: SUPPORTED. Carbs are present.
6. **Activity/context features**: PARTIALLY SUPPORTED. Too sparse.
7. **Patient-specific temporal modeling**: SUPPORTED.
8. **Nocturnal hypoglycemia**: SUPPORTED. Timestamp enables extracting night windows.
9. **Digital Twin state estimation**: SUPPORTED.
