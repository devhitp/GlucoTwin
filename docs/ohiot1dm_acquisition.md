# OhioT1DM Dataset Acquisition Guide

## Official Source
The legitimate OhioT1DM dataset is securely hosted by **PhysioNet** for authorized research use only.
- **Dataset Title**: OhioT1DM Dataset
- **Provider**: PhysioNet

## Access Requirements
The dataset contains sensitive physiological research data and is strictly protected. GlucoTwin strictly requires adherence to this access control and **will not** bypass it.

To obtain the data, you must:
1. Register for a PhysioNet account.
2. Complete the required credentialing/training for human research subjects if mandated.
3. Accept the Data Use Agreement specific to OhioT1DM.
4. Request access through the official portal.

## Local Placement
Once your request is approved and you have downloaded the authorized dataset:
1. Place the patient XML files directly into the `data/raw/` directory in this project.
2. DO NOT change the XML structures. 

The GlucoTwin pipeline will automatically discover them on the next run.

## Privacy & Version Control
**CRITICAL**: The `.gitignore` file strictly prohibits checking any raw or processed patient records into version control. NEVER commit the downloaded data, generated artifacts, or your personal PhysioNet access credentials to Git/GitHub. 

## Verification
After placing the files, you can verify detection by running:
```bash
python scripts/audit_real_dataset.py
```
If the audit succeeds, you may proceed with `scripts/run_real_pipeline.py`.
