# GlucoTwin Architecture

## Data Pipeline (Sprint 1)

```mermaid
flowchart TD
    A[OhioT1DM Dataset] --> B[Dataset Loader]
    B --> C[Parser]
    C --> D[Schema]
    D --> E[Validator]
    E --> F[Timestamp Normalization]
    F --> G[Synchronizer]
    G -.-> H[Future Processing / ML / Digital Twin Layers]
    
    style H stroke-dasharray: 5 5,fill:#f9f9f9,stroke:#666
```

- **Dataset Loader**: Locates dataset files and loads raw formats.
- **Parser**: Converts raw representations into internal structured records.
- **Schema**: Typed internal data models (CGM, Insulin, Meals, Wearables).
- **Validator**: Asserts data integrity, detecting missing values, chronological issues, and patient mix-ups.
- **Timestamp Normalization**: Ensures consistent timestamp tracking and timezone alignment.
- **Synchronizer**: Aligns multiple asynchronous streams (CGM, insulin, etc.) into a cohesive timeline.
- **Future Layers**: NOT IMPLEMENTED YET. (Machine Learning, What-If simulation, Digital Twin).
