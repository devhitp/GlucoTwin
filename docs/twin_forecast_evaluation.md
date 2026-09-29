# Twin Forecast Evaluation

> Twin forecast vs observed glucose is performed ONLY for evaluation.
> Future observations NEVER enter feature construction or model training.

## Methodology
At each held-out test window T:
1. The Twin generates a simulated trajectory from T.
2. Actual future glucose observations at T+30 and T+60 are collected **as evaluation targets only**.
3. Error metrics are computed between simulated trajectory and actual future observations.

## Metrics

| Metric | Definition |
|--------|-----------|
| MAE | Mean Absolute Error (simulated vs observed) |
| RMSE | Root Mean Squared Error |
| Median Absolute Error | Robust error measure |
| Mean Bias | Mean(simulated - observed) — positive = over-prediction |

## Important Caveats
- CGM units are unconfirmed. Metrics are expressed on the dataset's numerical glucose scale.
- The Twin is a physiological-inspired engineering model, not a validated physiological simulator.
- These metrics measure the Twin's trajectory accuracy, which informs the quality of Twin-derived features but does not directly determine risk prediction performance.

## Synthetic Pilot Results (Sprint 9)

| Horizon | N valid | MAE | RMSE | Mean Bias |
|---------|---------|-----|------|-----------|
| 30m | 50 | 7.77 | 8.93 | +0.47 |
| 60m | 50 | 9.10 | 10.78 | -3.25 |

> NOTE: These results are from synthetic data only. They do not represent real patient glucose prediction accuracy.
