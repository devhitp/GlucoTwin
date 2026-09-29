# Digital Twin Uncertainty Contract

## Purpose
The future Digital Twin must not output deterministic point-predictions without quantifying structural and epistemic uncertainty.

## Future Interface Representation
The prediction interface must be capable of generating and returning:
1. **Predicted Trajectory**: The central estimate (e.g., mean/median) of the physiological state.
2. **Uncertainty Interval**: The probabilistic bounds (e.g., 90% confidence interval) capturing prediction variance.
3. **Quality Metadata**: Embedded tracking of input completeness, tracking whether the prediction relies heavily on imputed missing data.
4. **Data Freshness**: A metric representing time elapsed since the last verifiable observation.

## Implementation Restrictions
- **No Fabricated Confidence**: The model must compute mathematically derived uncertainty bounds (e.g., via dropout, ensembles, or Bayesian inference). Do NOT invent arbitrary confidence heuristics.
- **No Premature Claims**: The repository documentation must not claim the existence of calibrated uncertainty or conformal prediction bounds until they are explicitly implemented and statistically evaluated on the held-out test cohort.
