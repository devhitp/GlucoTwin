# Digital Twin Personalization — Implementation Notes

## Implementation (Sprint 8)

Sprint 8 implements a **safe, minimal first-version** personalization layer.

### What is adapted
Only the patient's `baseline_glucose` is adapted — the single most causally unambiguous parameter.
It is computed as the mean of all valid historical glucose observations.

### What is NOT adapted (deferred to Sprint 9+)
- Insulin sensitivity (requires careful pharmacodynamic fitting)
- Meal absorption rate (requires matched meal+glucose pairs)
- Process noise (requires ensemble or Bayesian fitting)

### Causal guarantees
- Only observations occurring **before or at** the initialization timestamp are used.
- A minimum of 10 valid glucose observations is required. Below this threshold, population defaults are returned unchanged.
- The `is_personalized` flag in `TwinParameters` explicitly records whether adaptation occurred.

### Inference reproducibility
The same `PersonalizationEngine.adapt(history)` call during training can be reproduced identically at inference time given the same historical sequence. No external state is cached.

## Prohibited behaviors
- Adapting parameters using any observation from the evaluation horizon.
- Pre-computing patient profiles over the full chronological record, then injecting into the training window.
- Treating held-out test patient data as prior knowledge.
