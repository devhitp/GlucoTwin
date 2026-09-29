# Digital Twin State and Initialization Contract

## Purpose
This contract defines the conceptual state vector maintained by the Digital Twin and the strict causal rules for its initialization.

## State Vector Architecture
The state vector is conceptually defined as: `[G(t), X(t), I(t), S_sleep(t)]`

### Variable Definitions
- **`G(t)` (Glucose State)**: Observable state representing current blood glucose concentration and local trajectory dynamics.
- **`X(t)` (Latent Action State)**: Latent state representing insulin/carbohydrate action on the metabolic system (e.g., active insulin on board, gut absorption).
- **`I(t)` (Insulin State)**: Observable/Latent state representing recent insulin exposure and pump deliveries.
- **`S_sleep(t)` (Context State)**: Observable/Latent state representing circadian or behavioral context (e.g., nocturnal, active).

### Disclaimers
These state variables are **NOT clinically validated**. They are engineering constructs representing the minimal required dynamics for a physiological simulation of glucose metabolism.

## Update Cadence
The state vector is updated at a target resolution of **5 minutes**, matching the nominal CGM sampling rate.

## Initialization Contract (Causality)
When a new patient's Digital Twin is instantiated, the state vector must be initialized using **strictly historical and causal data**.

1. **No Future Data**: Initialization cannot consume observations from time `t > T_init`.
2. **Burn-in Period**: The Twin requires a defined "burn-in" sequence of recent glucose, insulin, and meal exposure to stabilize `X(t)` and `I(t)`.
3. **Priors**: If patient-specific baseline history is unavailable at initialization time, the Twin must instantiate using population-level priors, gradually adapting as observable data streams in.
