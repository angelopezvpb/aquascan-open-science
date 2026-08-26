# Validation Protocol

## Scientific Validation of Water Prospecting Predictions

AquaScan predictions must be validated against **independent** Ground Truth
data that was not used during model training or feature engineering.

## Split Protocol

### Mandatory Rules
1. **Never use random row-level splits** for geo/hydro data
2. **Same well/site must not appear in both train and test**
3. **Use geographic/distance-based holdout** for realistic evaluation
4. **Report leakage check results** for every experiment

### Split Types

#### Group K-Fold by Site/Well
Ensures all records from the same well stay together.

#### Geographic Holdout
Trains on known areas, tests on geographically distant areas.
Simulates real prospecting: "We know region A, can we predict region B?"

#### Spatial Distance Split
Minimum distance enforced between any train and test point.
The strictest test of generalization.

#### Geological Domain Split
Train and test on different geological domains.
Tests cross-domain generalization.

## Metrics

### Classification (water presence)
- Precision, Recall, F1
- ROC-AUC
- PR-AUC (preferred for imbalanced data)

### Calibration
- Brier Score
- Expected Calibration Error (ECE)
- Calibration curve

### Depth Estimation
- MAE (meters)
- RMSE (meters)
- MAPE (%)

### Abstention Analysis
- Coverage vs. Error curves
- Selective risk at different evidence thresholds

## Pre-registration

Experiments should be pre-registered:
- Dataset version
- Split strategy and random seed
- Model and hyperparameters
- Evaluation metrics

The `aquascan experiment fingerprint` command generates a reproducible ID.