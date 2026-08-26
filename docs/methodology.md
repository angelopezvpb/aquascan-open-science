# Open-Science Methodology

## Immutable RAW philosophy
- L0 RAW data is never modified.
- Every transformation creates a new artifact with provenance links
  (source record id, source URI, SHA-256, processing step/version, timestamp,
  parent artifact).

## Ground Truth evidence dimensions
Ground Truth is maintained as independent evidence dimensions:
- GEOLOGICAL_EVIDENCE (lithology intervals, aquifer flags)
- WATER_LEVEL_EVIDENCE (dated piezometry: Nivel / Valor)
- HYDRAULIC_EVIDENCE (pumping tests)
- WATER_QUALITY_EVIDENCE (chemistry)
- DOCUMENTARY_CONFIDENCE

## Independent targets
Readiness is computed **separately** for each target:
TARGET_WATER_PRESENCE, TARGET_DEPTH_TO_WATER,
TARGET_HYDRAULIC_HEAD_ELEVATION, TARGET_AQUIFER_INTERVAL,
TARGET_PRODUCTIVITY, TARGET_LITHOLOGY.
A record can be ready for one target and not another. There is no single
global `good_well` label.

## Temporal Ground Truth
- Piezometric level varies through time.
- Every water-level observation preserves: date, depth to water, hydraulic
  head elevation, source.
- Derived summaries (min/median/max/span) never replace the time series.
- `temporal_relevance_status` (RECENT / HISTORICAL / MULTI_DECADE_SERIES /
  DATE_UNKNOWN) prevents presenting historical levels as current.
- Historical IGME measurements are **not** contemporaneous Ground Truth.

## Spatial leakage prevention
- Naive random splits are forbidden for geo/hydro data.
- Grouped folds by site/proximity/domain, spatial-block holdout, and
  cross-domain holdout are enforced programmatically.
- Tests verify zero group overlap between train and test.

## Uncertainty and abstention
- Distribution-free conformal-style intervals are used where suitable.
- Confidence gates (GREEN/YELLOW/RED) allow abstention; thresholds are fit on
  validation data only, never on final test outcomes.
- OOD/support assessment is pre-prediction and independent of target error.

## Blinding and prediction freeze
- Phase 0 sites are blinded (AQ-P0-xxx); sealed mapping is separate.
- Predictions are frozen (with hashes) before Ground Truth reveal.
- Reveal is a logged operation; predictions are immutable after freeze.
