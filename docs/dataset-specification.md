# Dataset Specification

## AquaScan Canonical Dataset

The AquaScan dataset is organized in processing levels (L0-L4),
each adding value while preserving provenance to the original data.

### L0 — RAW

- **Status**: Immutable
- **Content**: Original data as received from the source
- **Metadata**: SHA-256 checksum, download timestamp, source URI, instrument version
- **Format**: Native source format (JSON, CSV, SEG-2, etc.)
- **Never**: Modified, cleaned, or overwritten

### L1 — CALIBRATED / QC

- **Status**: Quality-controlled, normalized
- **Content**: RAW data with field names normalized, units standardized, quality flags
- **Processing**: Deterministic only — no proprietary calibration AI
- **Additions**: Quality scores, duplicate flags, provenance records

### L2 — GEOPHYSICS

- **Status**: Derived geophysical products
- **Content**: Apparent resistivity, P-wave velocity, anomaly maps
- **Processing**: Public-domain geophysical computations
- **Additions**: Resistivity statistics, Vp estimates, uncertainty ranges

### L3 — ML READY

- **Status**: Feature vectors for machine learning
- **Content**: Engineered features, one-hot encoded categories
- **Processing**: Feature engineering, imputation
- **Additions**: Derived ratios, interaction features

### L4 — GROUND TRUTH

- **Status**: Independent verification data
- **Content**: Borehole logs, pump test results, water quality data
- **Source**: Borehole drilling results, verified field measurements
- **Classification**: A (strong), B (good), C (weak), D (unsuitable)

## Key Schema Fields

See `schemas/aquascan_dataset.schema.json` for the complete JSON Schema.

### Required Fields
- `campaign_id` — unique campaign identifier
- `site_id` — site identifier
- `schema_version` — schema version

### Critical Fields
- `latitude`, `longitude` — WGS84 coordinates
- `total_depth` — drilled depth (m)
- `ground_truth_score` — 0-1 quality score
- `ground_truth_class` — A/B/C/D

## null ≠ 0

Unknown values must be `null`. Never convert unknown to zero.
A flow rate of 0 L/s is very different from "flow rate unknown."