# Data Governance

## Data Management Principles

### 1. RAW Data Immutability
L0 data is **never** modified. Transformations create new artifacts with
provenance links to the original. If L0 data is found to contain errors,
the fix is recorded as a new processing step, not a modification to RAW.

### 2. Provenance Tracking
Every data product carries:
- Source record ID
- Source URI
- SHA-256 hash of source
- Processing step name and version
- Timestamp
- Parent artifact reference

### 3. Data Lineage
The provenance chain answers: "From which RAW data did this value originate?"
At any level (L1-L4), you can trace back through the chain to L0.

### 4. Data Versioning
Datasets are versioned independently from code:
- `dataset_version` tracks the dataset used
- `schema_version` tracks the schema
- `git_commit` tracks the processing code

### 5. Security
- No API keys, tokens, or passwords in the repository
- `.env` excluded from version control
- Private coordinates never stored in this repo
- Large geophysical files excluded from Git

### 6. Data Retention
- RAW data: kept indefinitely
- Intermediate data: can be regenerated from RAW
- Processed datasets: versioned with the code that produced them

## Data Directory Structure

```
data/
├── raw/         # L0 — Never modified
├── interim/     # L1/L2 — Intermediate processing
├── processed/   # L3/L4 — Final datasets
└── external/    # Third-party reference data
```

## Public vs. Private

| Data Type | Public (this repo) | Private |
|-----------|-------------------|---------|
| IGME records | ✓ | — |
| IGME schema | ✓ | — |
| Client borehole logs | — | ✓ |
| Production weights | — | ✓ |
| AquaScan-25 dataset | ✓ (HuggingFace) | — |
| AutoCal models | — | ✓ |