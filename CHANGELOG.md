# Changelog

All notable changes to AquaScan Open Science are documented in this file.

## [0.1.0] — Initial open-science research release candidate

### Added
- IGME public source ingestion and normalization
- RAW data immutability with SHA-256 provenance
- Ground Truth evidence methodology (A/B/C/D, target-specific readiness)
- Candidate Priority Engine (documented heuristic)
- Leakage-safe validation: spatial grouping, grouped / spatial-block / cross-domain evaluation
- Open-source context baseline (Baseline v2, leakage-free)
- GeoAI Context v0 (open-source tabular context model)
- Open-source physical encoder stack for ERT + P-wave seismic (canonical schemas, pipelines, encoders, registration, synthetic fixtures)
- Blinded-validation protocols (Gate 6.1.x / 7) for future field experiments
- Comprehensive tests and CI

### Explicit limitations (not validated in v0.1)
- No real ERT/seismic field validation has been performed
- No dry-well detection has been validated (no confirmed negative Ground Truth)
- Physical hardware acceptance: NOT YET PERFORMED (Gate 11 status)
- Región de los arenales is a documented out-of-distribution generalization failure
- GeoAI Context v0 does not meaningfully outperform Baseline v2

### Historical note
Gate 8 v1 contained target leakage (`_target_*` columns as inputs) and is marked
`INVALID_FOR_SCIENTIFIC_COMPARISON`; see  `docs/baseline-leakage-postmortem.md`.
All results in this release are the leakage-free Baseline v2 / forensic numbers.
