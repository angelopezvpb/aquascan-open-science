# Changelog

All notable changes to AquaScan Open Science are documented in this file.

## [0.2.0] — 2026-11-07

### Added
- **National Enrichment Engine v1** (`aquascan national ...`): resumable,
  interruptible, crash-safe builder for the 139,697-record normalized snapshot
  - persistent SQLite state with endpoint-level granularity
  - durable response cache with atomic writes
  - exact resume, stale `RUNNING` recovery, graceful pause/stop
  - bounded retries, exponential backoff, rate limiting, circuit breaker
  - checkpoints, immutable partitions, disk-space monitor
  - 4,744-record regression canary (offline, no Phase 0 exposure)
  - `init | start | pause | stop | resume | status | retry-failed | verify | finalize`
- Public aggregate summary: `reports/national_enrichment_summary_v1.json`
- Public model card README for GeoAI Context v0
- Documentation: `docs/national-enrichment.md`

### Changed
- Public repository cleanup: removed internal notes, hardware-acceptance
  bookkeeping, superseded reports and duplicate test names.
- Corrected `CITATION.cff`, `pyproject.toml` URLs and author metadata.
- License finalized as Apache-2.0.

### Removed
- `AGENTS.md`, `LICENSE_REVIEW.md`, `ROADMAP.md` (canonical: `docs/roadmap.md`)
- `docs/ip-boundary.md`, `docs/open-science.md`, `docs/index.md`,
  `docs/data-governance.md`, `docs/huggingface-roadmap.md`
- Gate 5 / Gate 11 internal bookkeeping and superseded report versions

### Limitations (unchanged)
- No real ERT/seismic field validation.
- No dry-well detection (no confirmed negative Ground Truth).
- GeoAI Context v0 does not meaningfully outperform Baseline v2.
- `Región de los arenales` remains a documented OOD failure.

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
