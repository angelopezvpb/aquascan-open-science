# Public Roadmap

High-level milestones only. No sealed field-site identities are disclosed.

## Completed
- National data engineering (normalization, provenance, QA)
- Ground Truth methodology (evidence dimensions, target readiness)
- Validation dataset architecture (leakage-safe)
- Open-source leakage-free context baseline (Baseline v2)
- GeoAI Context v0 (documentary context model)
- Open-source physical processing stack (ERT + seismic, synthetic-validated)

## Next
1. **Hardware acquisition / integration** — bench acceptance, AutoCal fault
   injection, GNSS/power/EMC tests (Gate 11 remains NOT READY without hardware).
   See [hardware.md](hardware.md) for the BOM, data contracts and acceptance
   checklist.
2. **Non-cohort engineering dry run** — AQ-TEST-001 style acquisition at a
   non-cohort site to prove the real-data import path.
3. **Blinded retrospective physical validation** — ERT + seismic at
   documentary Ground Truth sites with strict blinding and prediction freeze.
4. **Prospective blind drilling validation** — predictions frozen before
   drilling reveals outcomes (this is the scientific objective of Phase 0+).

## Out of scope (never claimed in v0.2)
- Real groundwater detection accuracy from ERT/seismic (not yet measured)
- Dry-well detection (no confirmed negative Ground Truth exists)
- Commercial claims of any kind
