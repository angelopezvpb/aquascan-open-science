# Project Architecture

```
IGME / documentary context
        |
        v
   Context Model  (GeoAI Context v0)          [IMPLEMENTED]
        |
        +---------------------+
        |                     |
ERT RAW -> ERT Pipeline -> ERT Encoder        [IMPLEMENTED, not field-validated]
        |                     |
Seismic RAW -> Seismic Pipeline -> Seismic Encoder
        |                     |
        +----------+----------+
                   |
                 Fusion        (interface defined)  [FUTURE]
                   |
            Target predictions
                   |
        uncertainty / abstention
```

## Status legend
- **IMPLEMENTED** — code exists and is tested.
- **NOT_YET_PHYSICALLY_VALIDATED** — pipeline works on synthetic fixtures but
  has not been exercised on real ERT/seismic field data.
- **FUTURE** — interface/contract defined; no implementation published.

## Components

| Layer | Status |
|---|---|
| IGME ingestion / normalization | IMPLEMENTED |
| Ground Truth evidence methodology | IMPLEMENTED |
| Leakage-safe validation (grouped / spatial / cross-domain) | IMPLEMENTED |
| Open-source context baseline (Baseline v2) | IMPLEMENTED, frozen |
| GeoAI Context v0 | IMPLEMENTED, frozen |
| ERT pipeline + encoder | IMPLEMENTED, NOT_YET_PHYSICALLY_VALIDATED |
| Seismic pipeline + encoder | IMPLEMENTED, NOT_YET_PHYSICALLY_VALIDATED |
| Cross-modal registration | IMPLEMENTED (derived interpolation only) |
| AutoCal | INTERFACE ONLY — proprietary implementation out of scope |
| Multimodal fusion | FUTURE |
| Physical hardware stack | NOT YET ACQUIRED (no hardware) |
