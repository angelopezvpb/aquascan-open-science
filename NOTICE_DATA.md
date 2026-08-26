# Data Notice — IGME Source Attribution

## Source data
This repository processes data from the **Spanish Geological Survey (IGME)**
public water-point inventory (Inventario de Puntos de Agua / bdaguas API).

- **Original data owner:** Instituto Geológico y Minero de España (IGME)
- **Access:** public ArcGIS REST service and bdaguas API
- **What AquaScan adds:** normalization, schemas, QA flags, provenance,
  Ground Truth methodology, aggregated statistics, and open-source code.

## Attribution and independence
- AquaScan is **independent** of IGME.
- IGME does **not** sponsor or endorse AquaScan.
- AquaScan does not claim ownership of original IGME data.
- Original field names and units are preserved wherever possible; derived
  values are explicitly marked `DERIVED`.

## What is NOT redistributed in v0.1
This release does **not** include:
- the complete national IGME dump,
- the 4,744-record enriched Ground Truth dataset,
- API response caches,
- any Phase 0 sealed site information.

Only schemas, code, aggregated statistics, and clearly-labeled synthetic
fixtures are published.

## License status
The exact IGME reuse/license terms for the source data could not be fully
verified within this repository.

```json
{"DATA_LICENSE_REVIEW_REQUIRED": true}
```

Human review of the current IGME data terms is required before any data
redistribution beyond code and aggregated statistics.
