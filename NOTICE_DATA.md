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

## What is NOT redistributed
This release does **not** include:
- the complete national IGME dump,
- the enriched Ground Truth dataset,
- API response caches,
- any sealed site information for the blinded field-validation cohort.

Only schemas, code, aggregated statistics, and clearly-labeled synthetic
fixtures are published.

## License status
This repository is released under the Apache-2.0 license (see `LICENSE`).
That license covers the code, schemas, documentation, and aggregated
statistics in this repository.

Original IGME source data is subject to IGME's own reuse terms, which are
independent of this repository's license. This release redistributes **no
original IGME records** — only code, schemas, aggregated statistics, and
clearly-labeled synthetic fixtures. Anyone redistributing original IGME data
must comply with IGME's terms directly.
