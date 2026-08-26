# AGENTS.md — Rules for AI Agents working on AquaScan Open Science

This file provides directives for AI coding agents (e.g., Codebuff, Claude, Copilot) when contributing to this repository. **Violating these rules may produce scientifically invalid results.**

---

## 1. Dataset First

> Quality data precedes complex models.

Do not build advanced model architecture before the dataset specification, ingestion pipeline, quality control, and validation are solid. Prioritize data correctness over model sophistication.

## 2. No Invented Scientific Results

> Never fabricate results, metrics, figures, or performance claims.

Do not generate synthetic benchmark numbers, fill tables with plausible values, or create "placeholder" figures with fake data. Use `RESULTS PENDING` when data is unavailable.

## 3. Preserve RAW

> L0 RAW data is immutable.

Never overwrite, modify, or delete raw source data. Every transformation must create a new artifact with provenance links to the original. Checksums (SHA-256) must be preserved.

## 4. Preserve Provenance

> Every processed value must trace back to its RAW source.

All data products must carry:
- `source_record_id`
- `source_uri`
- `source_hash`
- `processing_step`
- `processing_version`
- `timestamp`
- `parent_artifact`

## 5. Prevent Data Leakage

> Never use naive random splits for geo/hydro data.

Must implement and use:
- `GroupKFold` by `site_id`
- Spatial separation (distance-based)
- Hydrogeological domain separation
- Full geographic holdout splits
- Well-level separation (same well never in train AND test)

**Add explicit tests verifying no leakage.**

## 6. Ground Truth Uncertainty Matters

> Ground truth is uncertain. Scores must capture this.

Never treat `water_found = True/False` as absolute truth. Maintain:
- Ground Truth score (numeric)
- Ground Truth class (A/B/C/D)
- Evidence quality score
- Scoring reasons

## 7. Negative Examples Matter

> Dry wells are as scientifically valuable as productive ones.

Never discard negative examples. They are essential for training and evaluation. The dataset must balance and report both classes.

## 8. Synthetic Data is NOT Field Validation

> Synthetic data can test code but cannot validate real-world performance.

Clearly label all synthetic results. Never present synthetic-only experiments as evidence of real-world capability. Include a prominent warning near any synthetic data.

## 9. GeoAI Production is Out of Scope

> This repository implements public science, not production AI.

The proprietary GeoAI pipeline, trained weights, AutoCal algorithms, and client-specific models live in private infrastructure. Create only interfaces or documented placeholders where needed for architectural clarity.

## 10. AutoCal Production is Out of Scope

> Calibration AI is proprietary.

Implement only deterministic, public-domain calibration methods. Create interfaces for the AutoCal concept but never implement the proprietary version.

## 11. Never Expose Private Ground Truth

> Client borehole data must never appear in this repository.

No private coordinates, well logs, owner names, or identifiable field data. The public repo contains only the schema, scoring code, and public-domain examples.

## 12. Do Not Implement Complexity Without Measurable Scientific Benefit

> Every abstraction, module, or dependency must justify itself.

Avoid over-engineering. A simple, correct, well-tested implementation is better than a complex one. Add complexity only when simpler baselines have been proven insufficient.

## 13. Prefer Reproducibility Over Cleverness

> A reproducible experiment beats a clever one.

Every experiment must be reproducible via:
- `experiment_id`
- `git_commit`
- `dataset_version`
- `schema_version`
- Configuration / random seed / hyperparameters

## 14. Every Scientific Claim Requires Evidence

> Assert nothing without a verifiable source.

Citations, data, reproducible experiments, or documented reasoning must back every claim in documentation, comments, and outputs.

---

## Summary Checklist

Before submitting any change, verify:

- [ ] No invented data or results
- [ ] RAW data preserved
- [ ] Provenance links intact
- [ ] No data leakage in splits (tested)
- [ ] Ground truth uncertainty captured
- [ ] Negative examples preserved
- [ ] No proprietary code exposed
- [ ] Synthetic data clearly labeled
- [ ] Tests pass
- [ ] Ruff linting passes
- [ ] Complexity justified