# Scientific Integrity

The principles below govern all research output in this repository. They are
non-negotiable: violating them produces scientifically invalid results.

## 1. No invented scientific results

Never fabricate results, metrics, figures, or performance claims. When data
is unavailable, say so explicitly (`RESULTS PENDING`) instead of filling
tables with plausible values or placeholder figures with fake data.

## 2. RAW data is immutable

L0 RAW data is never modified, overwritten, or deleted. Every transformation
creates a new artifact with provenance links to the original, and SHA-256
checksums are preserved.

## 3. Provenance

Every processed value traces back to its RAW source. Data products carry:

- `source_record_id`
- `source_uri`
- `source_hash`
- `processing_step`
- `processing_version`
- `timestamp`
- `parent_artifact`

## 4. No data leakage

Naive random splits are forbidden for geo/hydro data. Evaluation uses grouped
folds by site, spatial (distance-based) separation, hydrogeological domain
separation, full geographic holdouts, and well-level separation — the same
well never appears in both train and test. Tests verify zero leakage.

## 5. Ground truth uncertainty matters

Ground truth is uncertain. Scores capture this: a numeric Ground Truth score,
an evidence-quality class (A/B/C/D), an evidence-quality score, and scoring
reasons. `water_found = True/False` is never treated as absolute truth.

## 6. Negative examples matter

Dry or low-productivity wells are as scientifically valuable as productive
ones. Negative examples are never discarded; the dataset balances and reports
both classes.

## 7. Synthetic data is not field validation

Synthetic data can test code but cannot validate real-world performance.
Synthetic results are clearly labeled and are never presented as evidence of
real-world capability.

## 8. Reproducibility over cleverness

Every experiment is reproducible via `experiment_id`, `git_commit`,
`dataset_version`, `schema_version`, and recorded configuration / random
seed / hyperparameters. A reproducible experiment beats a clever one.

## 9. Every scientific claim requires evidence

No assertion is made without a verifiable source: citations, data,
reproducible experiments, or documented reasoning. Claims are backed by
evidence or marked as pending.

## 10. Simplicity with justification

Every abstraction, module, or dependency must justify itself. A simple,
correct, well-tested implementation is preferred over a complex one; add
complexity only when simpler baselines have been proven insufficient.

## 11. Ground truth is temporal

Water levels change over time. Historical measurements are never presented
as contemporaneous truth; dated observations and temporal summaries (first
date, last date, span, count, median/min/max/latest) are kept distinct.

## 12. Uncertainty and abstention

Predictions carry uncertainty and are allowed to abstain when evidence is
insufficient (`INSUFFICIENT_CONFIDENCE` / `RED` gate). Forcing a guess is
never better than a documented abstention.
