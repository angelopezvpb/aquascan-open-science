# National Depth-to-Water Model v1

Research baseline that predicts **historical depth to water (metres)** from
documentary and contextual groundwater features across Spain. It is the
**current canonical model** and supersedes the five-domain
[GeoAI Context v0](current-results.md) baseline, which is preserved at the
[`v0-5domain` tag](https://huggingface.co/angellopez/aquascan-open-science-model/tree/v0-5domain).

> **Scope:** tabular regression on historical documentary records. The model
> does **not** detect groundwater, does **not** classify dry wells, and has
> **not** been validated in the field.

- **Model card and weights:**
  [`angellopez/aquascan-open-science-model`](https://huggingface.co/angellopez/aquascan-open-science-model)
- **Five-domain baseline:**
  [`v0-5domain` tag](https://huggingface.co/angellopez/aquascan-open-science-model/tree/v0-5domain)
- **Weights in Git:** not distributed. GitHub contains code, policies and
  aggregate metrics only.
- **Record-level data:** not redistributed.

## Model

| Item | Value |
|---|---|
| Experiment ID | `aquascan-national-depth-v1` |
| Task | Tabular regression: `TARGET_DEPTH_TO_WATER` (metres) |
| Estimator | `RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, random_state=42)` |
| Preprocessing | Median imputation + `StandardScaler` (numeric); most-frequent imputation + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` (categorical) |
| Inputs | 21 documentary/contextual features |
| Training population | 16,722 READY depth records |
| Provinces / domains | 44 / 323 |
| Proximity groups | 10,776 |
| Target | Median of dated observations (`temporal_summary_json.median_depth_to_water_m`) |
| License | Apache-2.0 |

The national snapshot contains 139,697 records, 48 provinces and 401 domains.
Requiring a READY depth-to-water target leaves 16,732 records. The Phase 0
firewall then excludes the **10 sealed San Antonio sites** (all 10 were READY),
giving the final 16,722-record modeling population. The sealed records are
excluded before proximity grouping, evaluation and training.

`group_proximity_500m` and `group_spatial_cell` are **not model inputs**. They
exist only to define leakage-safe evaluation folds. `igme_id` is audit-only.

## Evaluation

The same 500 m proximity cluster, province or domain is never split across
train and test. Figures are medians across folds, except the domain-holdout row
(MAE/RMSE weighted by test records, R² median across domains).

| Regime | RF MAE | Dummy (median) MAE | RF RMSE | RF R² |
|---|---:|---:|---:|---:|
| Grouped 5-fold CV (500 m proximity) | 11.52 m | 26.34 m | 23.28 m | 0.706 |
| Province holdout (5 folds; 44 provinces) | 16.33 m | 27.26 m | 31.30 m | 0.502 |
| Domain holdout (21 domains ≥150 records) | 12.82 m | 23.14 m | 25.23 m | 0.441 |

- RandomForest beats the median dummy in **20 of 21** scored domains. The
  exception is `Vega Media y Baja del Segura`.
- Split-conformal absolute-residual calibration at `alpha = 0.1` gives a
  **±31.62 m** 90% interval (width 63.25 m) with **90.0%** empirical coverage
  on the out-of-fold predictions.
- Out-of-fold MAE: **11.82 m**; out-of-fold R²: **0.712**.

The interval is **marginal/global**, not per-site. The confidence policy adds a
GREEN/YELLOW/RED support triage based on domain/province training support and
training-range extrapolation; RED means **abstain**. It is not a calibrated
probability.

### Highest domain-holdout errors

| Domain | n | RF MAE | Dummy MAE |
|---|---:|---:|---:|
| `Campo de Cartagena` | 243 | 36.30 m | 63.68 m |
| `Madrid- Talavera` | 174 | 26.91 m | 34.70 m |
| `Peñón- Montgó- Bernia` | 164 | 25.67 m | 58.95 m |
| `Campo de Dalías` | 972 | 23.98 m | 48.36 m |
| `Cresta del Gallo` | 150 | 20.61 m | 52.10 m |

These are aggregate domain results, not site-specific guarantees.

## Support

| Support class | Domains |
|---|---:|
| `IN_DISTRIBUTION` (≥30 training records) | 122 |
| `WEAK_SUPPORT` (1–29) | 201 |
| `OUT_OF_DISTRIBUTION` (0) | 0 |

Only the 21 domains with at least 150 records have a dedicated domain-holdout
evaluation. A national model in coverage is **not** a uniformly validated model
for every province or domain.

## Limitations

- **No field validation.** No ERT/seismic/GNSS groundwater validation has been
  performed.
- **Historical target.** The target is the median of dated observations; it is
  not the current water level.
- **No water-presence or dry-well classification.** The model predicts depth
  only and must not be used to claim that a well will be productive.
- **Uneven support.** 201 of 323 domains have fewer than 30 training records.
- **High-error domains.** Some supported domains have large holdout errors, so
  aggregate national metrics hide local failures.
- **Distribution shift.** Performance outside Spain is not established.
- **Marginal uncertainty.** The conformal interval is global and wide; it does
  not model site-specific uncertainty.
- **No physical inputs.** The model uses documentary/contextual features only.
- **Missing documentary data.** `construction_date` is missing for 51.3% of
  records; imputation is used.
- **Synthetic inputs are not field validation.** The example row in the model
  repository only demonstrates the API.

## Reproduction

The record-level dataset and the Phase 0 firewall list are **not** distributed
in this repository. With the private national snapshot and firewall file in
place, the full protocol is:

```bash
python scripts/train_national_model.py --evaluate --force
python scripts/train_national_model.py --reuse-evaluation --force
python -m pytest tests/test_national_model_no_leakage.py -q
python scripts/publish_national_model_hf.py
```

`--evaluate` runs grouped 5-fold CV by 500 m proximity, 5-fold province
holdout, and leave-one-domain-out for domains with at least 150 records.
`--reuse-evaluation` verifies the cached out-of-fold predictions against the
modeling frame before regenerating artifacts. The test module skips
automatically when private data or fitted weights are absent.

## Provenance

| Item | Value |
|---|---|
| Dataset SHA-256 | `8c69621e05178d2f97ea01a27623c43e759271b7afae74c625ff4f2b33d18fbc` |
| Snapshot records | 139,697 |
| READY depth records before firewall | 16,732 |
| Phase 0 READY records excluded | 10 |
| Modeling records | 16,722 |
| Random seed | 42 |
| Code | `src/aquascan/modeling/national_depth.py` |
| Entry point | `scripts/train_national_model.py` |
| Tests | `tests/test_national_model_no_leakage.py` |
| Aggregate reports | `reports/national_model_v1_cv.csv`, `reports/national_model_v1_domain_holdout.json` |
| Metadata | `models/national_v1/` |

The working tree used for the published artifacts is not a Git repository;
`models/national_v1/training_manifest.json` records SHA-256 hashes of the exact
source and report files instead of a commit hash. The model card on Hugging
Face carries the full artifact checksums (`SHA256SUMS`).

## License

Code and weights are released under **Apache-2.0**. This does not grant any
rights over the underlying IGME data.
