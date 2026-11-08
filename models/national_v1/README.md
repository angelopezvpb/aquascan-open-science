# AquaScan National Groundwater Depth Model v1

Research baseline that predicts **historical depth to water (metres)** from
documentary and contextual groundwater features across Spain. It is part of the
[AquaScan Open Science](https://github.com/angelopezvpb/aquascan-open-science)
project and is the **current canonical model**, superseding the five-domain
GeoAI Context v0 baseline.

> **This is not a groundwater detector.** It is a tabular regression baseline
> trained on historical documentary records. It has **not been validated in the
> field**, does **not** classify dry wells, and must not be used as the sole
> basis for drilling, investment or safety-critical decisions.

**Model card and fitted weights:**
[`angellopez/aquascan-open-science-model`](https://huggingface.co/angellopez/aquascan-open-science-model)

The five-domain baseline is preserved at the
[`v0-5domain` tag](https://huggingface.co/angellopez/aquascan-open-science-model/tree/v0-5domain).

The `.joblib` weights and preprocessor are published on Hugging Face, **not in
this Git repository**. This directory contains the input contract, policies,
aggregate metrics and provenance needed to audit the published artifacts.

## Model

| Item | Value |
|---|---|
| Task | Tabular regression: `TARGET_DEPTH_TO_WATER` (metres) |
| Model | `RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, random_state=42)` |
| Inputs | 21 documentary/contextual features |
| Training population | 16,722 READY depth records |
| Geographic scope | 44 of 48 provinces and 323 of 401 domains present in the 139,697-record national snapshot |
| Phase 0 firewall | Active — 10 sealed San Antonio sites excluded |
| License | Apache-2.0 (code and weights) |

The target is the median of dated observations
(`temporal_summary_json.median_depth_to_water_m`), not the current water level.
No target-derived column is a model input.

## Evaluation

All metrics use leakage-safe groups: the same 500 m proximity cluster, province
or domain is never split across train and test. Figures are medians across
folds, except the domain-holdout row (MAE/RMSE weighted by test records, R²
median across domains).

| Regime | RF MAE | Dummy (median) MAE | RF RMSE | RF R² |
|---|---:|---:|---:|---:|
| Grouped 5-fold CV (500 m proximity) | 11.52 m | 26.34 m | 23.28 m | 0.706 |
| Province holdout (5 folds; 44 provinces) | 16.33 m | 27.26 m | 31.30 m | 0.502 |
| Domain holdout (21 domains ≥150 records) | 12.82 m | 23.14 m | 25.23 m | 0.441 |

- RandomForest beats the median dummy in **20 of 21** scored domains.
- Split-conformal 90% interval: **±31.62 m** (width 63.25 m), empirical
  coverage **90.0%**.
- Out-of-fold MAE: **11.82 m**; out-of-fold R²: **0.712**.

The conformal interval is **marginal/global**, not a per-site guarantee.

## Support and limitations

- **Uneven support.** 122 of 323 domains have at least 30 training records;
  201 have 1–29. Only 21 domains have a dedicated holdout evaluation.
- **High-error domains.** The largest domain-holdout MAE is
  `Campo de Cartagena` (36.30 m), followed by `Madrid- Talavera` (26.91 m) and
  `Peñón- Montgó- Bernia` (25.67 m). Aggregate national metrics hide local
  failures.
- **No field validation.** No ERT/seismic/GNSS groundwater validation has been
  performed.
- **No water-presence classification.** The model predicts depth only and must
  not be used to claim that a well will be productive.
- **Distribution shift.** Performance outside Spain is not established.
- **Missing documentary data.** `construction_date` is missing for 51.3% of
  records; imputation is used.

## Files

| File | Description |
|---|---|
| `feature_manifest.json` | Exact 21-feature input contract |
| `confidence_policy.json` | GREEN/YELLOW/RED support triage and interval policy |
| `uncertainty_calibrator.json` | Split-conformal calibration |
| `ood_policy_reference.json` | Domain/province support policy |
| `metrics.json` | Aggregate evaluation metrics |
| `training_manifest.json` | Provenance, source hashes and hyperparameters |

Aggregate fold-level reports are in
[`reports/national_model_v1_cv.csv`](../../reports/national_model_v1_cv.csv)
and
[`reports/national_model_v1_domain_holdout.json`](../../reports/national_model_v1_domain_holdout.json).
Full documentation: [`docs/national-model.md`](../../docs/national-model.md).

## Data provenance

Trained on IGME-derived documentary records from the AquaScan national
enrichment (dataset SHA-256
`8c69621e05178d2f97ea01a27623c43e759271b7afae74c625ff4f2b33d18fbc`).
The **record-level dataset is not redistributed**. Only aggregate counts,
policies, provenance and fitted weights are published.

Code and weights are released under **Apache-2.0**. This does not grant rights
over the underlying IGME data.
