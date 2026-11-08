

<p align="center">
  <img src="docs/assets/aquascan-banner.png" alt="AquaScan Open Science" width="100%">
</p>

# AquaScan Open Science

Open-source research framework for investigating whether **multimodal
hydrogeophysical measurements (ERT + P-wave seismic + GNSS RTK) combined with
documentary context** can improve groundwater prospecting — with explicit
uncertainty, blinding, and leakage-safe validation.

> **Status: research software.** AquaScan does **not** "detect groundwater" in
> v0.2. No real ERT/seismic groundwater validation has been performed.

## What's new in v0.2

- **National Enrichment Engine v1**: resumable, interruptible, crash-safe
  builder for the 139,697-record normalized IGME snapshot. See
  [docs/national-enrichment.md](docs/national-enrichment.md).
- **National depth-to-water model v1** published on Hugging Face: 16,722 READY
  records, 44 provinces and 323 domains, with the Phase 0 firewall active. See
  [docs/national-model.md](docs/national-model.md).
- Aggregate data-quality summary:
  [`reports/national_enrichment_summary_v1.json`](reports/national_enrichment_summary_v1.json).
- Public cleanup: internal notes, hardware-acceptance bookkeeping and
  superseded reports removed.
- Citation, repository URLs and license metadata corrected.

The record-level dataset and raw API payloads are **not** redistributed. Model
weights are published on Hugging Face, not stored in this Git repository. See
[NOTICE_DATA.md](NOTICE_DATA.md).

## Why AquaScan

Groundwater is a critical resource. Borehole success rates depend on
geological judgement, and geophysical methods are often used without
quantified uncertainty. AquaScan is an attempt to build a **reproducible,
open, blinded** experimental framework to test whether geophysics adds
measurable predictive value beyond public documentary context.

## Scientific approach

```
Context (public hydrogeological records)
  + ERT + P-wave seismic + uncertainty
  + blinded validation (prediction freeze before Ground Truth reveal)
```

## Current status

- [x] IGME documentary ingestion / normalization / provenance (RAW immutable)
- [x] National enrichment engine v1 (139,697 records; resumable, crash-safe)
- [x] Ground Truth evidence methodology (independent targets, A/B/C/D quality)
- [x] Leakage-safe validation (grouped / spatial-block / cross-domain)
- [x] Open-source context baseline v2 (leakage-free, frozen)
- [x] GeoAI Context v0 (documentary context model)
- [x] ERT / seismic processing architecture (canonical schemas + pipelines)
- [ ] Physical hardware acceptance: **NOT YET PERFORMED** (no hardware acquired)
- [ ] Real groundwater validation with ERT/seismic: **NOT YET PERFORMED**
- [ ] Dry-well detection: **NOT VALIDATED**

Phrases like "detects groundwater", "finds water", or "proven system" describe
future objectives only, not current capability.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Ground Truth methodology

See [docs/methodology.md](docs/methodology.md).

## Leakage-safe validation

- No naive random splits. Grouped folds, spatial-block holdout, and
  cross-domain holdout are enforced with tests.
- **Historical note:** Gate 8 v1 contained target leakage and is marked
  invalid; see [docs/baseline-leakage-postmortem.md](docs/baseline-leakage-postmortem.md).
- All published results are the leakage-free Baseline v2 / forensic numbers;
  see [docs/current-results.md](docs/current-results.md).

## Open-source context baseline

Baseline v2 is the canonical comparator for future physical experiments.
See `reports/` (curated aggregate reports) and
[docs/current-results.md](docs/current-results.md).

## Model on Hugging Face

**[`angellopez/aquascan-open-science-model`](https://huggingface.co/angellopez/aquascan-open-science-model)**

The canonical model is **AquaScan National Groundwater Depth Model v1**, trained
on 16,722 READY depth records from 44 provinces and 323 domains, with the 10
sealed Phase 0 San Antonio sites excluded. It contains the model card, fitted
weights, preprocessor, feature manifest, split-conformal calibration and
confidence/OOD policies. It is a national-coverage research baseline, **not** a
uniformly validated model: only 21 domains have a dedicated holdout evaluation
and 201 have fewer than 30 training records. See
[docs/national-model.md](docs/national-model.md) for metrics, support and
limitations.

The previous five-domain GeoAI Context v0 baseline (563 records from five
domains in the Balearic Islands and Castilla y León / Duero basin) is preserved
at the
[`v0-5domain` tag](https://huggingface.co/angellopez/aquascan-open-science-model/tree/v0-5domain).
It is not the national model and is not validated outside those domains.

The **record-level training dataset is not published** on Hugging Face or
GitHub (IGME license review pending; record-level privacy). Model weights are
not stored in this Git repository; they are in the Hugging Face repo above.
Both models are research baselines: no field validation, no dry-well Ground
Truth, and no groundwater-detection capability.

## National enrichment engine

The national engine enriches the normalized IGME snapshot against documentary
endpoints and writes a partitioned Parquet dataset. Final execution:

- **139,697** records: 139,492 `COMPLETE`, 205 `PARTIAL`.
- **0** duplicate IDs, **0** missing IDs.
- **2,477** QA warnings, **0** critical.
- **280** partitions; 1,074,216 network requests; 4,419.5 s runtime.
- 4,744-record regression canary: 4,744 / 4,744, no mismatches.

The dataset itself is not redistributed (IGME license review and record-level
privacy). The aggregate summary and the engine code are public. See
[docs/national-enrichment.md](docs/national-enrichment.md).

## Physical processing stack

Canonical ERT/seismic observation schemas, deterministic pipelines,
handcrafted + grid encoders, cross-modal registration, and synthetic
engineering fixtures live in `src/aquascan/physical/`. It is tested on
**synthetic** data only; no real field performance is claimed.

## Field hardware

Phase 0 requires a geophysical field stack: a 48-electrode ERT system (5 m
spacing), a 24-channel P-wave seismic system (14 Hz geophones, 2 m spacing),
GNSS RTK positioning, a calibration reference box, field power (LiFePO4), and
rugged compute/storage. **No hardware has been acquired yet**; Gate 11
acceptance is **NOT READY**.

See [docs/hardware.md](docs/hardware.md) for the bill of materials, data
contracts, bench-acceptance checklist, budget scenarios and the rent-vs-buy
plan for the pilot. The model itself runs on CPU; no GPU is required.

## What is NOT yet validated

- Real ERT/seismic groundwater accuracy (no field data)
- Dry-well discrimination (no confirmed negative Ground Truth)
- Physical hardware acceptance (Gate 11 status: NOT READY — no hardware)
- Generalization to Región de los arenales (documented OOD failure)

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
aquascan --help
aquascan national --help
```

Synthetic demonstrations (no data required):

```bash
python -c "from aquascan.physical.synthetic.fixtures import make_homogeneous_halfspace; print('ok')"
```

The IGME downloader and enrichment scripts require network access and are
documented in `scripts/`; the core test suite uses fixtures/mocks and runs
offline.

## Project structure

```
src/aquascan/        # package: data, national, enrichment, ground_truth,
                     #   evaluation, baselines, features, modeling, physical
schemas/             # canonical JSON Schemas
protocols/           # public experimental protocols
examples/synthetic/  # clearly-labeled synthetic fixtures
docs/                # architecture, methodology, results, roadmap
reports/            # curated aggregate reports (no record-level data)
tests/               # public test suite (runs without private data)
```

## Data attribution

See [NOTICE_DATA.md](NOTICE_DATA.md). IGME is the source of original water-point
records; AquaScan is independent and not endorsed by IGME.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache-2.0 (see [LICENSE](LICENSE)).

## Author

Ángel López Morales — [GitHub](https://github.com/angelopezvpb)
