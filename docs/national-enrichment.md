# National Enrichment Engine v1

Resumable, interruptible, crash-safe builder for the national normalized IGME
snapshot. It enriches documentary endpoints (lithology, piezometry, pumping
tests, hydrometry, hydrochemistry, technical data and documents) and produces a
partitioned Parquet dataset.

> **Scope:** documentary enrichment only. The engine does not validate
> groundwater, does not replace field studies, and does not use proprietary
> AutoCal code.

## Results (aggregate)

| Metric | Value |
|---|---:|
| Records | 139,697 |
| Status `COMPLETE` | 139,492 |
| Status `PARTIAL` | 205 |
| Duplicate IDs | 0 |
| Missing IDs | 0 |
| QA `WARNING` | 2,477 |
| QA `CRITICAL` | 0 |
| Partitions | 280 |
| Network requests | 1,074,216 |
| Runtime | 4,419.5 s |
| Regression canary | 4,744 / 4,744, no mismatches |

Endpoint coverage (records without that endpoint):

| Endpoint | Records |
|---|---:|
| `no_lithology` | 75,169 |
| `no_piezometry` | 89,008 |
| `no_pumping_test` | 10,107 |
| `no_chemistry` | 25,276 |
| `no_hydrometry` | 50,243 |
| `no_technical` | 2,900 |
| `no_documents` | 0 |

The aggregate summary is in
[`reports/national_enrichment_summary_v1.json`](../reports/national_enrichment_summary_v1.json).

## Architecture

```
normalized IGME snapshot (Parquet)
        │
        ▼
SQLite state DB (per record / per endpoint)
        │
        ├── durable response cache (atomic writes)
        ├── bounded retries + exponential backoff
        ├── global rate limiter + circuit breaker
        ├── checkpoints / immutable partitions
        └── disk-space monitor
        │
        ▼
partitioned Parquet + manifest (SHA-256)
```

## Commands

```bash
# Configure the engine against the normalized snapshot
aquascan national init --source data/processed/igme_candidates_v1.parquet

# Smoke test against the API (human review before a full run)
aquascan national start --limit 20

# Run / resume / pause / stop
aquascan national start
aquascan national resume
aquascan national pause
aquascan national stop

# Inspect and repair
aquascan national status
aquascan national retry-failed
aquascan national verify

# Assemble partitions into the final dataset
aquascan national finalize
```

For offline development, `start` and `resume` accept `--simulated`. Simulated
runs exercise the code and the state machine; they are **not** field validation
and must not be presented as real-world results.

## Operational guarantees

- Exact resume from durable state; never restarts from record 1.
- Stale `RUNNING` recovery after a crash.
- Graceful `SIGINT` / `SIGTERM`; recovery after `SIGKILL`.
- Atomic cache writes; a cache entry is never silently overwritten.
- Idempotent endpoint-level retries with bounded backoff.
- Global rate limiting and a network circuit breaker.
- Immutable batch partitions and SHA-256 verification.
- Disk-critical pause instead of corrupting state.
- No Phase 0 sealed mapping or raw API payload in reports.

## Data availability

The national dataset (≈200 MB Parquet) is **not redistributed** in this
repository. Reasons:

- the IGME reuse terms require human review (`NOTICE_DATA.md`);
- the dataset is record-level and may contain identifying or sensitive
  information;
- the public repository contains only code, schemas, aggregate statistics and
  clearly-labelled synthetic fixtures.

## Tests

```bash
python -m pytest tests/test_national_engine.py -q
python -m pytest tests/test_national_canary.py -q
```

`test_national_engine.py` is fully offline (simulated fetcher).
`test_national_canary.py` skips when the validated cohort and cache are not
present in the checkout.
