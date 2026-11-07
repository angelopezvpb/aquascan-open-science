"""AquaScan National Enrichment Engine v1.

Resumable, interruptible and crash-safe national dataset builder.

The engine enriches the normalized national IGME snapshot (139,697 records)
against the validated Gate 4/5 enrichment branches (lithology, piezometry,
pumping tests, hydrometry, hydrochemistry, technical data, documents) using:

- a persistent SQLite execution state database
- a durable per-record/per-endpoint response cache
- endpoint-level resume with exact continuation from durable state
- atomic writes, checkpoints, graceful pause/stop, bounded retries,
  rate limiting, bounded concurrency, network circuit breaker,
  disk-space monitoring, session history and deterministic output.

The scientific enrichment semantics are the validated Gate 4/5 semantics.
This module implements operational scale and reliability only.
"""

from __future__ import annotations

ENGINE_NAME = "national_enrichment_engine"
ENGINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1"

__all__ = ["ENGINE_NAME", "ENGINE_VERSION", "SCHEMA_VERSION"]
