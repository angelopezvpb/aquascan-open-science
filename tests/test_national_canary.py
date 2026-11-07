"""Regression canary tests (Gate 22) using the real validated cohort.

The canary reprocesses the already-validated 4,744-record cohort from its
durable cache with the CURRENT pipeline and verifies semantic equivalence.
It makes no network requests and reveals no raw API payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquascan.national.canary import run_canary

REPO = Path(__file__).resolve().parent.parent
COHORT = REPO / "data/processed/ground_truth_cohort5000_v0_4.parquet"
CACHE = REPO / "data/discovery_cache/gate5_cohort5000/enrichment_cache.json"


@pytest.mark.skipif(
    not COHORT.exists() or not CACHE.exists(),
    reason="validated cohort / cache not present in this checkout",
)
def test_canary_passes_on_real_cohort(tmp_path: Path) -> None:
    """Semantic equivalence must hold for the validated 4,744-record cohort."""
    result = run_canary(COHORT, CACHE)
    assert result.total == 4744
    assert result.compared == 4744
    assert not result.inputs_missing
    # No semantic mismatches: lithology, piezometry, pumping, chemistry,
    # hydrometry, technical fields, and Ground Truth target readiness.
    assert result.valid, result.mismatches
    assert result.detail


@pytest.mark.skipif(
    not COHORT.exists() or not CACHE.exists(),
    reason="validated cohort / cache not present in this checkout",
)
def test_canary_report_exposes_no_raw_payloads(tmp_path: Path) -> None:
    """The canary report must never reveal Phase 0 sealed mapping / raw data."""
    from aquascan.national.canary import canary_report

    result = run_canary(COHORT, CACHE)
    path = canary_report(result, tmp_path)
    blob = path.read_text()
    assert "raw_response" not in blob
    assert '"Litologia"' not in blob
    assert '"Nivel"' not in blob
    assert '"Coordinates"' not in blob
    # Only aggregate counts and truncated ID lists are emitted.
    data = json.loads(blob)
    assert data["compared"] == 4744
    assert data["mismatches"] == {}


@pytest.mark.skipif(
    not COHORT.exists() or not CACHE.exists(),
    reason="validated cohort / cache not present in this checkout",
)
def test_canary_detects_pipeline_regression(tmp_path: Path) -> None:
    """A deliberately broken normalize must make the canary fail (STOP)."""
    import aquascan.national.canary as canary_mod

    original = canary_mod.build_normalized_record

    def broken(igme_id, **kwargs):
        raw = kwargs["raw"]
        # Simulate a semantic regression: drop all lithology evidence.
        raw["lithology"] = []
        return original(igme_id, **kwargs)

    canary_mod.build_normalized_record = broken
    try:
        result = run_canary(COHORT, CACHE)
    finally:
        canary_mod.build_normalized_record = original

    assert not result.valid
    assert result.mismatches.get("geological_evidence_count", 0) > 0
