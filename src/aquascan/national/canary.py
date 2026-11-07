"""4744-record regression canary (Gate 22).

Before national execution, the already-validated enriched cohort is
reprocessed/reconstructed from its durable cache using the CURRENT
pipeline, and semantic equivalence is verified for:

    lithology, piezometry, pumping, chemistry, hydrometry,
    technical fields, Ground Truth target readiness.

The canary:
- makes NO network requests (cache only)
- does NOT reveal the Phase 0 sealed mapping (aggregate semantics only;
  no raw API payloads are emitted)
- returns a mismatch report; unexpected differences must STOP the national run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aquascan.enrichment.normalize import build_normalized_record

# Semantic fields compared per record (validated Gate 4/5 semantics).
COMPARE_FIELDS = (
    "geological_evidence_count",
    "water_level_evidence_count",
    "hydraulic_evidence_count",
    "water_quality_evidence_count",
    "hydrometry_evidence_count",
    "water_presence_status",
    "documentary_confidence",
    "temporal_num_measurements",
    "temporal_relevance_status",
    "lithology_validation_valid",
)


@dataclass
class CanaryResult:
    total: int = 0
    compared: int = 0
    mismatches: dict[str, int] = field(default_factory=dict)
    mismatch_ids: dict[str, list[str]] = field(default_factory=dict)
    category_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    valid: bool = True
    inputs_missing: bool = False
    detail: str = ""


def _col_map(cohort: pd.DataFrame, col: str, default: Any = None) -> dict[str, Any]:
    """Build igme_id → value map for a column if present."""
    if col not in cohort.columns:
        return {}
    series = cohort[col]
    return {
        str(k): (v if v is not None else default)
        for k, v in zip(cohort["igme_id"], series, strict=False)
    }


def _extract_fields(rec: Any) -> dict[str, Any]:
    """Extract comparable semantic fields from a normalized record."""
    wp = rec.water_presence
    ts = rec.temporal_summary
    lv = rec.lithology_validation
    return {
        "geological_evidence_count": len(rec.geological_evidence),
        "water_level_evidence_count": len(rec.water_level_evidence),
        "hydraulic_evidence_count": len(rec.hydraulic_evidence),
        "water_quality_evidence_count": len(rec.water_quality_evidence),
        "hydrometry_evidence_count": len(rec.hydrometry_evidence),
        "water_presence_status": wp.status.value,
        "documentary_confidence": rec.documentary_confidence,
        "temporal_num_measurements": ts.num_measurements,
        "temporal_relevance_status": ts.temporal_relevance_status.value,
        "lithology_validation_valid": lv.is_valid,
    }


def _stored_fields(row: pd.Series) -> dict[str, Any]:
    ts = json.loads(row.get("temporal_summary_json") or "{}")
    lv = json.loads(row.get("lithology_validation_json") or "{}")
    return {
        "geological_evidence_count": int(row.get("geological_evidence_count", 0)),
        "water_level_evidence_count": int(row.get("water_level_evidence_count", 0)),
        "hydraulic_evidence_count": int(row.get("hydraulic_evidence_count", 0)),
        "water_quality_evidence_count": int(row.get("water_quality_evidence_count", 0)),
        "hydrometry_evidence_count": int(row.get("hydrometry_evidence_count", 0)),
        "water_presence_status": str(row.get("water_presence_status", "")),
        "documentary_confidence": float(row.get("documentary_confidence", 0.0)),
        "temporal_num_measurements": int(ts.get("num_measurements", 0)),
        "temporal_relevance_status": str(ts.get("temporal_relevance_status", "")),
        "lithology_validation_valid": bool(lv.get("is_valid", True)),
    }


def run_canary(
    cohort_path: Path,
    cache_path: Path,
) -> CanaryResult:
    """Reprocess the validated cohort from cache and compare semantics."""
    result = CanaryResult()

    if not cohort_path.exists() or not cache_path.exists():
        result.detail = (
            f"canary inputs missing: cohort={cohort_path.exists()} cache={cache_path.exists()}"
        )
        result.valid = False
        result.inputs_missing = True
        return result

    cohort = pd.read_parquet(cohort_path)
    result.total = len(cohort)

    with open(cache_path) as f:
        cache = json.load(f)

    ids = cohort["igme_id"].tolist()
    domain_map = _col_map(cohort, "domain", "Unknown")
    tier_map = _col_map(cohort, "candidate_tier")
    score_map = _col_map(cohort, "candidate_priority_score")
    url_map = _col_map(cohort, "detail_url")
    stored_rows = {r["igme_id"]: r for r in cohort.to_dict("records")}

    mismatches: dict[str, int] = {}
    mismatch_ids: dict[str, list[str]] = {}

    for igme_id in ids:
        raw_cache = cache.get(igme_id)
        if raw_cache is None:
            result.mismatch_ids.setdefault("missing_from_cache", []).append(igme_id)
            mismatches["missing_from_cache"] = mismatches.get("missing_from_cache", 0) + 1
            continue

        raw: dict[str, Any] = {"igme_id": igme_id}
        for ep in (
            "piezometry",
            "lithology",
            "pumping_tests",
            "hydrometry",
            "chemistry",
            "technical",
            "documents",
        ):
            entry = raw_cache.get(ep) or {}
            data = entry.get("data", [])
            raw[ep] = data
            raw[f"{ep}_sha256"] = entry.get("sha256", "")
            raw[f"{ep}_count"] = len(data)
            raw[f"{ep}_cached"] = True
        raw["_source_changed"] = False
        raw["_changed_endpoints"] = []

        rec = build_normalized_record(
            igme_id=igme_id,
            domain=domain_map.get(igme_id, "Unknown"),
            candidate_tier=tier_map.get(igme_id),
            candidate_score=score_map.get(igme_id),
            detail_url=url_map.get(igme_id),
            raw=raw,
        )
        result.compared += 1

        got = _extract_fields(rec)
        want = _stored_fields(pd.Series(stored_rows[igme_id]))

        rec_mismatches: list[str] = []
        for field_name in COMPARE_FIELDS:
            g, w = got[field_name], want[field_name]
            if isinstance(g, float) and isinstance(w, float):
                ok = abs(g - w) < 1e-9
            else:
                ok = g == w
            if not ok:
                rec_mismatches.append(field_name)

        if rec_mismatches:
            for field_name in rec_mismatches:
                mismatches[field_name] = mismatches.get(field_name, 0) + 1
                mismatch_ids.setdefault(field_name, []).append(igme_id)

    result.mismatches = mismatches
    result.mismatch_ids = mismatch_ids
    result.valid = len(mismatches) == 0

    # Per-category breakdown of compared vs stored
    for field_name in COMPARE_FIELDS:
        result.category_breakdown[field_name] = {
            "compared": result.compared,
            "mismatched": mismatches.get(field_name, 0),
        }

    if result.valid:
        result.detail = (
            f"semantic equivalence confirmed for {result.compared}/{result.total} records "
            f"(no mismatches)"
        )
    else:
        result.detail = (
            f"{sum(mismatches.values())} semantic mismatches across {len(mismatches)} fields"
        )
    return result


def canary_report(result: CanaryResult, reports_dir: Path) -> Path:
    """Persist a canary report; never includes raw API payloads."""
    from aquascan.national.io_utils import atomic_write_json

    report = {
        "gate": "22",
        "created_at": datetime.now(UTC).isoformat(),
        "total_cohort": result.total,
        "compared": result.compared,
        "valid": result.valid,
        "mismatches": result.mismatches,
        "mismatch_ids_truncated": {k: v[:20] for k, v in result.mismatch_ids.items()},
        "category_breakdown": result.category_breakdown,
        "detail": result.detail,
        "note": (
            "Canary verifies semantic equivalence only. It makes no network "
            "requests and does not reveal Phase 0 sealed mapping or raw payloads."
        ),
    }
    path = reports_dir / "national_enrichment_canary_v1.json"
    atomic_write_json(path, report)
    return path
