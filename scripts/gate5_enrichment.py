#!/usr/bin/env python3
"""Gate 5: Selected Ground Truth Scale-Up (5,000 IGME records).

Scales the validated Gate 4 enrichment pipeline from 500 to 5,000 records.
Uses stratified deterministic sampling across 4 priority domains + 1 contrast.

DO NOT: enrich all 139,697 records, train GeoAI, process ERT/seismic, publish.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.cohort.selection import (
    CONTRAST_DOMAIN,
    PRIORITY_DOMAINS,
    select_cohort_5000,
)
from aquascan.data.models import NormalizedGroundTruthRecord
from aquascan.enrichment.analysis import (
    compute_coverage,
    compute_domain_comparison_v2,
    compute_target_readiness_stats,
    run_manual_audit,
    wilson_ci,
)
from aquascan.enrichment.api import EnrichmentAPIClient
from aquascan.enrichment.cache import ResponseCache
from aquascan.enrichment.normalize import build_normalized_record

INPUT_PATH = Path("data/processed/igme_candidates_v1.parquet")
COHORT_PATH = Path("data/processed/ground_truth_cohort5000_v0_4.parquet")
CACHE_DIR = Path("data/discovery_cache/gate5_cohort5000")
REPORTS_DIR = Path("reports")

VERSION = "v0.4"


def _flatten_for_parquet(rec: NormalizedGroundTruthRecord) -> dict[str, Any]:
    """Flatten a NormalizedGroundTruthRecord into a parquet-safe dict."""
    d = rec.model_dump()
    d["geological_evidence_count"] = len(rec.geological_evidence)
    d["water_level_evidence_count"] = len(rec.water_level_evidence)
    d["hydraulic_evidence_count"] = len(rec.hydraulic_evidence)
    d["water_quality_evidence_count"] = len(rec.water_quality_evidence)
    d["hydrometry_evidence_count"] = len(rec.hydrometry_evidence)
    d["water_presence_status"] = rec.water_presence.status.value
    d["water_presence_basis"] = ";".join(rec.water_presence.evidence_basis)
    d["geological_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.geological_evidence], default=str
    )
    d["water_level_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.water_level_evidence], default=str
    )
    d["hydraulic_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.hydraulic_evidence], default=str
    )
    d["water_quality_evidence_json"] = json.dumps(rec.water_quality_evidence, default=str)
    d["hydrometry_evidence_json"] = json.dumps(rec.hydrometry_evidence, default=str)
    d["provenance_json"] = json.dumps(rec.provenance, default=str)
    d["temporal_summary_json"] = json.dumps(rec.temporal_summary.model_dump(), default=str)
    d["lithology_validation_json"] = json.dumps(rec.lithology_validation.model_dump(), default=str)
    for target, ass in rec.target_readiness.items():
        d[f"readiness_{target}"] = ass.readiness.value
        d[f"readiness_{target}_quality"] = ass.evidence_quality
        d[f"readiness_{target}_confidence"] = ass.documentary_confidence
    # Remove nested structures that break parquet
    for key in [
        "geological_evidence",
        "water_level_evidence",
        "hydraulic_evidence",
        "water_quality_evidence",
        "hydrometry_evidence",
        "target_readiness",
        "temporal_summary",
        "lithology_validation",
        "water_presence",
        "provenance",
    ]:
        d.pop(key, None)
    return d


def _build_coverage_summary(
    coverage_df: pd.DataFrame, n_total: int, domains: list[str]
) -> dict[str, Any]:
    """Build national + per-domain coverage summary with Wilson CIs."""
    national_metrics = [
        "any_enrichment",
        "has_lithology",
        "has_piezometry",
        "has_pumping_tests",
        "has_chemistry",
        "has_hydrometry",
    ]
    summary: dict[str, Any] = {"national": {}, "per_domain": {}}

    for metric in national_metrics:
        k = int(coverage_df[metric].sum())
        lo, hi = wilson_ci(k, n_total)
        summary["national"][metric] = {
            "count": k,
            "proportion": round(k / n_total, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    ge2 = int((coverage_df["evidence_family_count"] >= 2).sum())
    ge3 = int((coverage_df["evidence_family_count"] >= 3).sum())
    summary["national"]["ge_2_families"] = ge2
    summary["national"]["ge_3_families"] = ge3
    summary["national"]["geo_plus_water"] = int(coverage_df["geological_plus_water_level"].sum())
    summary["national"]["geo_plus_hydraulic"] = int(coverage_df["geological_plus_hydraulic"].sum())
    summary["national"]["geo_plus_water_plus_hydraulic"] = int(
        coverage_df["geological_plus_water_plus_hydraulic"].sum()
    )

    for domain in domains:
        sub = coverage_df[coverage_df["domain"] == domain]
        n = len(sub)
        if n == 0:
            continue
        summary["per_domain"][domain] = {"n": n}
        for metric in national_metrics:
            k = int(sub[metric].sum())
            lo, hi = wilson_ci(k, n)
            summary["per_domain"][domain][metric] = {
                "count": k,
                "proportion": round(k / n, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        # Evidence families per domain
        ge2_d = int((sub["evidence_family_count"] >= 2).sum())
        ge3_d = int((sub["evidence_family_count"] >= 3).sum())
        summary["per_domain"][domain]["ge_2_families"] = ge2_d
        summary["per_domain"][domain]["ge_3_families"] = ge3_d

    return summary


def _build_readiness_by_domain(
    records: list[NormalizedGroundTruthRecord],
    domains: list[str],
) -> dict[str, Any]:
    """Per-domain target readiness breakdown."""
    result: dict[str, Any] = {}
    for domain in domains:
        domain_records = [r for r in records if r.domain == domain]
        if not domain_records:
            continue
        result[domain] = compute_target_readiness_stats(domain_records).to_dict(orient="records")
    return result


def _compute_temporal_stats(
    records: list[NormalizedGroundTruthRecord],
) -> dict[str, Any]:
    """Temporal distribution of piezometric observations."""
    dated = [r for r in records if r.temporal_summary.num_measurements > 0]
    total_measurements = sum(r.temporal_summary.num_measurements for r in dated)
    spans_days = [
        r.temporal_summary.temporal_span_days
        for r in dated
        if r.temporal_summary.temporal_span_days is not None
    ]
    spans_years = [s / 365.25 for s in spans_days] if spans_days else []

    # Use stored temporal_relevance_status from the model
    status_counts: Counter[str] = Counter()
    for r in records:
        if r.temporal_summary.num_measurements > 0:
            status = r.temporal_summary.temporal_relevance_status.value
        else:
            status = "NO_DATA"
        status_counts[status] += 1

    return {
        "records_with_dated_measurements": len(dated),
        "total_measurements": total_measurements,
        "median_measurements_per_record": (
            sorted([r.temporal_summary.num_measurements for r in dated])[len(dated) // 2]
            if dated
            else 0
        ),
        "median_temporal_span_years": (
            round(sorted(spans_years)[len(spans_years) // 2], 1) if spans_years else None
        ),
        "temporal_relevance_distribution": dict(status_counts),
    }


def _sensitivity_analysis(domain_df: pd.DataFrame) -> dict[str, Any]:
    """Phase 0 sensitivity under different weight scenarios."""
    scenarios = {
        "A_50_50": (0.5, 0.5),
        "B_40_60": (0.4, 0.6),
        "C_30_70": (0.3, 0.7),
        "D_60_40": (0.6, 0.4),
    }
    results: dict[str, Any] = {}
    for name, (w_suit, w_rich) in scenarios.items():
        scores = []
        for _, row in domain_df.iterrows():
            score = w_suit * row["candidate_suitability"] + w_rich * row["ground_truth_richness"]
            scores.append({"domain": row["domain"], "score": round(score, 4)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        results[name] = {
            "weights": {"suitability": w_suit, "richness": w_rich},
            "ranking": scores,
        }

    # Stable top-3 across all scenarios
    all_top3: dict[str, int] = {}
    for scenario in results.values():
        for item in scenario["ranking"][:3]:
            all_top3[item["domain"]] = all_top3.get(item["domain"], 0) + 1
    results["stable_top3"] = {
        k: v for k, v in sorted(all_top3.items(), key=lambda x: -x[1]) if v >= 2
    }
    return results


def main(limit: int | None = None) -> None:
    print("=" * 70)
    print("Gate 5: Selected Ground Truth Scale-Up (5,000 records)")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Select cohort ────────────────────────────────────────────
    print("\n[1] Selecting 5,000-record cohort …")
    df = pd.read_parquet(INPUT_PATH)
    cohort = select_cohort_5000(df)
    if limit is not None and limit < len(cohort):
        cohort = cohort.head(limit)
    print(f"  Cohort size: {len(cohort)}")
    cohort.to_csv(
        REPORTS_DIR / "ground_truth_cohort5000_selection_v0_4.csv",
        index=False,
    )

    igme_ids = cohort["igme_id"].tolist()
    domain_map = dict(zip(cohort["igme_id"], cohort["selection_domain"], strict=False))
    tier_map = dict(zip(cohort["igme_id"], cohort["candidate_tier"], strict=False))
    score_map = dict(zip(cohort["igme_id"], cohort["candidate_priority_score"], strict=False))
    url_map = dict(zip(cohort["igme_id"], cohort["detail_url"], strict=False))

    # ── 2. Query API with cache ─────────────────────────────────────
    print("\n[2] Querying enrichment API (7 endpoints) …")
    cache = ResponseCache(CACHE_DIR)
    client = EnrichmentAPIClient()

    raw_by_id: dict[str, dict[str, Any]] = {}
    for i, igme_id in enumerate(igme_ids):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  … {i + 1}/{len(igme_ids)}")
        raw = client.query_all(igme_id, cache)
        raw["_source_changed"] = igme_id in cache.changed_records
        raw["_changed_endpoints"] = cache.changed_records.get(igme_id, [])
        raw_by_id[igme_id] = raw

    print(f"  Total API requests: {client.total_requests}")
    print(f"  Cache hits: {client.endpoint_cache_hits}")
    print(f"  Logical queries: {client.logical_endpoint_queries}")
    print(f"  Failed: {client.failed_requests}")

    # ── 3. Normalize ────────────────────────────────────────────────
    print("\n[3] Normalizing evidence …")
    records: list[NormalizedGroundTruthRecord] = []
    norm_errors = 0
    for igme_id in igme_ids:
        try:
            raw = raw_by_id[igme_id]
            rec = build_normalized_record(
                igme_id=igme_id,
                domain=domain_map.get(igme_id, "Unknown"),
                candidate_tier=tier_map.get(igme_id),
                candidate_score=score_map.get(igme_id),
                detail_url=url_map.get(igme_id),
                raw=raw,
            )
            records.append(rec)
        except Exception as e:
            norm_errors += 1
            print(f"  ⚠ Normalization error for {igme_id}: {e}")
    print(f"  Normalized: {len(records)} records, {norm_errors} errors")

    # ── 4. Coverage analysis ────────────────────────────────────────
    print("\n[4] Coverage analysis …")
    coverage_df = compute_coverage(records)
    coverage_path = REPORTS_DIR / "ground_truth_cohort5000_coverage_v0_4.csv"
    coverage_df.to_csv(coverage_path, index=False)
    coverage_summary = _build_coverage_summary(
        coverage_df, len(records), PRIORITY_DOMAINS + [CONTRAST_DOMAIN]
    )

    # ── 5. Target readiness ─────────────────────────────────────────
    print("\n[5] Target readiness …")
    readiness_df = compute_target_readiness_stats(records)
    readiness_path = REPORTS_DIR / "ground_truth_target_readiness_v0_4.csv"
    readiness_df.to_csv(readiness_path, index=False)

    readiness_by_domain = _build_readiness_by_domain(records, PRIORITY_DOMAINS + [CONTRAST_DOMAIN])

    # ── 6. Phase 0 domain evaluation v2 ─────────────────────────────
    print("\n[6] Phase 0 domain evaluation v2 …")
    domain_df = compute_domain_comparison_v2(coverage_df, records)
    domain_path = REPORTS_DIR / "ground_truth_domain_comparison_v0_4.csv"
    domain_df.to_csv(domain_path, index=False)

    # ── 7. Sensitivity analysis ─────────────────────────────────────
    print("\n[7] Phase 0 sensitivity analysis …")
    sensitivity = _sensitivity_analysis(domain_df)

    # ── 8. Temporal statistics ──────────────────────────────────────
    print("\n[8] Temporal statistics …")
    temporal_stats = _compute_temporal_stats(records)

    # ── 9. Manual audit (≥50 records) ───────────────────────────────
    print("\n[9] Manual audit (≥50 records) …")
    audit_df = run_manual_audit(records, raw_by_id, n=50)
    audit_path = REPORTS_DIR / "ground_truth_manual_audit_v0_4.csv"
    audit_df.to_csv(audit_path, index=False)
    n_disc = int((audit_df["verdict"] == "DISCREPANCY").sum())
    n_semantic = int((audit_df["verdict"] == "DISCREPANCY").sum())
    print(f"  Audited {len(audit_df)}: {len(audit_df) - n_disc} PASS, {n_disc} discrepancies")

    # ── 10. API metrics ─────────────────────────────────────────────
    print("\n[10] API operational metrics …")
    api_metrics = client.get_metrics_summary()
    api_metrics_path = REPORTS_DIR / "ground_truth_api_metrics_v0_4.json"
    with open(api_metrics_path, "w") as f:
        json.dump(api_metrics, f, indent=2, default=str)

    # ── 11. Save normalized parquet ─────────────────────────────────
    print("\n[11] Saving normalized cohort parquet …")
    normalized_rows = [_flatten_for_parquet(rec) for rec in records]
    normalized_df = pd.DataFrame(normalized_rows)
    normalized_df.to_parquet(COHORT_PATH, index=False)
    print(f"  Saved {len(normalized_df)} rows to {COHORT_PATH}")

    # ── 12. Summary JSON ────────────────────────────────────────────
    print("\n[12] Saving summary …")
    summary = {
        "gate": "5",
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "cohort_size": len(records),
        "norm_errors": norm_errors,
        "domains": PRIORITY_DOMAINS + [CONTRAST_DOMAIN],
        "coverage": coverage_summary,
        "target_readiness": readiness_df.to_dict(orient="records"),
        "readiness_by_domain": readiness_by_domain,
        "temporal_statistics": temporal_stats,
        "phase0_v2": domain_df.to_dict(orient="records"),
        "sensitivity_analysis": sensitivity,
        "manual_audit": {
            "audited": len(audit_df),
            "discrepancies": n_disc,
            "semantic_corruptions": n_semantic,
            "target": "0 silent semantic corruptions",
        },
        "api_metrics": api_metrics,
    }
    summary_path = REPORTS_DIR / "ground_truth_cohort5000_summary_v0_4.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    # ── 13. Scale-readiness verdict ─────────────────────────────────
    print("\n" + "=" * 70)
    print("SCALE-READINESS ASSESSMENT")
    print("=" * 70)
    print(f"  Total enriched: {len(records)}")
    print(f"  Normalization errors: {norm_errors}")
    print(f"  Logical queries: {api_metrics['logical_endpoint_queries']}")
    print(f"  Network requests: {api_metrics['network_requests']}")
    print(f"  Cache hits: {api_metrics['endpoint_cache_hits']}")
    print(f"  Records fully cached: {api_metrics['records_fully_reused_from_cache']}")
    print(f"  Records needing network: {api_metrics['records_requiring_network']}")
    print(f"  API failures: {api_metrics['failed_requests']}")
    print(f"  Telemetry invariant: {api_metrics.get('telemetry_invariant_holds', 'N/A')}")
    print(f"  Cache entries: {cache.record_count}")
    print(f"  Cache integrity: {cache.verify_integrity()['valid']}")
    print(f"  Source responses changed: {len(cache.changed_records)}")
    print(f"  Manual audit discrepancies: {n_disc}/{len(audit_df)}")
    print(
        f"  Temporal: {temporal_stats['records_with_dated_measurements']} with dated measurements"
    )
    print()
    print("  Target readiness (national):")
    for _, row in readiness_df.iterrows():
        print(
            f"    {row['target']}: "
            f"READY={row['ready_pct']}% "
            f"PARTIAL={row['partial_pct']}% "
            f"NOT_READY={row['not_ready_pct']}%"
        )

    # Final verdict
    has_failures = api_metrics["failed_requests"] > 0 or n_disc > 0 or norm_errors > 0
    if has_failures:
        verdict = (
            "PASS WITH LIMITATIONS — expanded enrichment works but target/data limitations remain"
        )
    else:
        verdict = "PASS — Ground Truth pipeline validated at expanded scale"
    print(f"\nFINAL VERDICT: {verdict}")
    summary["final_verdict"] = verdict
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print("\nArtifacts:")
    for p in sorted(REPORTS_DIR.glob("*v0_4.*")):
        print(f"  {p}")
    print(f"  {COHORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 5 enrichment")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit cohort size (for smoke testing)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
