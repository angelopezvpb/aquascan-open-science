#!/usr/bin/env python3
"""Gate 4: Controlled Ground Truth Enrichment on 500 IGME records.

Builds and validates a production-like Ground Truth enrichment pipeline on a
controlled cohort of exactly 500 IGME records (100 per domain across 5 Phase 0
domains).

DO NOT: enrich the national dataset, train GeoAI, process ERT/seismic, publish.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.cohort.selection import DOMAINS, select_cohort_500
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
COHORT_PATH = Path("data/processed/ground_truth_cohort500_v0_3.parquet")
CACHE_DIR = Path("data/discovery_cache/gate4_cohort500")
REPORTS_DIR = Path("reports")

VERSION = "v0.3"


def main(limit: int | None = None) -> None:
    print("=" * 70)
    print("Gate 4: Controlled Ground Truth Enrichment (500 records)")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Select cohort ────────────────────────────────────────────
    print("\n[1] Selecting 500-record cohort …")
    df = pd.read_parquet(INPUT_PATH)
    cohort = select_cohort_500(df)
    if limit is not None and limit < len(cohort):
        cohort = cohort.head(limit)
    print(f"  Cohort size: {len(cohort)}")
    cohort.to_csv(REPORTS_DIR / "ground_truth_cohort500_selection_v0_3.csv", index=False)
    print(f"  Saved selection to {REPORTS_DIR}/ground_truth_cohort500_selection_v0_3.csv")

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
        if (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(igme_ids)}")
        raw = client.query_all(igme_id, cache)
        # Propagate source-change flags
        changed = cache.changed_records.get(igme_id, [])
        raw["_source_changed"] = igme_id in cache.changed_records
        raw["_changed_endpoints"] = changed
        raw_by_id[igme_id] = raw

    # ── 3. Normalize ────────────────────────────────────────────────
    print("\n[3] Normalizing evidence …")
    records: list[NormalizedGroundTruthRecord] = []
    for igme_id in igme_ids:
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

    # ── 4. Coverage analysis ────────────────────────────────────────
    print("\n[4] Coverage analysis …")
    coverage_df = compute_coverage(records)
    coverage_path = REPORTS_DIR / "ground_truth_cohort500_coverage_v0_3.csv"
    coverage_df.to_csv(coverage_path, index=False)

    # National + per-domain coverage summary
    national_metrics = [
        "any_enrichment",
        "has_lithology",
        "has_piezometry",
        "has_pumping_tests",
        "has_chemistry",
        "has_hydrometry",
    ]
    coverage_summary: dict[str, Any] = {"national": {}, "per_domain": {}}

    n_total = len(records)
    for metric in national_metrics:
        k = int(coverage_df[metric].sum())
        lo, hi = wilson_ci(k, n_total)
        coverage_summary["national"][metric] = {
            "count": k,
            "proportion": round(k / n_total, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    # Evidence families
    ge2 = int((coverage_df["evidence_family_count"] >= 2).sum())
    ge3 = int((coverage_df["evidence_family_count"] >= 3).sum())
    coverage_summary["national"]["ge_2_families"] = ge2
    coverage_summary["national"]["ge_3_families"] = ge3
    coverage_summary["national"]["geo_plus_water"] = int(
        coverage_df["geological_plus_water_level"].sum()
    )
    coverage_summary["national"]["geo_plus_hydraulic"] = int(
        coverage_df["geological_plus_hydraulic"].sum()
    )
    coverage_summary["national"]["geo_plus_water_plus_hydraulic"] = int(
        coverage_df["geological_plus_water_plus_hydraulic"].sum()
    )

    for domain in DOMAINS:
        sub = coverage_df[coverage_df["domain"] == domain]
        n = len(sub)
        if n == 0:
            continue
        coverage_summary["per_domain"][domain] = {"n": n}
        for metric in national_metrics:
            k = int(sub[metric].sum())
            lo, hi = wilson_ci(k, n)
            coverage_summary["per_domain"][domain][metric] = {
                "count": k,
                "proportion": round(k / n, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }

    # ── 5. Target readiness ─────────────────────────────────────────
    print("\n[5] Target readiness …")
    readiness_df = compute_target_readiness_stats(records)
    readiness_path = REPORTS_DIR / "ground_truth_target_readiness_v0_3.csv"
    readiness_df.to_csv(readiness_path, index=False)

    # ── 6. Phase 0 domain evaluation v2 ─────────────────────────────
    print("\n[6] Phase 0 domain evaluation v2 …")
    domain_df = compute_domain_comparison_v2(coverage_df, records)
    domain_path = REPORTS_DIR / "ground_truth_domain_comparison_v0_3.csv"
    domain_df.to_csv(domain_path, index=False)

    # ── 7. Manual audit ─────────────────────────────────────────────
    print("\n[7] Manual audit (≥25 records) …")
    audit_df = run_manual_audit(records, raw_by_id, n=25)
    audit_path = REPORTS_DIR / "ground_truth_manual_audit_v0_3.csv"
    audit_df.to_csv(audit_path, index=False)
    n_disc = int((audit_df["verdict"] == "DISCREPANCY").sum())
    print(f"  Audited {len(audit_df)}: {len(audit_df) - n_disc} PASS, {n_disc} discrepancies")

    # ── 8. API metrics ──────────────────────────────────────────────
    print("\n[8] API operational metrics …")
    api_metrics = client.get_metrics_summary()
    api_metrics_path = REPORTS_DIR / "ground_truth_api_metrics_v0_3.json"
    with open(api_metrics_path, "w") as f:
        json.dump(api_metrics, f, indent=2, default=str)

    # ── 9. Save normalized parquet ──────────────────────────────────
    print("\n[9] Saving normalized cohort parquet …")
    normalized_rows = []
    for rec in records:
        d = rec.model_dump()
        # Flatten for parquet: serialize nested evidence as JSON, keep scalars
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
        d["lithology_validation_json"] = json.dumps(
            rec.lithology_validation.model_dump(), default=str
        )
        # Target readiness flattened per target
        for target, ass in rec.target_readiness.items():
            d[f"readiness_{target}"] = ass.readiness.value
            d[f"readiness_{target}_quality"] = ass.evidence_quality
            d[f"readiness_{target}_confidence"] = ass.documentary_confidence
        # Remove raw nested dicts that break parquet
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
        normalized_rows.append(d)

    normalized_df = pd.DataFrame(normalized_rows)
    normalized_df.to_parquet(COHORT_PATH, index=False)

    # ── 10. Summary JSON ────────────────────────────────────────────
    print("\n[10] Saving summary …")
    summary = {
        "gate": "4",
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "cohort_size": len(records),
        "domains": DOMAINS,
        "coverage": coverage_summary,
        "target_readiness": readiness_df.to_dict(orient="records"),
        "phase0_v2": domain_df.to_dict(orient="records"),
        "manual_audit": {
            "audited": len(audit_df),
            "discrepancies": n_disc,
            "target": "0 silent semantic corruptions",
        },
        "api_metrics": api_metrics,
    }
    summary_path = REPORTS_DIR / "ground_truth_cohort500_summary_v0_3.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Artifacts created:")
    for p in sorted(REPORTS_DIR.glob("*v0_3.*")):
        print(f"  {p}")
    print(f"  {COHORT_PATH}")
    print("=" * 70)

    # ── 11. Scale-readiness verdict ─────────────────────────────────
    print("\n" + "=" * 70)
    print("SCALE-READINESS ASSESSMENT (step 15)")
    print("=" * 70)
    print(f"  Extraction reliable: {api_metrics['failed_requests'] == 0}")
    print(f"  Cache entries: {cache.record_count}")
    print(f"  Cache integrity valid: {cache.verify_integrity()['valid']}")
    print(f"  Source responses changed: {len(cache.changed_records)} records")
    print(f"  Manual audit discrepancies: {n_disc}/{len(audit_df)}")

    # Final verdict
    if api_metrics["failed_requests"] > 0 or n_disc > 0:
        verdict = "PASS WITH LIMITATIONS — controlled enrichment works but scale issues remain"
    else:
        verdict = "PASS — Ground Truth pipeline validated at controlled scale"
    print(f"\nFINAL VERDICT: {verdict}")
    summary["final_verdict"] = verdict
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 4 enrichment")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit cohort size (for smoke testing)"
    )
    args = parser.parse_args()
    main(limit=args.limit)
