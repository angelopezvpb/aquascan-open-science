#!/usr/bin/env python3
"""Run the Candidate Priority Engine on IGME v2 and produce all output artifacts.

Generates:
  data/processed/igme_candidates_v1.parquet
  reports/igme_candidate_summary_v1.json
  reports/igme_candidate_domains_v1.csv
  reports/igme_candidate_top1000_v1.csv
  reports/igme_phase0_domain_ranking_v1.csv
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.candidates.scoring import (
    SCORING_VERSION,
    CandidatePriorityEngine,
)
from aquascan.data.provenance import compute_sha256_file

INPUT_PATH = Path("data/interim/igme_national_20260823T171204Z_v2.parquet")
OUTPUT_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

# Phase 0 domain ranking weights
PHASE0_WEIGHTS = {
    "p1_count": 0.30,
    "p2_count": 0.20,
    "sondeo_fraction": 0.15,
    "depth_coverage": 0.10,
    "detail_url_count": 0.05,
    "median_score": 0.10,
    "concentration": 0.10,  # candidates per km² proxy (candidates proportion)
}


def compute_phase0_domain_score(
    domain_stats: pd.DataFrame,
    total_candidates: int,
) -> pd.DataFrame:
    """Compute Phase 0 domain suitability scores.

    A separate metric from Candidate Priority Score. Ranks domains by
    their ability to support a controlled scientific validation cohort.
    """
    ds = domain_stats.copy()

    # P1 fraction of domain
    ds["p1_fraction"] = ds["p1_count"] / ds["total_records"]

    # P2 fraction
    ds["p2_fraction"] = ds["p2_count"] / ds["total_records"]

    # Sondeo/Pozo fraction
    ds["sondeo_fraction"] = ds["useful_count"] / ds["total_records"]

    # Depth coverage
    ds["depth_coverage"] = ds["records_with_depth"] / ds["total_records"]

    # Median score (already there, normalize to 0-1)
    score_max = ds["median_score"].max()
    ds["score_norm"] = ds["median_score"] / score_max if score_max > 0 else 0

    # Concentration: P1 count as fraction of total P1s
    ds["concentration"] = ds["p1_count"] / total_candidates if total_candidates > 0 else 0

    # Composite Phase 0 score
    ds["phase0_score"] = (
        ds["p1_fraction"] * PHASE0_WEIGHTS["p1_count"]
        + ds["p2_fraction"] * PHASE0_WEIGHTS["p2_count"]
        + ds["sondeo_fraction"] * PHASE0_WEIGHTS["sondeo_fraction"]
        + ds["depth_coverage"] * PHASE0_WEIGHTS["depth_coverage"]
        + ds["score_norm"] * PHASE0_WEIGHTS["median_score"]
        + ds["concentration"] * PHASE0_WEIGHTS["concentration"]
    )

    # Scale to 0-100
    ds["phase0_score"] = ds["phase0_score"] / ds["phase0_score"].max() * 100

    return ds.sort_values("phase0_score", ascending=False)


def count_major_qa(row: pd.Series) -> str:
    """List major QA issues for a record."""
    issues = []
    for flag in [
        "coordinate_outlier",
        "elevation_outlier",
        "depth_outlier",
        "negative_reference_flow",
        "missing_hydrogeological_unit",
        "invalid_construction_date",
    ]:
        if row.get(f"qa_{flag}", False):
            issues.append(flag)
    return "; ".join(issues) if issues else "none"


def main() -> None:
    timestamp = datetime.now(UTC).isoformat()

    print("=" * 70)
    print("AquaScan Candidate Priority Engine — Gate 2")
    print(f"Scoring version: {SCORING_VERSION}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # --- Load v2 ---
    print(f"\nLoading: {INPUT_PATH}")
    if INPUT_PATH.exists():
        input_checksum = compute_sha256_file(str(INPUT_PATH))
        print(f"  SHA-256: {input_checksum[:16]}...")
    else:
        input_checksum = "unknown"
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Records: {len(df)}")

    # --- Score ---
    print("\nScoring...")
    engine = CandidatePriorityEngine()
    scored = engine.score_dataframe(df)
    tiered = engine.assign_tiers(scored)
    print(f"  Scored: {len(tiered)} records")
    print(f"  Tier thresholds: {engine.tier_thresholds}")

    scores = tiered["candidate_priority_score"]

    # --- National analysis ---
    print("\n" + "=" * 70)
    print("National Candidate Analysis")
    print("=" * 70)

    stats = {
        "scoring_version": SCORING_VERSION,
        "timestamp": timestamp,
        "input_path": str(INPUT_PATH),
        "input_checksum": input_checksum,
        "total_scored": len(scores),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_median": float(scores.median()),
        "score_std": float(scores.std()),
        "score_p10": float(scores.quantile(0.10)),
        "score_p25": float(scores.quantile(0.25)),
        "score_p50": float(scores.quantile(0.50)),
        "score_p75": float(scores.quantile(0.75)),
        "score_p90": float(scores.quantile(0.90)),
        "score_p95": float(scores.quantile(0.95)),
        "score_p99": float(scores.quantile(0.99)),
        "tier_thresholds": {k: float(v) for k, v in (engine.tier_thresholds or {}).items()},
        "tier_counts": {
            tier: int((tiered["candidate_tier"] == tier).sum()) for tier in ["P1", "P2", "P3", "P4"]
        },
    }

    # QA counts by tier
    qa_cols = [c for c in tiered.columns if c.startswith("qa_")]
    qa_by_tier = {}
    for tier in ["P1", "P2", "P3", "P4"]:
        tier_df = tiered[tiered["candidate_tier"] == tier]
        qa_by_tier[tier] = {
            col: int(tier_df[col].sum()) for col in qa_cols if tier_df[col].sum() > 0
        }
    stats["qa_counts_by_tier"] = qa_by_tier

    print(
        f"\n  Scores: min={stats['score_min']:.0f} max={stats['score_max']:.0f} "
        f"mean={stats['score_mean']:.1f} median={stats['score_median']:.0f}"
    )
    print(f"  σ={stats['score_std']:.1f}")
    print(
        f"  P10={stats['score_p10']:.0f} P25={stats['score_p25']:.0f} "
        f"P50={stats['score_p50']:.0f} P75={stats['score_p75']:.0f} "
        f"P90={stats['score_p90']:.0f}"
    )
    print("\n  Tier counts:")
    for tier, count in stats["tier_counts"].items():
        print(f"    {tier}: {count} ({100 * count / len(scores):.1f}%)")

    # --- Save artifacts ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parquet
    parquet_path = OUTPUT_DIR / "igme_candidates_v1.parquet"
    tiered.to_parquet(parquet_path, index=False)
    print(f"\n  Saved: {parquet_path}")

    # Summary JSON
    summary_path = REPORTS_DIR / "igme_candidate_summary_v1.json"
    summary_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"  Saved: {summary_path}")

    # Top 1000
    top1000 = tiered.nlargest(1000, "candidate_priority_score")
    top1000_path = REPORTS_DIR / "igme_candidate_top1000_v1.csv"
    top1000.to_csv(top1000_path, index=False)
    print(f"  Saved: {top1000_path}")

    # --- Domain analysis ---
    print("\n" + "=" * 70)
    print("Domain Analysis")
    print("=" * 70)

    # By hydrogeological unit
    hgu_groups = tiered.groupby("hydrogeological_unit")
    hgu_stats = []
    for name, group in hgu_groups:
        useful = group["point_nature"].fillna("").str.contains("Sondeo|Pozo|Galería", case=False)
        hgu_stats.append(
            {
                "hydrogeological_unit": name,
                "total_records": len(group),
                "records_with_depth": int(group["total_depth_m"].notna().sum()),
                "records_with_url": int(group["detail_url"].notna().sum()),
                "records_clean_coords": int((~group["qa_coordinate_outlier"]).sum()),
                "useful_count": int(useful.sum()),
                "records_with_aquifer": int(group["aquifer_system"].notna().sum()),
                "p1_count": int((group["candidate_tier"] == "P1").sum()),
                "p2_count": int((group["candidate_tier"] == "P2").sum()),
                "p3_count": int((group["candidate_tier"] == "P3").sum()),
                "p4_count": int((group["candidate_tier"] == "P4").sum()),
                "median_score": float(group["candidate_priority_score"].median()),
                "major_qa": group.apply(count_major_qa, axis=1).value_counts().head(3).to_dict(),
            }
        )
    hgu_df = pd.DataFrame(hgu_stats)

    # Phase 0 ranking
    total_p1 = int((tiered["candidate_tier"] == "P1").sum())
    phase0 = compute_phase0_domain_score(hgu_df, total_p1)
    phase0 = phase0[phase0["total_records"] >= 50]  # Minimum size filter
    phase0 = phase0.head(30)

    print("\n  Top 10 Phase 0 domains:")
    for _, row in phase0.head(10).iterrows():
        print(
            f"    {row['hydrogeological_unit']:<45s} "
            f"P0={row['phase0_score']:.1f} "
            f"n={row['total_records']:>6d} "
            f"P1={row['p1_count']:>4d}"
        )

    # Domain CSVs
    domains_path = REPORTS_DIR / "igme_candidate_domains_v1.csv"
    hgu_df.sort_values("total_records", ascending=False).to_csv(domains_path, index=False)
    print(f"\n  Saved: {domains_path}")

    phase0_path = REPORTS_DIR / "igme_phase0_domain_ranking_v1.csv"
    phase0.to_csv(phase0_path, index=False)
    print(f"  Saved: {phase0_path}")

    # By province
    prov_groups = tiered.groupby("province")
    prov_stats = []
    for name, group in prov_groups:
        useful = group["point_nature"].fillna("").str.contains("Sondeo|Pozo|Galería", case=False)
        prov_stats.append(
            {
                "province": name,
                "total_records": len(group),
                "p1_count": int((group["candidate_tier"] == "P1").sum()),
                "median_score": float(group["candidate_priority_score"].median()),
                "useful_count": int(useful.sum()),
            }
        )
    prov_df = pd.DataFrame(prov_stats).sort_values("p1_count", ascending=False)
    print("\n  Top 10 provinces by P1 count:")
    for _, row in prov_df.head(10).iterrows():
        print(
            f"    {row['province']:<35s} "
            f"n={row['total_records']:>6d} "
            f"P1={row['p1_count']:>4d} "
            f"med={row['median_score']:.0f}"
        )

    # --- Point nature breakdown ---
    print("\n  Candidate tiers by point nature:")
    nature_tiers = tiered.groupby(["point_nature", "candidate_tier"]).size().unstack(fill_value=0)
    for nature, row in nature_tiers.iterrows():
        total = row.sum()
        print(
            f"    {nature:<45s} n={total:>6d} P1={row.get('P1', 0):>4d} P2={row.get('P2', 0):>4d}"
        )

    # --- Completeness matrix ---
    print("\n  Completeness by tier:")
    fields = [
        "total_depth_m",
        "hydrogeological_unit",
        "aquifer_system",
        "drilling_method",
        "construction_date",
        "reference_flow_l_s",
    ]
    for tier in ["P1", "P2", "P3", "P4"]:
        tdf = tiered[tiered["candidate_tier"] == tier]
        row_str = f"    {tier}:"
        for f in fields:
            pct = 100 * tdf[f].notna().sum() / len(tdf)
            row_str += f" {f}={pct:.0f}%"
        print(row_str)

    print("\n" + "=" * 70)
    print("Candidate Priority Engine complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
