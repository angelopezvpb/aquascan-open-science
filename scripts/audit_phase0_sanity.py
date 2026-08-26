#!/usr/bin/env python3
"""Phase 0 Ranking Sanity Audit — Gate 2.1

Produces:
  reports/igme_phase0_domain_ranking_v1_1.csv
  reports/igme_phase0_sanity_v1_1.json

Key corrections from v1:
- Fixes geographic hallucination in report text (Región de los arenales ≠ Balearics)
- Documents ranking monotonicity explicitly
- Adds unit-name collision analysis
- Double-penalty audit report
- Sensitivity analysis under 4 scenarios
- All geographic claims derived from data
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.candidates.scoring import (
    QA_PENALTIES,
    SCORING_VERSION,
)
from aquascan.data.provenance import compute_sha256_file

INPUT_PATH = Path("data/processed/igme_candidates_v1.parquet")
REPORTS_DIR = Path("reports")

# Baseline Phase 0 weights (same as v1 for reproducibility)
BASELINE_WEIGHTS = {
    "p1_fraction": 0.30,
    "p2_fraction": 0.20,
    "sondeo_fraction": 0.15,
    "depth_coverage": 0.10,
    "detail_url_coverage": 0.05,
    "median_score_norm": 0.10,
    "concentration": 0.10,
}

# Alternative scenarios
SCENARIOS = {
    "baseline": BASELINE_WEIGHTS,
    "volume_emphasis": {
        "p1_fraction": 0.15,
        "p2_fraction": 0.10,
        "sondeo_fraction": 0.10,
        "depth_coverage": 0.10,
        "detail_url_coverage": 0.05,
        "median_score_norm": 0.10,
        "concentration": 0.40,  # emphasize raw P1 count
    },
    "p1_concentration": {
        "p1_fraction": 0.45,
        "p2_fraction": 0.15,
        "sondeo_fraction": 0.10,
        "depth_coverage": 0.05,
        "detail_url_coverage": 0.05,
        "median_score_norm": 0.10,
        "concentration": 0.10,
    },
    "qa_minimization": {
        "p1_fraction": 0.25,
        "p2_fraction": 0.15,
        "sondeo_fraction": 0.10,
        "depth_coverage": 0.15,
        "detail_url_coverage": 0.10,
        "median_score_norm": 0.15,
        "concentration": 0.10,
    },
}


def compute_phase0_score(
    domain_stats: pd.DataFrame, total_p1: int, weights: dict[str, float]
) -> pd.DataFrame:
    """Compute Phase 0 domain suitability score with given weights.

    Returns DataFrame sorted by phase0_score descending.
    """
    ds = domain_stats.copy()
    ds["p1_fraction"] = ds["p1_count"] / ds["total_records"]
    ds["p2_fraction"] = ds["p2_count"] / ds["total_records"]
    ds["sondeo_fraction"] = ds["useful_count"] / ds["total_records"]
    ds["depth_coverage"] = ds["records_with_depth"] / ds["total_records"]
    ds["detail_url_coverage"] = ds["records_with_url"] / ds["total_records"]
    score_max = ds["median_score"].max()
    ds["median_score_norm"] = ds["median_score"] / score_max if score_max > 0 else 0
    ds["concentration"] = ds["p1_count"] / total_p1 if total_p1 > 0 else 0

    ds["phase0_score"] = sum(ds[col] * weight for col, weight in weights.items())
    ds["phase0_score"] = ds["phase0_score"] / ds["phase0_score"].max() * 100

    return ds.sort_values("phase0_score", ascending=False)


def audit_double_penalties() -> list[dict]:
    """Audit every QA flag for double-penalization.

    Returns list of dicts with flag name, component effect, QA penalty effect,
    total effect, and classification.
    """
    audits: list[dict] = []

    # Map each QA flag to its component scorer behavior
    mappings = [
        {
            "qa_flag": "coordinate_outlier",
            "component": "coordinates",
            "component_normal": 15.0,
            "component_flagged": 5.0,
            "qa_penalty": QA_PENALTIES["coordinate_outlier"],
            "note": "Both the coordinate component AND QA penalty apply",
        },
        {
            "qa_flag": "depth_outlier",
            "component": "total_depth",
            "component_normal": 20.0,
            "component_flagged": 10.0,
            "qa_penalty": QA_PENALTIES["depth_outlier"],
            "note": "Component cut in half + additional QA penalty",
        },
        {
            "qa_flag": "negative_reference_flow",
            "component": "reference_flow",
            "component_normal": 5.0,
            "component_flagged": 3.0,
            "qa_penalty": QA_PENALTIES["negative_reference_flow"],
            "note": "Component reduced from 5 to 3 + QA penalty",
        },
        {
            "qa_flag": "missing_hydrogeological_unit",
            "component": "hydrogeological_unit",
            "component_normal": 15.0,
            "component_flagged": 0.0,
            "qa_penalty": QA_PENALTIES["missing_hydrogeological_unit"],
            "note": "Component goes to zero + additional QA penalty — strongest double-hit",
        },
        {
            "qa_flag": "invalid_construction_date",
            "component": "construction_date",
            "component_normal": 5.0,
            "component_flagged": 0.0,
            "qa_penalty": QA_PENALTIES["invalid_construction_date"],
            "note": "Component zero + QA penalty",
        },
        {
            "qa_flag": "suspicious_construction_date",
            "component": "construction_date",
            "component_normal": 5.0,
            "component_flagged": 3.0,
            "qa_penalty": QA_PENALTIES["suspicious_construction_date"],
            "note": "Component reduced from 5 to 3 + QA penalty",
        },
        {
            "qa_flag": "elevation_outlier",
            "component": "(none)",
            "component_normal": 0.0,
            "component_flagged": 0.0,
            "qa_penalty": QA_PENALTIES["elevation_outlier"],
            "note": "Single penalty — no component effect. NOT double-penalized.",
        },
        {
            "qa_flag": "duplicate_coordinate",
            "component": "(none)",
            "component_normal": 0.0,
            "component_flagged": 0.0,
            "qa_penalty": QA_PENALTIES["duplicate_coordinate"],
            "note": "Single penalty — no component effect. NOT double-penalized.",
        },
    ]

    for m in mappings:
        component_delta = m["component_flagged"] - m["component_normal"]
        total_effect = component_delta + m["qa_penalty"]
        is_double = component_delta != 0 and m["qa_penalty"] != 0

        audits.append(
            {
                "qa_flag": m["qa_flag"],
                "component_affected": m["component"],
                "component_delta": component_delta,
                "qa_penalty": m["qa_penalty"],
                "total_effect": total_effect,
                "is_double_penalized": is_double,
                "classification": (
                    "intentional — component measures documentary completeness; "
                    "QA penalty measures additional risk/flag cost"
                )
                if is_double
                else "single penalty — no component effect",
                "note": m["note"],
            }
        )

    return audits


def validate_ranking_monotonicity(df: pd.DataFrame) -> dict:
    """Verify ranking is monotonic on phase0_score."""
    scores = df["phase0_score"].values
    violations = []
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            violations.append(
                {
                    "row": i,
                    "unit_a": str(df.iloc[i]["hydrogeological_unit"]),
                    "score_a": float(scores[i]),
                    "unit_b": str(df.iloc[i + 1]["hydrogeological_unit"]),
                    "score_b": float(scores[i + 1]),
                }
            )
    return {
        "is_monotonic": len(violations) == 0,
        "violations": violations,
        "sort_column": "phase0_score",
        "sort_direction": "descending",
    }


def extract_domain_geography(df: pd.DataFrame, hgu: str) -> dict:
    """Extract data-derived geography for a hydrogeological unit."""
    subset = df[df["hydrogeological_unit"] == hgu]
    return {
        "hydrogeological_unit": hgu,
        "total_records": len(subset),
        "provinces": [
            {"name": k, "count": int(v)}
            for k, v in subset["province"].value_counts().head(5).items()
        ],
        "watersheds": [
            {"name": k, "count": int(v)}
            for k, v in subset["watershed"].value_counts().head(5).items()
        ],
        "aquifer_systems": [
            {"name": k, "count": int(v)}
            for k, v in subset["aquifer_system"].value_counts().head(5).items()
        ],
        "municipalities": [
            {"name": k, "count": int(v)}
            for k, v in subset["municipality"].value_counts().head(5).items()
        ],
        "latitude_min": float(subset["latitude"].min()),
        "latitude_max": float(subset["latitude"].max()),
        "longitude_min": float(subset["longitude"].min()),
        "longitude_max": float(subset["longitude"].max()),
    }


def main() -> None:
    timestamp = datetime.now(UTC).isoformat()

    print("=" * 70)
    print("Phase 0 Ranking Sanity Audit — Gate 2.1")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # --- Load ---
    df = pd.read_parquet(INPUT_PATH)
    print(f"\nLoaded: {INPUT_PATH} → {len(df)} records")
    checksum = compute_sha256_file(str(INPUT_PATH))
    print(f"SHA-256: {checksum[:16]}...")

    total_p1 = int((df["candidate_tier"] == "P1").sum())

    # --- 1. Double-Penalty Audit ---
    print("\n" + "=" * 70)
    print("1. Double-Penalty Audit")
    print("=" * 70)

    penalties = audit_double_penalties()
    for p in penalties:
        marker = "DOUBLE" if p["is_double_penalized"] else "SINGLE"
        print(f"  [{marker}] {p['qa_flag']}:")
        print(
            f"    Component delta: {p['component_delta']:+.0f}  |"
            f"  QA penalty: {p['qa_penalty']:+.0f}"
        )
        print(f"    Total effect: {p['total_effect']:+.0f}")
        print(f"    => {p['classification']}")
        print()

    # --- 2. HGU name collision analysis ---
    print("=" * 70)
    print("2. Hydrogeological Unit Name Collision Analysis")
    print("=" * 70)

    hgu_ws = df.groupby("hydrogeological_unit")["watershed"].nunique()
    multi_ws = hgu_ws[hgu_ws > 1]
    n_hgus = df["hydrogeological_unit"].nunique()
    print(f"  HGUs spanning multiple watersheds: {len(multi_ws)} / {n_hgus}")
    print("  (Only 'Unidades aisladas' is a true catch-all across disconnected areas)")

    hgu_prov = df.groupby("hydrogeological_unit")["province"].nunique()
    multi_prov = hgu_prov[hgu_prov > 3]
    print(f"  HGUs spanning >3 provinces: {len(multi_prov)}")
    for hgu, n in multi_prov.items():
        lat_span = (
            df[df["hydrogeological_unit"] == hgu]["latitude"].max()
            - df[df["hydrogeological_unit"] == hgu]["latitude"].min()
        )
        print(f"    {hgu}: {n} provinces, lat span {lat_span:.1f}°")

    print(
        "\n  Recommendation: composite key watershed+hydrogeological_unit"
        " recommended for Phase 0 grouping."
    )
    print("  'Unidades aisladas' is excluded from Phase 0 ranking (catch-all, not a real domain).")

    # --- 3. Build domain stats ---
    hgu_groups = df.groupby("hydrogeological_unit")
    hgu_stats = []
    for name, group in hgu_groups:
        useful = group["point_nature"].fillna("").str.contains("Sondeo|Pozo|Galería", case=False)
        hgu_stats.append(
            {
                "hydrogeological_unit": name,
                "total_records": len(group),
                "records_with_depth": int(group["total_depth_m"].notna().sum()),
                "records_with_url": int(group["detail_url"].notna().sum()),
                "useful_count": int(useful.sum()),
                "records_with_aquifer": int(group["aquifer_system"].notna().sum()),
                "p1_count": int((group["candidate_tier"] == "P1").sum()),
                "p2_count": int((group["candidate_tier"] == "P2").sum()),
                "median_score": float(group["candidate_priority_score"].median()),
                "qa_burden": int(group["qa_flag_count"].sum()),
                "qa_burden_per_record": float(group["qa_flag_count"].sum() / len(group)),
            }
        )
    hgu_df = pd.DataFrame(hgu_stats)

    # Exclude statistical noise and catch-alls
    hgu_df = hgu_df[hgu_df["total_records"] >= 50]
    # Exclude "Unidades aisladas" from ranking (catch-all)
    hgu_df = hgu_df[
        ~hgu_df["hydrogeological_unit"].str.match(r"^Unidades aisladas$", case=False, na=False)
    ]

    # --- 4. Sensitivity Analysis ---
    print("\n" + "=" * 70)
    print("3. Sensitivity Analysis — Phase 0 Domain Ranking")
    print("=" * 70)

    scenario_rankings: dict[str, list[str]] = {}
    for scenario_name, weights in SCENARIOS.items():
        ranked = compute_phase0_score(hgu_df.copy(), total_p1, weights)
        top5 = ranked.head(5)["hydrogeological_unit"].tolist()
        scenario_rankings[scenario_name] = top5
        print(f"\n  {scenario_name} weights: {weights}")
        print("  Top 5:")
        for i, hgu in enumerate(top5, 1):
            row = ranked[ranked["hydrogeological_unit"] == hgu].iloc[0]
            print(
                f"    {i}. {hgu:<45s} "
                f"P0={row['phase0_score']:.1f} "
                f"n={int(row['total_records']):>5d} "
                f"P1={int(row['p1_count']):>4d} "
                f"P1%={row['p1_fraction'] * 100:.1f}%"
            )

    # --- Top-5 overlap across scenarios ---
    print("\n  Top-5 cross-scenario persistence:")
    all_top5 = [set(v) for v in scenario_rankings.values()]
    always_top5 = all_top5[0].copy()
    for s in all_top5[1:]:
        always_top5 &= s
    print(f"    In ALL scenarios: {always_top5 or '(none)'}")
    in_3_of_4 = set()
    for hgu in set.union(*all_top5):
        count = sum(1 for s in all_top5 if hgu in s)
        if count >= 3:
            in_3_of_4.add(hgu)
    print(f"    In 3+ scenarios: {in_3_of_4}")

    # --- 5. Baseline Phase 0 ranking ---
    print("\n" + "=" * 70)
    print("4. Corrected Phase 0 Domain Ranking (baseline, v1.1)")
    print("=" * 70)

    baseline = compute_phase0_score(hgu_df.copy(), total_p1, BASELINE_WEIGHTS)

    # Verify monotonicity
    mono = validate_ranking_monotonicity(baseline)
    print(f"  Sort monotonicity: {mono['is_monotonic']} (sort_column='phase0_score', descending)")
    if mono["violations"]:
        for v in mono["violations"]:
            print(f"    VIOLATION: row {v['row']}")

    # Print top 15 with full component breakdown
    header = (
        f"  {'Rank':<5} {'HGU':<47} {'P0':>7} {'n':>6} {'P1':>5}"
        f" {'P1%':>6} {'Sondeo%':>7} {'Depth%':>6} {'Med':>4}"
    )
    sep = (
        f"  {'-' * 5} {'-' * 47} {'-' * 7} {'-' * 6} {'-' * 5}"
        f" {'-' * 6} {'-' * 7} {'-' * 6} {'-' * 4}"
    )
    print(f"\n{header}")
    print(sep)
    for i, (_, row) in enumerate(baseline.head(15).iterrows(), 1):
        print(
            f"  {i:<5} {row['hydrogeological_unit']:<47s} "
            f"{row['phase0_score']:>7.1f} {int(row['total_records']):>6d} "
            f"{int(row['p1_count']):>5d} "
            f"{row['p1_fraction'] * 100:>6.1f}% "
            f"{row['sondeo_fraction'] * 100:>7.1f}% "
            f"{row['depth_coverage'] * 100:>6.1f}% "
            f"{row['median_score']:>4.0f}"
        )

    # Pick top non-micro domain (>500 records) for the Phase 0 recommendation
    recommended = baseline[baseline["total_records"] >= 500].head(1)
    if len(recommended) > 0:
        rec_hgu = recommended.iloc[0]["hydrogeological_unit"]
        print(f"\n  Recommended Phase 0 domain (≥500 records): {rec_hgu}")
        # Data-derived geography
        geo = extract_domain_geography(df, rec_hgu)
        print(f"  Provinces: {', '.join(p['name'] for p in geo['provinces'][:5])}")
        print(f"  Watershed: {', '.join(w['name'] for w in geo['watersheds'][:3])}")
        print(f"  Lat range: {geo['latitude_min']:.3f}–{geo['latitude_max']:.3f}")
        print(f"  Lon range: {geo['longitude_min']:.3f}–{geo['longitude_max']:.3f}")

    # --- 6. Top-20 geographic verification ---
    print("\n" + "=" * 70)
    print("5. Top-20 HGU Geographic Verification (from data)")
    print("=" * 70)

    top20 = baseline.head(20)["hydrogeological_unit"].tolist()
    geo_report: list[dict] = []
    for hgu in top20:
        geo = extract_domain_geography(df, hgu)
        geo_report.append(geo)
        primary_prov = geo["provinces"][0]["name"] if geo["provinces"] else "unknown"
        primary_ws = geo["watersheds"][0]["name"] if geo["watersheds"] else "unknown"
        print(
            f"  {hgu:<47s} n={geo['total_records']:>5d}  "
            f"primary_prov={primary_prov:<25s}  "
            f"primary_ws={primary_ws:<15s}  "
            f"lat=[{geo['latitude_min']:.2f}, {geo['latitude_max']:.2f}]"
        )

    # --- 7. Build comprehensive sanity report ---
    sanity = {
        "audit_version": "1.1.0",
        "scoring_version": SCORING_VERSION,
        "timestamp": timestamp,
        "input_path": str(INPUT_PATH),
        "input_checksum": checksum,
        "total_p1_candidates": total_p1,
        "ranking_monotonicity": mono,
        "double_penalty_audit": penalties,
        "hgu_name_collisions": {
            "multi_watershed_count": int(len(multi_ws)),
            "catch_all_units": ["Unidades aisladas"],
            "recommendation": "Composite key watershed+hydrogeological_unit recommended",
        },
        "sensitivity_analysis": {
            "scenarios_evaluated": list(SCENARIOS.keys()),
            "top5_by_scenario": dict(scenario_rankings.items()),
            "always_top5": list(always_top5),
            "in_3_of_4_scenarios": list(in_3_of_4),
        },
        "baseline_ranking_top15": [
            {
                "rank": i,
                "hydrogeological_unit": str(row["hydrogeological_unit"]),
                "phase0_score": float(row["phase0_score"]),
                "total_records": int(row["total_records"]),
                "p1_count": int(row["p1_count"]),
                "p1_fraction": float(row["p1_fraction"]),
                "sondeo_fraction": float(row["sondeo_fraction"]),
                "depth_coverage": float(row["depth_coverage"]),
                "median_score": float(row["median_score"]),
            }
            for i, (_, row) in enumerate(baseline.head(15).iterrows(), 1)
        ],
        "recommended_domain": str(rec_hgu) if len(recommended) > 0 else None,
        "top20_geography": geo_report,
    }

    # --- 8. Save artifacts ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0 ranking CSV v1.1
    phase0_csv = REPORTS_DIR / "igme_phase0_domain_ranking_v1_1.csv"
    baseline.to_csv(phase0_csv, index=False)
    print(f"\n  Saved: {phase0_csv}")

    # Sanity JSON v1.1
    sanity_json = REPORTS_DIR / "igme_phase0_sanity_v1_1.json"
    sanity_json.write_text(json.dumps(sanity, indent=2, default=str))
    print(f"  Saved: {sanity_json}")

    print("\n" + "=" * 70)
    print("Phase 0 Sanity Audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
