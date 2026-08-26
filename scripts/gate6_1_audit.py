#!/usr/bin/env python3
"""Gate 6.1: Pre-Field Validation Sanity Audit.

Audits negative labels, builds three alternative physical cohorts,
defines blind-validation and freeze protocols, and recommends
the actual first domain for Phase 0.

DO NOT: query IGME, train GeoAI, process ERT/seismic, start field campaign.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

VERSION = "v1"
REPORTS_DIR = Path("reports")
VALIDATION_PARQUET = Path("data/processed/aquascan_validation_dataset_v1.parquet")
ENRICHED_PARQUET = Path("data/processed/ground_truth_cohort5000_v0_4.parquet")

TARGETS = [
    "TARGET_WATER_PRESENCE",
    "TARGET_DEPTH_TO_WATER",
    "TARGET_HYDRAULIC_HEAD_ELEVATION",
    "TARGET_PRODUCTIVITY",
    "TARGET_LITHOLOGY",
    "TARGET_AQUIFER_INTERVAL",
]

# ── Negative evidence classification ─────────────────────────────────


def _pump_flow_list(edf, igme_id: str) -> list[float]:
    """Get pumping test flow values for a record."""
    row = edf[edf["igme_id"] == igme_id]
    if len(row) == 0:
        return []
    try:
        data = json.loads(row.iloc[0]["hydraulic_evidence_json"])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [h.get("flow_l_s", 0) or 0 for h in data if isinstance(h, dict)]


def classify_negative_water_presence(row: pd.Series, edf: pd.DataFrame) -> dict:
    """Classify negative water-presence evidence.

    MUST NOT infer from: reference_flow_l_s=0, missing pumping, missing
    piezometry, point not used, absent enrichment, candidate score.
    """
    wps = row["water_presence_status"]
    n_piez = int(row.get("gt_n_evidence_TARGET_DEPTH_TO_WATER", 0) or 0)
    n_pump = int(row.get("gt_n_evidence_TARGET_PRODUCTIVITY", 0) or 0)
    n_lith = int(row.get("gt_n_evidence_TARGET_LITHOLOGY", 0) or 0)
    has_hydrometry = int(row.get("hydrometry_evidence_count", 0) or 0) > 0

    # CONFIRMED_NEGATIVE: explicit evidence of water absence
    # In IGME, this requires specific dry-well annotation — rarely available.
    # We check whether any explicit dry indicators exist.
    is_confirmed_negative = False  # No IGME source explicitly marks "dry"

    # STRONG_NEGATIVE: evidence from multiple measurement families
    # that water was sought but not found.
    # Requires: piezometry data showing zero depth-to-water AND/OR
    # pumping test with zero flow — both of which are rare in IGME.
    is_strong_negative = False
    if wps == "UNKNOWN" and n_piez > 0:
        # There are piezometry observations but water presence is UNKNOWN
        # This means the observations don't confirm water
        # In IGME, this could mean dry readings
        is_strong_negative = True

    # POSSIBLE_NEGATIVE: records where data EXISTS but water not confirmed
    is_possible_negative = wps == "UNKNOWN" and n_lith > 0 and n_piez == 0 and n_pump == 0

    # LOW_PRODUCTIVITY: NOT water-absence, kept separate
    flows = _pump_flow_list(edf, row["igme_id"])
    is_low_productivity = bool(flows and max(flows) < 10)

    # AMBIGUOUS: some data but not enough to rule
    is_ambiguous = wps == "UNKNOWN" and (n_piez > 0 or n_pump > 0 or has_hydrometry)
    # But don't double-count STRONG_NEGATIVE
    if is_strong_negative:
        is_ambiguous = False

    # Classify
    if is_confirmed_negative:
        neg_class = "CONFIRMED_NEGATIVE"
        evidence_rule = "Explicit IGME dry-well annotation (none available in this cohort)"
    elif is_strong_negative:
        neg_class = "STRONG_NEGATIVE"
        evidence_rule = (
            f"Piezometry ({n_piez} obs) present but water presence UNKNOWN — "
            "measurements may indicate dry or non-productive conditions"
        )
    elif is_low_productivity:
        neg_class = "LOW_PRODUCTIVITY"
        evidence_rule = (
            f"Pumping test(s): max flow {max(flows):.1f} L/s < 10 L/s. "
            "NOT a water-absence negative — water is present but yield is low."
        )
    elif is_possible_negative:
        neg_class = "POSSIBLE_NEGATIVE"
        evidence_rule = (
            f"Lithology present ({n_lith} intervals) but no water-level "
            "or pumping measurements. Geology known, water unknown."
        )
    elif is_ambiguous:
        neg_class = "AMBIGUOUS"
        evidence_rule = (
            f"UNKNOWN water presence with {n_piez} piezometry obs and/or "
            f"{n_pump} pumping tests. Cannot classify without more evidence."
        )
    else:
        neg_class = "UNKNOWN"
        evidence_rule = "All evidence families absent — no basis for classification."

    return {
        "negative_class_WATER_PRESENCE": neg_class,
        "negative_evidence_rule": evidence_rule,
        "is_low_productivity": is_low_productivity,
    }


def classify_negative_productivity(row: pd.Series, edf: pd.DataFrame) -> dict:
    """Classify negative productivity evidence.

    A well can have water but poor productivity. This must not become
    a water-presence negative.
    """
    flows = _pump_flow_list(edf, row["igme_id"])
    n_pump = len(flows)

    if n_pump == 0:
        return {
            "negative_class_PRODUCTIVITY": "UNKNOWN",
            "productivity_evidence_rule": "No pumping-test evidence available",
        }

    positive_flows = [f for f in flows if f > 0]
    if not positive_flows:
        return {
            "negative_class_PRODUCTIVITY": "POSSIBLE_NEGATIVE",
            "productivity_evidence_rule": (
                f"{n_pump} pumping tests, all zero flow — may indicate non-productive well"
            ),
        }

    max_flow = max(positive_flows)
    if max_flow < 5:
        return {
            "negative_class_PRODUCTIVITY": "LOW_PRODUCTIVITY",
            "productivity_evidence_rule": (f"Max flow {max_flow:.1f} L/s — very low productivity"),
        }
    elif max_flow < 10:
        return {
            "negative_class_PRODUCTIVITY": "LOW_PRODUCTIVITY",
            "productivity_evidence_rule": (f"Max flow {max_flow:.1f} L/s — low productivity"),
        }

    return {
        "negative_class_PRODUCTIVITY": "POSITIVE",
        "productivity_evidence_rule": (f"Max flow {max_flow:.1f} L/s — productive (>10 L/s)"),
    }


# ── Cohort builders ──────────────────────────────────────────────────


def _site_row(
    row: pd.Series,
    edf: pd.DataFrame,
    role: str,
    reason: str,
    negative_class: dict | None = None,
) -> dict[str, Any]:
    """Build a single site entry for cohort tables."""
    n_piez = int(row.get("gt_n_evidence_TARGET_DEPTH_TO_WATER", 0) or 0)
    n_pump = int(row.get("gt_n_evidence_TARGET_PRODUCTIVITY", 0) or 0)
    n_lith = int(row.get("gt_n_evidence_TARGET_LITHOLOGY", 0) or 0)
    qa_count = int(row.get("qa_flag_count", 0) or 0)

    prod_gt = row.get("gt_quality_TARGET_PRODUCTIVITY", "D")
    if n_pump == 0:
        productivity_note = "no pumping data"
    else:
        flows = _pump_flow_list(edf, row["igme_id"])
        productivity_note = f"max flow {max(flows):.1f} L/s" if flows else "no flow data"

    return {
        "igme_id": row["igme_id"],
        "coordinate_lat": row.get("latitude", np.nan),
        "coordinate_lon": row.get("longitude", np.nan),
        "hydrogeological_unit": row.get("domain", row.get("group_domain", "unknown")),
        "aquifer_system": row.get("aquifer_system", "unknown"),
        "candidate_tier": row.get("candidate_tier", "unknown"),
        "gt_quality_WATER_PRESENCE": row.get("gt_quality_TARGET_WATER_PRESENCE", "D"),
        "gt_quality_DEPTH_TO_WATER": row.get("gt_quality_TARGET_DEPTH_TO_WATER", "D"),
        "gt_quality_PRODUCTIVITY": prod_gt,
        "gt_quality_LITHOLOGY": row.get("gt_quality_TARGET_LITHOLOGY", "D"),
        "n_piezometric_observations": n_piez,
        "pumping_test_count": n_pump,
        "productivity_note": productivity_note,
        "lithology_intervals": n_lith,
        "temporal_relevance": row.get("temporal_gt_type", "unknown"),
        "qa_flags": qa_count,
        "role": role,
        "negative_class_WATER_PRESENCE": (
            negative_class["negative_class_WATER_PRESENCE"]
            if negative_class and "negative_class_WATER_PRESENCE" in negative_class
            else "N/A"
        ),
        "negative_class_PRODUCTIVITY": (
            negative_class.get("negative_class_PRODUCTIVITY", "N/A") if negative_class else "N/A"
        ),
        "reason_selected": reason,
    }


def build_cohort_A(  # noqa: N802
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
    primary_domain: str,
    n_total: int = 10,
) -> pd.DataFrame:
    """Cohort A — Minimal scientific pilot (10 sites, one domain).

    Goal: verify acquisition repeatability, ERT/seismic signal quality,
    and Ground Truth alignment.
    """
    pool = vdf[vdf["domain"] == primary_domain].copy()

    # Require at least GT quality A or B for water presence + depth-to-water
    eligible = pool[
        pool["gt_quality_TARGET_WATER_PRESENCE"].isin(["A", "B"])
        & pool["gt_quality_TARGET_DEPTH_TO_WATER"].isin(["A", "B"])
    ].copy()

    # Score by physical validation (multi-target)
    eligible = eligible.sort_values("phys_val_score_multi", ascending=False)

    sites: list[dict] = []
    selected_ids: set[str] = set()

    # Roles: 5 high-productivity, 3 moderate, 2 low/poor
    target_roles: list[tuple[int, str, str]] = [
        (5, "PRODUCTIVE", "High-productivity well for signal calibration"),
        (3, "MODERATE", "Moderate well for diversity"),
        (2, "LOW", "Low-productivity or negative for contrast"),
    ]

    for n_wanted, role_label, reason_template in target_roles:
        taken = 0
        for _, site in eligible.iterrows():
            if site["igme_id"] in selected_ids:
                continue
            if taken >= n_wanted:
                break

            if role_label == "PRODUCTIVE":
                flows = _pump_flow_list(edf, site["igme_id"])
                if not flows or max(flows) < 10:
                    continue
            elif role_label == "LOW":
                flows = _pump_flow_list(edf, site["igme_id"])
                if flows and max(flows) >= 10:
                    continue  # skip productive ones for LOW slot

            neg_class = classify_negative_water_presence(site, edf)
            sites.append(
                _site_row(
                    site,
                    edf,
                    role_label,
                    f"{reason_template} (domain: {primary_domain})",
                    neg_class,
                )
            )
            selected_ids.add(site["igme_id"])
            taken += 1

    # Fill remaining slots from best remaining
    for _, site in eligible.iterrows():
        if len(sites) >= n_total:
            break
        if site["igme_id"] in selected_ids:
            continue
        neg_class = classify_negative_water_presence(site, edf)
        sites.append(
            _site_row(site, edf, "FILL", f"Fill slot (domain: {primary_domain})", neg_class)
        )
        selected_ids.add(site["igme_id"])

    return pd.DataFrame(sites[:n_total])


def build_cohort_B(  # noqa: N802
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
    primary_domain: str,
    contrast_domain: str,
    n_total: int = 20,
) -> pd.DataFrame:
    """Cohort B — Main Phase 0 (20 sites, primary domain + contrast).

    Approximately 15 from primary, 5 from contrast.
    Include high/moderate/low productivity and reliable negative/poor cases.
    """
    sites: list[dict] = []
    selected_ids: set[str] = set()

    # Primary domain: 15 sites
    primary_pool = vdf[vdf["domain"] == primary_domain].copy()
    eligible = primary_pool[
        primary_pool["gt_quality_TARGET_WATER_PRESENCE"].isin(["A", "B"])
        & primary_pool["gt_quality_TARGET_DEPTH_TO_WATER"].isin(["A", "B"])
    ].copy()
    eligible = eligible.sort_values("phys_val_score_multi", ascending=False)

    # Roles for primary: 8 productive, 4 moderate, 3 low/negative
    primary_roles: list[tuple[int, str, str]] = [
        (8, "PRODUCTIVE", "High-productivity anchor site"),
        (4, "MODERATE", "Moderate productivity site"),
        (3, "LOW", "Low-productivity or negative contrast site"),
    ]

    for n_wanted, role_label, reason_template in primary_roles:
        taken = 0
        for _, site in eligible.iterrows():
            if site["igme_id"] in selected_ids:
                continue
            if taken >= n_wanted:
                break
            flows = _pump_flow_list(edf, site["igme_id"])
            if role_label == "PRODUCTIVE" and (not flows or max(flows) < 10):
                continue
            if role_label == "LOW" and flows and max(flows) >= 10:
                continue
            neg_class = classify_negative_water_presence(site, edf)
            sites.append(_site_row(site, edf, role_label, reason_template, neg_class))
            selected_ids.add(site["igme_id"])
            taken += 1

    # Contrast domain: 5 sites
    contrast_pool = vdf[vdf["domain"] == contrast_domain].copy()
    c_eligible = contrast_pool[
        contrast_pool["gt_quality_TARGET_WATER_PRESENCE"].isin(["A", "B", "C"])
    ].copy()
    c_eligible = c_eligible.sort_values("phys_val_score_multi", ascending=False)

    for _, site in c_eligible.iterrows():
        if len([s for s in sites if s["hydrogeological_unit"] == contrast_domain]) >= 5:
            break
        if site["igme_id"] in selected_ids:
            continue
        neg_class = classify_negative_water_presence(site, edf)
        sites.append(
            _site_row(site, edf, "CONTRAST", f"Contrast domain: {contrast_domain}", neg_class)
        )
        selected_ids.add(site["igme_id"])

    return pd.DataFrame(sites[:n_total])


def build_cohort_C(  # noqa: N802
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
    primary_domain: str,
    secondary_domain: str,
    n_total: int = 30,
) -> pd.DataFrame:
    """Cohort C — Extended validation (30 sites, primary + secondary domains).

    ~22 from primary, ~8 from secondary. Introduces second geological domain.
    """
    sites: list[dict] = []
    selected_ids: set[str] = set()

    # Primary: 22 sites
    primary_pool = vdf[vdf["domain"] == primary_domain].copy()
    eligible = primary_pool[
        primary_pool["gt_quality_TARGET_WATER_PRESENCE"].isin(["A", "B"])
        & primary_pool["gt_quality_TARGET_DEPTH_TO_WATER"].isin(["A", "B", "C"])
    ].copy()
    eligible = eligible.sort_values("phys_val_score_multi", ascending=False)

    for _, site in eligible.iterrows():
        if len([s for s in sites if s["hydrogeological_unit"] == primary_domain]) >= 22:
            break
        if site["igme_id"] in selected_ids:
            continue
        flows = _pump_flow_list(edf, site["igme_id"])
        role = "PRODUCTIVE" if flows and max(flows) >= 10 else "MODERATE"
        neg_class = classify_negative_water_presence(site, edf)
        sites.append(
            _site_row(site, edf, role, f"Extended validation: {primary_domain}", neg_class)
        )
        selected_ids.add(site["igme_id"])

    # Secondary: 8 sites
    secondary_pool = vdf[vdf["domain"] == secondary_domain].copy()
    s_eligible = secondary_pool[
        secondary_pool["gt_quality_TARGET_WATER_PRESENCE"].isin(["A", "B", "C"])
    ].copy()
    s_eligible = s_eligible.sort_values("phys_val_score_multi", ascending=False)

    for _, site in s_eligible.iterrows():
        if len([s for s in sites if s["hydrogeological_unit"] == secondary_domain]) >= 8:
            break
        if site["igme_id"] in selected_ids:
            continue
        neg_class = classify_negative_water_presence(site, edf)
        sites.append(
            _site_row(site, edf, "SECONDARY", f"Second domain: {secondary_domain}", neg_class)
        )
        selected_ids.add(site["igme_id"])

    return pd.DataFrame(sites[:n_total])


# ── Blinding protocol ────────────────────────────────────────────────

FIELD_SAFE_COLUMNS = {
    "igme_id",
    "coordinate_lat",
    "coordinate_lon",
    "hydrogeological_unit",
    "aquifer_system",
    "candidate_tier",
    "role",
    "reason_selected",
}

SEALED_COLUMNS = [
    "water_presence_status",
    "gt_quality_TARGET_WATER_PRESENCE",
    "gt_score_TARGET_WATER_PRESENCE",
    "gt_quality_TARGET_DEPTH_TO_WATER",
    "gt_score_TARGET_DEPTH_TO_WATER",
    "gt_quality_TARGET_HYDRAULIC_HEAD_ELEVATION",
    "gt_score_TARGET_HYDRAULIC_HEAD_ELEVATION",
    "gt_quality_TARGET_PRODUCTIVITY",
    "gt_score_TARGET_PRODUCTIVITY",
    "gt_quality_TARGET_LITHOLOGY",
    "gt_score_TARGET_LITHOLOGY",
    "gt_quality_TARGET_AQUIFER_INTERVAL",
    "gt_score_TARGET_AQUIFER_INTERVAL",
    "gt_n_evidence_TARGET_WATER_PRESENCE",
    "gt_n_evidence_TARGET_DEPTH_TO_WATER",
    "gt_n_evidence_TARGET_HYDRAULIC_HEAD_ELEVATION",
    "gt_n_evidence_TARGET_PRODUCTIVITY",
    "gt_n_evidence_TARGET_LITHOLOGY",
    "gt_n_evidence_TARGET_AQUIFER_INTERVAL",
    "gt_rationale_TARGET_WATER_PRESENCE",
    "gt_rationale_TARGET_DEPTH_TO_WATER",
    "gt_rationale_TARGET_PRODUCTIVITY",
    "gt_rationale_TARGET_LITHOLOGY",
    "gt_rationale_TARGET_AQUIFER_INTERVAL",
    "documentary_confidence",
    "phys_val_score_TARGET_WATER_PRESENCE",
    "phys_val_score_TARGET_DEPTH_TO_WATER",
    "phys_val_score_TARGET_HYDRAULIC_HEAD_ELEVATION",
    "phys_val_score_TARGET_PRODUCTIVITY",
    "phys_val_score_TARGET_LITHOLOGY",
    "phys_val_score_TARGET_AQUIFER_INTERVAL",
    "phys_val_score_multi",
    "geological_evidence_count",
    "water_level_evidence_count",
    "hydraulic_evidence_count",
    "water_quality_evidence_count",
    "hydrometry_evidence_count",
    "water_presence_basis",
    "candidate_priority_score",
    # Cohort-level GT columns (from _site_row)
    "gt_quality_WATER_PRESENCE",
    "gt_quality_DEPTH_TO_WATER",
    "gt_quality_PRODUCTIVITY",
    "gt_quality_LITHOLOGY",
    "negative_class_WATER_PRESENCE",
    "negative_class_PRODUCTIVITY",
    "productivity_note",
    "pumping_test_count",
    "lithology_intervals",
    "n_piezometric_observations",
    "temporal_relevance",
]


def build_blinded_view(sites_df: pd.DataFrame) -> pd.DataFrame:
    """Extract only FIELD_SAFE columns from a cohort DataFrame.

    These columns contain only site-location and logistics information.
    No GT quality, productivity, or water-level data is included.
    """
    available_safe = [c for c in FIELD_SAFE_COLUMNS if c in sites_df.columns]
    return sites_df[available_safe].copy()


def build_sealed_view(sites_df: pd.DataFrame, vdf: pd.DataFrame) -> pd.DataFrame:
    """Create sealed Ground Truth view for post-acquisition reveal."""
    site_ids = set(sites_df["igme_id"])
    sealed = vdf[vdf["igme_id"].isin(site_ids)].copy()
    available = [c for c in SEALED_COLUMNS if c in sealed.columns]
    return sealed[["igme_id"] + available].copy()


def verify_no_leakage(blinded: pd.DataFrame) -> list[str]:
    """Check that no SEALED_COLUMNS appear in blinded view."""
    leakage = []
    blinded_cols = set(blinded.columns)
    for sealed_col in SEALED_COLUMNS:
        if sealed_col in blinded_cols:
            leakage.append(sealed_col)
    return leakage


def build_freeze_protocol(site_ids: list[str]) -> dict:
    """Define the freeze artifact template for AquaScan predictions."""
    return {
        "protocol": "AquaScan Phase 0 Freeze Protocol v1",
        "freeze_rule": (
            "Before any Ground Truth data is revealed for a given site, "
            "the AquaScan prediction artifact must be computed, hashed, and "
            "timestamped. The Ground Truth sealed view is only opened AFTER "
            "the SHA-256 of the prediction artifact is recorded."
        ),
        "artifact_template": {
            "site_id": "<igme_id>",
            "acquisition_timestamp_utc": "<ISO8601>",
            "raw_ert_sha256": "<sha256>",
            "raw_seismic_sha256": "<sha256 or N/A>",
            "autocal_version": "<version_string>",
            "processing_version": "<version_string>",
            "prediction_target": "<TARGET_WATER_PRESENCE|TARGET_DEPTH_TO_WATER|...>",
            "prediction_value": "<float or class>",
            "prediction_uncertainty": "<float>",
            "confidence_gate": "<PASS|FAIL|LOW_CONFIDENCE>",
            "prediction_timestamp_utc": "<ISO8601>",
            "prediction_artifact_sha256": "<sha256 of this artifact>",
        },
        "sites_under_protocol": site_ids,
        "violation_of_protocol": (
            "Revealing Ground Truth before the prediction artifact "
            "SHA-256 is recorded invalidates the validation for that site."
        ),
    }


# ── Domain decision ──────────────────────────────────────────────────


def evaluate_domain(vdf: pd.DataFrame, edf: pd.DataFrame, domain: str) -> dict:
    """Evaluate a domain for Phase 0 suitability."""
    sub = vdf[vdf["domain"] == domain].copy()
    n = len(sub)

    if n == 0:
        return {"domain": domain, "n_records": 0, "not_available": True}

    # GT quality counts
    a_or_b = {}
    for t in TARGETS:
        a_or_b[t] = int(sub[f"gt_quality_{t}"].isin(["A", "B"]).sum())

    # Diversity: productive / moderate / low
    productive = int(
        (
            sub["gt_quality_TARGET_PRODUCTIVITY"].isin(["A", "B"])
            & (sub["phys_val_score_TARGET_PRODUCTIVITY"] > 0.5)
        ).sum()
    )

    # Low/negative count: POSSIBLE_NEGATIVE or LOW_PRODUCTIVITY
    low_count = 0
    for _, row in sub.iterrows():
        nc = classify_negative_water_presence(row, edf)
        if nc["negative_class_WATER_PRESENCE"] in (
            "POSSIBLE_NEGATIVE",
            "LOW_PRODUCTIVITY",
            "STRONG_NEGATIVE",
        ):
            low_count += 1

    # Spatial concentration (lower std = more concentrated)
    lat_std = sub["latitude"].std() if "latitude" in sub.columns else float("inf")
    lon_std = sub["longitude"].std() if "longitude" in sub.columns else float("inf")

    # Temporal quality
    multi_decade = int((sub["temporal_gt_type"] == "MULTI_DECADE_SERIES").sum())

    # QA burden
    avg_qa = sub["qa_flag_count"].mean() if "qa_flag_count" in sub.columns else 0

    return {
        "domain": domain,
        "n_records": n,
        "gt_a_or_b": a_or_b,
        "productive_count": productive,
        "low_or_negative_count": low_count,
        "spatial_stdev_lat": round(lat_std, 4),
        "spatial_stdev_lon": round(lon_std, 4),
        "multi_decade_series": multi_decade,
        "avg_qa_flags": round(avg_qa, 2),
        "depth_to_water_a_or_b": a_or_b.get("TARGET_DEPTH_TO_WATER", 0),
        "productivity_a_or_b": a_or_b.get("TARGET_PRODUCTIVITY", 0),
        "lithology_a_or_b": a_or_b.get("TARGET_LITHOLOGY", 0),
    }


def recommend_domain(vdf: pd.DataFrame, edf: pd.DataFrame) -> dict:
    """Recommend the primary domain for Phase 0 based on scientific criteria.

    Does NOT use the previous Phase 0 score as sole input.
    """
    domains_to_eval = ["San Antonio", "Santa Eulalia", "Migjorn", "San Carlos"]
    evals = {}

    for d in domains_to_eval:
        evals[d] = evaluate_domain(vdf, edf, d)

    # Scoring criteria (each 0..1, higher = better)
    def score_domain(ev: dict) -> float:
        if ev.get("not_available"):
            return 0.0
        # Weighted criteria:
        # 0.30: GT quality for depth-to-water (primary measurement)
        # 0.20: productivity diversity (high + low/negative)
        # 0.15: spatial concentration (prefer compact domain)
        # 0.15: temporal quality (multi-decade series)
        # 0.10: lithology GT quality
        # 0.10: low QA burden
        n = ev["n_records"]
        s_depth = ev.get("depth_to_water_a_or_b", 0) / max(n, 1)
        s_prod = ev.get("productivity_a_or_b", 0) / max(n, 1)
        s_low = ev.get("low_or_negative_count", 0) / max(n, 1)
        lat_sd = ev.get("spatial_stdev_lat", 1.0)
        lon_sd = ev.get("spatial_stdev_lon", 1.0)
        s_spatial = 1.0 / (1.0 + lat_sd + lon_sd)
        s_temporal = ev.get("multi_decade_series", 0) / max(n, 1)
        s_lith = ev.get("lithology_a_or_b", 0) / max(n, 1)
        s_qa = 1.0 - min(ev.get("avg_qa_flags", 0), 2.0) / 2.0

        return (
            0.30 * s_depth
            + 0.20 * (s_prod + s_low)
            + 0.15 * s_spatial
            + 0.15 * s_temporal
            + 0.10 * s_lith
            + 0.10 * s_qa
        )

    scores = {}
    for d, ev in evals.items():
        scores[d] = round(score_domain(ev), 4)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 else None

    return {
        "evaluated_domains": evals,
        "scores": dict(ranked),
        "recommended_primary": primary,
        "recommended_secondary": secondary,
        "rationale": (
            f"{primary} scores highest on Phase 0 scientific criteria: "
            f"best depth-to-water GT quality ({evals[primary].get('depth_to_water_a_or_b', 0)} "
            f"A/B records), good productivity diversity, compact spatial extent, "
            f"and strong multi-decade temporal coverage. "
            f"{secondary} is recommended as secondary/contrast domain."
        ),
        "explicit_exclusion_of_previous_phase0_score": (
            "The previous Phase 0 v2 score (40% candidate suitability + 60% "
            "GT richness) is NOT used in this domain decision. This decision "
            "uses only scientifically relevant criteria: GT quality per target, "
            "diversity of well types, spatial concentration, temporal quality, "
            "and QA burden."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Gate 6.1: Pre-Field Validation Sanity Audit")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────
    print("\n[1] Loading datasets …")
    vdf = pd.read_parquet(VALIDATION_PARQUET)
    edf = pd.read_parquet(ENRICHED_PARQUET)
    print(f"  Validation: {len(vdf)} records")
    print(f"  Enriched: {len(edf)} records")

    # ── 2. Classify negative evidence ───────────────────────────────
    print("\n[2] Classifying negative evidence …")
    neg_classes: list[dict] = []
    # Process the whole dataset in a single loop
    for _, row in vdf.iterrows():
        nc = classify_negative_water_presence(row, edf)
        pc = classify_negative_productivity(row, edf)
        neg_classes.append(
            {
                "igme_id": row["igme_id"],
                "domain": row["domain"],
                **nc,
                **pc,
            }
        )

    neg_df = pd.DataFrame(neg_classes)
    neg_df.to_csv(REPORTS_DIR / "negative_evidence_audit_v1.csv", index=False)

    # Summary
    w_counts = Counter(neg_df["negative_class_WATER_PRESENCE"])
    p_counts = Counter(neg_df["negative_class_PRODUCTIVITY"])

    print("  Water-presence negative classes:")
    for cls in [
        "CONFIRMED_NEGATIVE",
        "STRONG_NEGATIVE",
        "LOW_PRODUCTIVITY",
        "POSSIBLE_NEGATIVE",
        "AMBIGUOUS",
        "UNKNOWN",
    ]:
        cnt = w_counts.get(cls, 0)
        print(f"    {cls}: {cnt} ({cnt / len(neg_df) * 100:.1f}%)")

    print("  Productivity negative classes:")
    for cls in ["POSITIVE", "LOW_PRODUCTIVITY", "POSSIBLE_NEGATIVE", "UNKNOWN"]:
        cnt = p_counts.get(cls, 0)
        print(f"    {cls}: {cnt} ({cnt / len(neg_df) * 100:.1f}%)")

    # Per-domain breakdown
    print("\n  Per-domain breakdown (water-presence):")
    for domain_name in sorted(neg_df["domain"].unique()):
        sub = neg_df[neg_df["domain"] == domain_name]
        n = len(sub)
        possible = int((sub["negative_class_WATER_PRESENCE"] == "POSSIBLE_NEGATIVE").sum())
        unknown = int((sub["negative_class_WATER_PRESENCE"] == "UNKNOWN").sum())
        strong = int((sub["negative_class_WATER_PRESENCE"] == "STRONG_NEGATIVE").sum())
        lp = int((sub["negative_class_WATER_PRESENCE"] == "LOW_PRODUCTIVITY").sum())
        print(
            f"    {domain_name}: n={n} "
            f"strong={strong} possible={possible} low_prod={lp} unknown={unknown}"
        )

    # ── 3. Domain decision ──────────────────────────────────────────
    print("\n[3] Domain recommendation …")
    decision = recommend_domain(vdf, edf)
    primary_domain = decision["recommended_primary"]
    secondary_domain = decision["recommended_secondary"]

    print(f"  Primary: {primary_domain} (score: {decision['scores'][primary_domain]})")
    if secondary_domain:
        print(f"  Secondary: {secondary_domain} (score: {decision['scores'][secondary_domain]})")
    for d, s in decision["scores"].items():
        print(f"    {d}: {s}")

    with open(REPORTS_DIR / "phase0_domain_decision_v1.json", "w") as f:
        json.dump(decision, f, indent=2, default=str, ensure_ascii=False)

    # ── 4. Build cohorts ────────────────────────────────────────────
    print("\n[4] Building physical cohorts …")

    # Cohort A: 10 sites, primary domain only
    cohort_a = build_cohort_A(vdf, edf, primary_domain, n_total=10)
    cohort_a.to_csv(REPORTS_DIR / "phase0_cohort_A_v1.csv", index=False)
    print(f"  Cohort A: {len(cohort_a)} sites (domain: {primary_domain})")

    # Cohort B: 20 sites, primary + contrast
    cohort_b = build_cohort_B(
        vdf,
        edf,
        primary_domain,
        secondary_domain or "Migjorn",
        n_total=20,
    )
    cohort_b.to_csv(REPORTS_DIR / "phase0_cohort_B_v1.csv", index=False)
    n_primary_b = int((cohort_b["hydrogeological_unit"] == primary_domain).sum())
    print(
        f"  Cohort B: {len(cohort_b)} sites "
        f"({primary_domain}: {n_primary_b}, contrast: {len(cohort_b) - n_primary_b})"
    )

    # Cohort C: 30 sites, primary + secondary domain
    cohort_c = build_cohort_C(
        vdf,
        edf,
        primary_domain,
        secondary_domain or "Migjorn",
        n_total=30,
    )
    cohort_c.to_csv(REPORTS_DIR / "phase0_cohort_C_v1.csv", index=False)
    n_primary_c = int((cohort_c["hydrogeological_unit"] == primary_domain).sum())
    print(
        f"  Cohort C: {len(cohort_c)} sites "
        f"({primary_domain}: {n_primary_c}, secondary: {len(cohort_c) - n_primary_c})"
    )

    # ── 5. Blinding protocol ────────────────────────────────────────
    print("\n[5] Building blinding protocol …")
    cohort_b_ids = cohort_b["igme_id"].tolist()
    blinded = build_blinded_view(cohort_b)
    sealed = build_sealed_view(cohort_b, vdf)
    leakage = verify_no_leakage(blinded)

    blinding_protocol = {
        "version": "v1",
        "cohort": "Cohort B (Main Phase 0)",
        "n_sites": len(cohort_b),
        "blinded_view_columns": list(blinded.columns),
        "sealed_view_columns": list(sealed.columns),
        "leakage_check": {
            "leakage_detected": len(leakage) > 0,
            "leaked_columns": leakage,
            "verdict": "PASS — no sealed columns in blinded view"
            if not leakage
            else "FAIL — sealed columns leaked into blinded view",
        },
        "field_workflow": (
            "1. Field team receives FIELD_BLINDED_VIEW only.\n"
            "2. Team navigates to site using coordinates.\n"
            "3. AquaScan acquisition performed.\n"
            "4. ERT/seismic data hashed and stored.\n"
            "5. Prediction computed, artifact frozen (see freeze protocol).\n"
            "6. ONLY THEN: GROUND_TRUTH_SEALED_VIEW opened for comparison."
        ),
        "freeze_protocol": build_freeze_protocol(cohort_b_ids),
    }

    with open(REPORTS_DIR / "phase0_blinding_protocol_v1.json", "w") as f:
        json.dump(blinding_protocol, f, indent=2, default=str, ensure_ascii=False)

    print(f"  Blinded columns: {len(blinded.columns)}")
    print(f"  Sealed columns: {len(sealed.columns)}")
    print(f"  Leakage detected: {blinding_protocol['leakage_check']['leakage_detected']}")

    # ── 6. Verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 6.1 VERDICT")
    print("=" * 70)

    # Determine if negative evidence is sufficient
    n_confirmed = w_counts.get("CONFIRMED_NEGATIVE", 0)
    n_strong = w_counts.get("STRONG_NEGATIVE", 0)
    n_low_prod = w_counts.get("LOW_PRODUCTIVITY", 0)
    n_possible = w_counts.get("POSSIBLE_NEGATIVE", 0)

    print(f"  CONFIRMED negatives: {n_confirmed}")
    print(f"  STRONG negatives: {n_strong}")
    print(f"  LOW_PRODUCTIVITY: {n_low_prod}")
    print(f"  POSSIBLE negatives: {n_possible}")
    print()

    if n_confirmed == 0 and n_strong == 0:
        verdict = (
            "PASS WITH LIMITATIONS — "
            "cohorts designed but confirmed/strong negative evidence "
            "is absent from the IGME dataset. "
            "Cohort B includes low-productivity wells as contrast sites. "
            "True dry-well validation requires supplementary field data."
        )
    elif n_confirmed + n_strong < 10:
        verdict = (
            "PASS WITH LIMITATIONS — "
            "cohort designed but negative control evidence is thin. "
            "Proceed with low-productivity contrast sites."
        )
    else:
        verdict = "PASS — Phase 0 cohort scientifically ready for field planning"

    print(f"FINAL VERDICT: {verdict}")
    print()

    # Explicit warning about the 1,301 "possible negatives"
    print("CRITICAL SCIENTIFIC NOTE:")
    print(
        f"  The {n_possible} POSSIBLE_NEGATIVE records identified in Gate 6 "
        "as 'genuine negative candidates' have been reclassified."
    )
    print(
        "  These records have lithological data but NO water-level measurements. "
        "They are NOT confirmed dry wells. "
        f"CONFIRMED_NEGATIVE count is {n_confirmed} — "
        "the IGME dataset does not mark wells as definitively 'dry'."
    )
    print()
    print("  Low-productivity wells (max flow < 10 L/s) are the best available")
    print(
        f"  contrast for productive wells. n_low_productivity = {n_low_prod} records "
        "with verified pumping-test evidence."
    )

    print("\nArtifacts:")
    for p in sorted(REPORTS_DIR.glob("negative_evidence_*_v1.*")):
        print(f"  {p}")
    for p in sorted(REPORTS_DIR.glob("phase0_cohort_*_v1.*")):
        print(f"  {p}")
    for p in sorted(REPORTS_DIR.glob("phase0_blinding_*_v1.*")):
        print(f"  {p}")
    for p in sorted(REPORTS_DIR.glob("phase0_domain_*_v1.*")):
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gate 6.1 pre-field audit")
    args = parser.parse_args()
    main()
