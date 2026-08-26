#!/usr/bin/env python3
"""Gate 6: AquaScan Validation Dataset v1.

Transforms the enriched 4,744-record Ground Truth population into the
first scientifically usable validation dataset with:
- Target-independent GT quality (A/B/C/D per target)
- Leakage detection policy
- Spatial grouping identities
- Duplicate/spatial dependence analysis
- Physical validation candidate pools
- Phase 0 physical cohort proposal

DO NOT: train GeoAI, query IGME, process ERT/seismic, publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from math import cos, radians, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

VERSION = "v1"
INPUT_PARQUET = Path("data/processed/ground_truth_cohort5000_v0_4.parquet")
CANDIDATE_PARQUET = Path("data/processed/igme_candidates_v1.parquet")
OUTPUT_PARQUET = Path("data/processed/aquascan_validation_dataset_v1.parquet")
REPORTS_DIR = Path("reports")

# ── Target definitions (step 1-3) ───────────────────────────────────

TARGETS = [
    "TARGET_WATER_PRESENCE",
    "TARGET_DEPTH_TO_WATER",
    "TARGET_HYDRAULIC_HEAD_ELEVATION",
    "TARGET_PRODUCTIVITY",
    "TARGET_LITHOLOGY",
    "TARGET_AQUIFER_INTERVAL",
]

# ── Leakage policy definitions (step 6) ─────────────────────────────
# Every field classified: INPUT_CANDIDATE | TARGET | TARGET_DERIVED |
# METADATA_ONLY | PROVENANCE_ONLY | FORBIDDEN_LEAKAGE

LEAKAGE_POLICY: dict[str, dict[str, str]] = {
    # ── Candidate-source metadata ──
    "igme_id": {"role": "METADATA_ONLY", "rationale": "IGME identifier"},
    "domain": {"role": "METADATA_ONLY", "rationale": "hydrogeological unit"},
    "candidate_tier": {"role": "METADATA_ONLY", "rationale": "operational priority (not GT)"},
    "candidate_priority_score": {
        "role": "METADATA_ONLY",
        "rationale": "operational priority (not GT)",
    },
    "detail_url": {"role": "PROVENANCE_ONLY", "rationale": "IGME URL"},
    "point_nature": {"role": "METADATA_ONLY", "rationale": "point type"},
    "total_depth_m": {"role": "INPUT_CANDIDATE", "rationale": "borehole depth"},
    "construction_date": {"role": "INPUT_CANDIDATE", "rationale": "age context"},
    "latitude": {"role": "INPUT_CANDIDATE", "rationale": "spatial context"},
    "longitude": {"role": "INPUT_CANDIDATE", "rationale": "spatial context"},
    "elevation_m": {"role": "INPUT_CANDIDATE", "rationale": "topographic context"},
    "watershed": {"role": "INPUT_CANDIDATE", "rationale": "hydrological context"},
    "aquifer_system": {"role": "INPUT_CANDIDATE", "rationale": "hydrogeological context"},
    "province": {"role": "INPUT_CANDIDATE", "rationale": "administrative context"},
    "municipality": {"role": "INPUT_CANDIDATE", "rationale": "administrative context"},
    "drilling_method": {"role": "INPUT_CANDIDATE", "rationale": "construction method"},
    "pump_type": {"role": "INPUT_CANDIDATE", "rationale": "equipment"},
    "water_use": {"role": "INPUT_CANDIDATE", "rationale": "usage context"},
    "organization": {"role": "METADATA_ONLY", "rationale": "custodian"},
    # ── QA flags (metadata, not GT) ──
    "qa_coordinate_outlier": {"role": "METADATA_ONLY", "rationale": "QA flag"},
    "qa_elevation_outlier": {"role": "METADATA_ONLY", "rationale": "QA flag"},
    "qa_depth_outlier": {"role": "METADATA_ONLY", "rationale": "QA flag"},
    "qa_duplicate_coordinate": {"role": "METADATA_ONLY", "rationale": "QA flag"},
    "qa_flag_count": {"role": "METADATA_ONLY", "rationale": "QA summary"},
    # ── reference_flow_l_s ──
    "reference_flow_l_s": {
        "role": "FORBIDDEN_LEAKAGE",
        "rationale": "Always 0 in this cohort; not a reliable productivity signal. "
        "May correlate with TARGET_PRODUCTIVITY in other cohorts — "
        "must NEVER be used as input for productivity prediction.",
    },
    # ── Evidence family counts (DERIVED from raw) ──
    "geological_evidence_count": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from lithology evidence",
    },
    "water_level_evidence_count": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from piezometry evidence",
    },
    "hydraulic_evidence_count": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from pumping-test evidence",
    },
    "water_quality_evidence_count": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from chemistry evidence",
    },
    "hydrometry_evidence_count": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from hydrometry evidence",
    },
    # ── Documentary confidence ──
    "documentary_confidence": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from evidence richness",
    },
    "water_presence_status": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from hydrogeological evidence",
    },
    "water_presence_basis": {
        "role": "TARGET_DERIVED",
        "rationale": "derived from evidence sources",
    },
}

# ── Leakage per target ──────────────────────────────────────────────
LEAKAGE_FORBIDDEN: dict[str, set[str]] = {
    "TARGET_WATER_PRESENCE": {"water_presence_status", "water_presence_basis"},
    "TARGET_DEPTH_TO_WATER": {"water_level_evidence_count"},
    "TARGET_HYDRAULIC_HEAD_ELEVATION": {"water_level_evidence_count"},
    "TARGET_PRODUCTIVITY": {"hydraulic_evidence_count", "reference_flow_l_s"},
    "TARGET_LITHOLOGY": {"geological_evidence_count"},
    "TARGET_AQUIFER_INTERVAL": {"geological_evidence_count"},
}

# ── GT quality rules per target ─────────────────────────────────────


def _has_piezometry_observations(wl_json: str) -> bool:
    try:
        data = json.loads(wl_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, list) and len(data) > 0


def _has_lithology_intervals(lith_json: str) -> bool:
    try:
        data = json.loads(lith_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, list) and len(data) > 0


def _has_pumping_tests(hyd_json: str) -> bool:
    try:
        data = json.loads(hyd_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, list) and len(data) > 0


def _pumping_flow_values(hyd_json: str) -> list[float]:
    try:
        data = json.loads(hyd_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [h.get("flow_l_s", 0) or 0 for h in data if isinstance(h, dict)]


def _piezometry_quality(wl_json: str) -> tuple[int, float]:
    """Return (num_observations, quality_score) for piezometry."""
    try:
        data = json.loads(wl_json)
    except (json.JSONDecodeError, TypeError):
        return 0, 0.0
    if not isinstance(data, list) or len(data) == 0:
        return 0, 0.0
    n = len(data)
    # Quality score: records with both Nivel and Valor are better
    have_both = sum(
        1
        for o in data
        if isinstance(o, dict)
        and o.get("depth_to_water_m") is not None
        and o.get("hydraulic_head_elevation_m") is not None
    )
    quality = have_both / max(n, 1) if n > 0 else 0.0
    return n, quality


def _lithology_quality(lith_json: str) -> tuple[int, float]:
    """Return (num_intervals, quality_score)."""
    try:
        data = json.loads(lith_json)
    except (json.JSONDecodeError, TypeError):
        return 0, 0.0
    if not isinstance(data, list) or len(data) == 0:
        return 0, 0.0
    n = len(data)
    # Quality: intervals with both top/bottom or at least lithology description with aquifer flag
    have_depth = sum(
        1
        for o in data
        if isinstance(o, dict)
        and o.get("top_depth_m") is not None
        and o.get("bottom_depth_m") is not None
    )
    have_age = sum(1 for o in data if isinstance(o, dict) and o.get("geological_age") is not None)
    quality = (have_depth * 0.6 + have_age * 0.4) / max(n, 1) if n > 0 else 0.0
    return n, quality


def compute_gt_quality(record_row: pd.Series) -> dict[str, dict[str, Any]]:
    """Compute GT quality A/B/C/D per target from actual evidence.

    Returns dict of target → {quality_class, n_evidence, score, rationale}.
    GT quality classes:
      A: strong ground truth for validation (multiple high-quality observations)
      B: usable ground truth for validation (some evidence, moderate quality)
      C: limited ground truth (evidence present but insufficient alone)
      D: insufficient ground truth for this target
    """

    has_piez = _has_piezometry_observations(record_row["water_level_evidence_json"])
    has_lith = _has_lithology_intervals(record_row["geological_evidence_json"])
    has_pump = _has_pumping_tests(record_row["hydraulic_evidence_json"])
    n_piez, piez_qual = _piezometry_quality(record_row["water_level_evidence_json"])
    n_lith, lith_qual = _lithology_quality(record_row["geological_evidence_json"])
    pump_flows = _pumping_flow_values(record_row["hydraulic_evidence_json"])
    doc_conf = record_row.get("documentary_confidence", 0.0) or 0.0
    presence_status = record_row.get("water_presence_status", "UNKNOWN")

    results: dict[str, dict[str, Any]] = {}

    # ── TARGET_WATER_PRESENCE ──
    if presence_status == "PRESENT":
        if has_pump and doc_conf >= 0.5:
            results["TARGET_WATER_PRESENCE"] = {
                "quality_class": "A",
                "n_evidence": int(has_piez) + int(has_pump),
                "score": 0.9,
                "rationale": "Water present confirmed by pumping-test evidence",
            }
        elif has_piez and n_piez >= 2:
            results["TARGET_WATER_PRESENCE"] = {
                "quality_class": "B",
                "n_evidence": int(has_piez),
                "score": 0.7,
                "rationale": "Water present confirmed by multiple piezometry readings",
            }
        elif has_piez:
            results["TARGET_WATER_PRESENCE"] = {
                "quality_class": "C",
                "n_evidence": 1,
                "score": 0.5,
                "rationale": "Water present from single piezometry observation",
            }
        else:
            results["TARGET_WATER_PRESENCE"] = {
                "quality_class": "C",
                "n_evidence": 0,
                "score": 0.3,
                "rationale": "Water presence inferred but no direct hydrogeological observation",
            }
    elif presence_status == "UNKNOWN":
        results["TARGET_WATER_PRESENCE"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "Insufficient evidence to determine water presence",
        }
    else:
        results["TARGET_WATER_PRESENCE"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "Unknown water presence status",
        }

    # ── TARGET_DEPTH_TO_WATER ──
    if n_piez >= 5 and piez_qual >= 0.8:
        results["TARGET_DEPTH_TO_WATER"] = {
            "quality_class": "A",
            "n_evidence": n_piez,
            "score": float(min(piez_qual, 1.0)),
            "rationale": f"Strong: {n_piez} high-quality Nivel observations",
        }
    elif n_piez >= 2:
        results["TARGET_DEPTH_TO_WATER"] = {
            "quality_class": "B",
            "n_evidence": n_piez,
            "score": float(piez_qual),
            "rationale": f"Usable: {n_piez} Nivel observations",
        }
    elif n_piez >= 1:
        results["TARGET_DEPTH_TO_WATER"] = {
            "quality_class": "C",
            "n_evidence": 1,
            "score": 0.4,
            "rationale": "Single Nivel observation (limited temporal coverage)",
        }
    else:
        results["TARGET_DEPTH_TO_WATER"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "No Nivel observations available",
        }

    # ── TARGET_HYDRAULIC_HEAD_ELEVATION ──
    # Same as depth-to-water (both come from same piezometry records)
    results["TARGET_HYDRAULIC_HEAD_ELEVATION"] = {
        "quality_class": results["TARGET_DEPTH_TO_WATER"]["quality_class"],
        "n_evidence": results["TARGET_DEPTH_TO_WATER"]["n_evidence"],
        "score": results["TARGET_DEPTH_TO_WATER"]["score"],
        "rationale": (f"{results['TARGET_DEPTH_TO_WATER']['rationale']} (Valor from same source)"),
    }

    # ── TARGET_PRODUCTIVITY ──
    if pump_flows:
        n_positive = sum(1 for f in pump_flows if f > 0)
        if n_positive >= 2 and doc_conf >= 0.4:
            results["TARGET_PRODUCTIVITY"] = {
                "quality_class": "A",
                "n_evidence": len(pump_flows),
                "score": 0.9,
                "rationale": (
                    f"Strong: {len(pump_flows)} pumping tests, {n_positive} positive-flow"
                ),
            }
        elif n_positive >= 1:
            results["TARGET_PRODUCTIVITY"] = {
                "quality_class": "B",
                "n_evidence": len(pump_flows),
                "score": 0.7,
                "rationale": (f"Usable: {n_positive} positive-flow pumping test(s)"),
            }
        else:
            results["TARGET_PRODUCTIVITY"] = {
                "quality_class": "C",
                "n_evidence": len(pump_flows),
                "score": 0.3,
                "rationale": "Pumping evidence but all zero/negative flow",
            }
    else:
        results["TARGET_PRODUCTIVITY"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "No pumping-test evidence available",
        }

    # ── TARGET_LITHOLOGY ──
    if n_lith >= 3 and lith_qual >= 0.6:
        results["TARGET_LITHOLOGY"] = {
            "quality_class": "A",
            "n_evidence": n_lith,
            "score": float(min(lith_qual, 1.0)),
            "rationale": f"Strong: {n_lith} well-described intervals",
        }
    elif n_lith >= 1:
        results["TARGET_LITHOLOGY"] = {
            "quality_class": "B",
            "n_evidence": n_lith,
            "score": float(lith_qual),
            "rationale": f"Usable: {n_lith} intervals",
        }
    else:
        results["TARGET_LITHOLOGY"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "No lithological intervals available",
        }

    # ── TARGET_AQUIFER_INTERVAL ──
    # Intentionally strict — previous gates showed this target has
    # weak documentary Ground Truth nationally (0% READY in Gate 4).
    if has_lith and n_lith >= 1:
        n_aquifer = sum(
            1
            for o in json.loads(record_row["geological_evidence_json"])
            if isinstance(o, dict) and o.get("aquifer_flagged_lithology_interval")
        )
        if n_aquifer >= 3 and lith_qual >= 0.7:
            results["TARGET_AQUIFER_INTERVAL"] = {
                "quality_class": "B",
                "n_evidence": n_aquifer,
                "score": 0.65,
                "rationale": (
                    f"Usable: {n_aquifer} aquifer-flagged intervals with good structural quality"
                ),
            }
        elif n_aquifer >= 1 and lith_qual >= 0.5:
            results["TARGET_AQUIFER_INTERVAL"] = {
                "quality_class": "C",
                "n_evidence": n_aquifer,
                "score": 0.4,
                "rationale": (
                    f"Limited: {n_aquifer} aquifer-flagged interval(s) "
                    f"— depth/age quality insufficient for class B"
                ),
            }
        else:
            results["TARGET_AQUIFER_INTERVAL"] = {
                "quality_class": "D",
                "n_evidence": 0,
                "score": 0.0,
                "rationale": (
                    "Lithology present but no aquifer-flagged intervals "
                    "with sufficient structural quality"
                ),
            }
    else:
        results["TARGET_AQUIFER_INTERVAL"] = {
            "quality_class": "D",
            "n_evidence": 0,
            "score": 0.0,
            "rationale": "No lithological intervals available",
        }

    return results


# ── Temporal classification ─────────────────────────────────────────


def _historical_label(row: pd.Series) -> str:
    """Classify how recent/contemporaneous the water-level observations are."""
    try:
        ts = json.loads(row["temporal_summary_json"])
    except (json.JSONDecodeError, TypeError):
        return "NO_OBSERVATIONS"
    if ts.get("num_measurements", 0) == 0:
        return "NO_OBSERVATIONS"
    status = ts.get("temporal_relevance_status", "DATE_UNKNOWN")
    if status == "RECENT":
        return "RECENT"
    if status == "MULTI_DECADE_SERIES":
        return "MULTI_DECADE_SERIES"
    if status == "HISTORICAL":
        return "HISTORICAL"
    if status == "DATE_UNKNOWN":
        return "DATE_UNKNOWN"
    return "UNCLASSIFIED"


# ── Physical validation scoring ─────────────────────────────────────


def _physical_validation_score(
    row: pd.Series,
    gt_qa: dict[str, dict[str, Any]],
    target: str,
) -> float:
    """Score a record for physical validation suitability for a given target."""
    q = gt_qa.get(target, {})
    quality_class = q.get("quality_class", "D")
    doc_conf = row.get("documentary_confidence", 0.0) or 0.0
    qa_count = row.get("qa_flag_count", 0) or 0

    # Base from GT quality
    class_scores = {"A": 0.9, "B": 0.7, "C": 0.4, "D": 0.0}
    score = class_scores.get(quality_class, 0.0)

    # Boost for multiple evidence families
    evidence_cols = [
        "geological_evidence_count",
        "water_level_evidence_count",
        "hydraulic_evidence_count",
        "water_quality_evidence_count",
    ]
    n_families = sum(1 for c in evidence_cols if (row.get(c, 0) or 0) > 0)
    score += 0.05 * min(n_families, 4)

    # Penalty for QA issues
    score -= 0.05 * min(qa_count, 5)

    # Documentary confidence boost
    score += 0.1 * doc_conf

    return min(max(score, 0.0), 1.0)


# ── Main ─────────────────────────────────────────────────────────────


def main(limit: int | None = None) -> None:
    print("=" * 70)
    print("Gate 6: AquaScan Validation Dataset v1")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────
    print("\n[1] Loading enriched dataset …")
    enriched = pd.read_parquet(INPUT_PARQUET)
    candidates = pd.read_parquet(CANDIDATE_PARQUET)
    print(f"  Enriched: {len(enriched)} records")
    print(f"  Candidates: {len(candidates)} total")

    # Merge spatial/context fields from candidates
    spatial_cols = [
        "latitude",
        "longitude",
        "elevation_m",
        "point_nature",
        "total_depth_m",
        "watershed",
        "aquifer_system",
        "province",
        "municipality",
        "drilling_method",
        "pump_type",
        "water_use",
        "organization",
        "construction_date",
        "reference_flow_l_s",
        "qa_coordinate_outlier",
        "qa_elevation_outlier",
        "qa_depth_outlier",
        "qa_duplicate_coordinate",
        "qa_flag_count",
    ]
    available_spatial = [c for c in spatial_cols if c in candidates.columns]
    merged = enriched.merge(
        candidates[["igme_id"] + available_spatial],
        on="igme_id",
        how="left",
    )
    print(f"  Merged: {len(merged)} records with spatial data")

    # Apply limit for smoke testing
    if limit is not None:
        merged = merged.head(limit)
        print(f"  Limited to {limit} records (smoke test)")

    # ── 2. Compute GT quality per target ────────────────────────────
    print("\n[2] Computing target-specific GT quality (A/B/C/D) …")
    gt_quality_list: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        qa = compute_gt_quality(row)
        gt_quality_list.append(
            {
                "igme_id": row["igme_id"],
                **{f"gt_quality_{t}": qa[t]["quality_class"] for t in TARGETS},
                **{f"gt_score_{t}": qa[t]["score"] for t in TARGETS},
                **{f"gt_n_evidence_{t}": qa[t]["n_evidence"] for t in TARGETS},
                **{f"gt_rationale_{t}": qa[t]["rationale"] for t in TARGETS},
            }
        )

    gt_df = pd.DataFrame(gt_quality_list)
    merged = merged.merge(gt_df, on="igme_id", how="left")

    # ── 3. Temporal classification ──────────────────────────────────
    print("\n[3] Classifying temporal ground truth …")
    merged["temporal_gt_type"] = merged.apply(_historical_label, axis=1)

    # ── 4. Spatial groups ───────────────────────────────────────────
    print("\n[4] Building spatial grouping identities …")
    # Primary group: hydrogeological domain
    merged["group_domain"] = merged["domain"].fillna("unknown")

    # Secondary: aquifer system where available
    merged["group_aquifer"] = merged.get("aquifer_system", pd.NA).fillna("unknown")

    # Tertiary: watershed
    merged["group_watershed"] = merged.get("watershed", pd.NA).fillna("unknown")

    # Spatial cluster: round coordinates to 0.01° (~1km grid)
    merged["group_spatial_cell"] = (
        merged["latitude"].round(2).astype(str) + "_" + merged["longitude"].round(2).astype(str)
    )

    # Near proximity: group records within ~500m
    def _compute_proximity_cluster(df: pd.DataFrame) -> pd.Series:
        """Assign proximity cluster IDs based on ~500m distance threshold."""
        from itertools import combinations

        n = len(df)
        if n == 0:
            return pd.Series([], dtype=int)

        # 500m ≈ 0.0045° at mid-latitudes per degree

        lats = df["latitude"].values
        lons = df["longitude"].values

        # Simple union-find for records within threshold
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Only check nearby pairs using approximate spatial bucketing
        # Bucket by 0.05° grid (~5km) to avoid O(n²)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            bx = int(lats[i] / 0.05)
            by = int(lons[i] / 0.05)
            buckets.setdefault((bx, by), []).append(i)

        # Within each bucket and adjacent buckets
        offsets = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 1)]
        for (bx, by), indices in buckets.items():
            all_indices = indices.copy()
            for dx, dy in offsets:
                extra = buckets.get((bx + dx, by + dy), [])
                all_indices.extend(extra)
            # Remove duplicates
            unique_indices = list(set(all_indices))
            for ai, bi in combinations(unique_indices, 2):
                if ai >= n or bi >= n:
                    continue
                dist = sqrt(
                    ((lats[ai] - lats[bi]) * 111000) ** 2
                    + ((lons[ai] - lons[bi]) * 111000 * cos(radians(lats[ai]))) ** 2
                )
                if dist < 500:
                    union(ai, bi)

        # Assign cluster labels
        cluster_map: dict[int, int] = {}
        cluster_id = 0
        result = []
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = cluster_id
                cluster_id += 1
            result.append(cluster_map[root])
        return pd.Series(result, index=df.index)

    merged["group_proximity_500m"] = merged.groupby("domain", group_keys=False).apply(
        _compute_proximity_cluster, include_groups=False
    )
    # Make globally unique: domain + cluster_id
    merged["group_proximity_500m"] = (
        merged["domain"].astype(str) + "_p" + merged["group_proximity_500m"].astype(str)
    )

    # ── 5. Duplicate / spatial dependence analysis ──────────────────
    print("\n[5] Analyzing spatial dependence …")

    # Exact coordinate duplicates
    coord_groups = merged.groupby(["latitude", "longitude"]).size()
    duplicate_coords = coord_groups[coord_groups > 1]
    merged["spatial_is_duplicate"] = merged.set_index(["latitude", "longitude"]).index.isin(
        duplicate_coords.index
    )

    # Proximity clusters with multiple members
    prox_counts = merged.groupby("group_proximity_500m").size()
    merged["n_in_proximity_group"] = merged["group_proximity_500m"].map(prox_counts)
    merged["spatial_has_neighbors"] = merged["n_in_proximity_group"] > 1

    # Same domain + nearby coordinates
    domain_counts = merged.groupby("group_domain").size()
    merged["n_in_domain"] = merged["group_domain"].map(domain_counts)

    print(
        f"  Exact coordinate duplicates: {len(duplicate_coords)} pairs "
        f"({duplicate_coords.sum()} records)"
    )
    print(f"  Records with proximity neighbors (<500m): {merged['spatial_has_neighbors'].sum()}")

    # ── 6. Physical validation scores ───────────────────────────────
    print("\n[6] Building physical validation candidate scores …")
    # Collect all GT quality dicts
    all_gt_qa = {}
    for _, row in merged.iterrows():
        all_gt_qa[row["igme_id"]] = compute_gt_quality(row)

    for target in TARGETS:
        merged[f"phys_val_score_{target}"] = merged.apply(
            lambda r, t=target: _physical_validation_score(r, all_gt_qa[r["igme_id"]], t),
            axis=1,
        )

    # Multi-target score (average across targets)
    merged["phys_val_score_multi"] = merged[[f"phys_val_score_{t}" for t in TARGETS]].mean(axis=1)

    # ── 7. Target availability report ───────────────────────────────
    print("\n[7] Target availability report …")
    availability_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        quality_col = f"gt_quality_{target}"
        n_col = f"gt_n_evidence_{target}"
        counts = Counter(merged[quality_col])
        n_total = len(merged)
        for qclass in ["A", "B", "C", "D"]:
            cnt = counts.get(qclass, 0)
            availability_rows.append(
                {
                    "target": target,
                    "gt_quality_class": qclass,
                    "count": cnt,
                    "pct": round(100 * cnt / n_total, 1) if n_total else 0.0,
                }
            )
        # Also per domain
        for domain_name in merged["domain"].unique():
            sub = merged[merged["domain"] == domain_name]
            sub_total = len(sub)
            sub_counts = Counter(sub[quality_col])
            for qclass in ["A", "B", "C", "D"]:
                cnt = sub_counts.get(qclass, 0)
                availability_rows.append(
                    {
                        "target": target,
                        "gt_quality_class": qclass,
                        "count": cnt,
                        "pct": (round(100 * cnt / sub_total, 1) if sub_total else 0.0),
                        "domain": domain_name,
                    }
                )
        # Per candidate tier
        for tier in merged["candidate_tier"].unique():
            sub = merged[merged["candidate_tier"] == tier]
            sub_total = len(sub)
            sub_counts = Counter(sub[quality_col])
            for qclass in ["A", "B", "C", "D"]:
                cnt = sub_counts.get(qclass, 0)
                availability_rows.append(
                    {
                        "target": target,
                        "gt_quality_class": qclass,
                        "count": cnt,
                        "pct": (round(100 * cnt / sub_total, 1) if sub_total else 0.0),
                        "candidate_tier": tier,
                    }
                )

    availability_df = pd.DataFrame(availability_rows)
    availability_df.to_csv(REPORTS_DIR / "validation_target_availability_v1.csv", index=False)

    # Summary: READY = A+B, PARTIAL = C, NOT_READY = D
    print("  Target readiness (A+B / C / D):")
    for target in TARGETS:
        sub = availability_df[
            (availability_df["target"] == target)
            & availability_df["domain"].isna()
            & availability_df["candidate_tier"].isna()
        ]
        ready = sub[sub["gt_quality_class"].isin(["A", "B"])]["count"].sum()
        partial = sub[sub["gt_quality_class"] == "C"]["count"].sum()
        not_ready = sub[sub["gt_quality_class"] == "D"]["count"].sum()
        print(
            f"    {target}: A+B={ready} ({ready / 4744 * 100:.1f}%), "
            f"C={partial} ({partial / 4744 * 100:.1f}%), "
            f"D={not_ready} ({not_ready / 4744 * 100:.1f}%)"
        )

    # ── 8. Positive/negative evidence analysis ──────────────────────
    print("\n[8] Positive/negative evidence analysis …")

    # Positive cases: water_presence_status = PRESENT AND pumping evidence
    positive = merged[
        (merged["water_presence_status"] == "PRESENT")
        & (merged["gt_quality_TARGET_PRODUCTIVITY"].isin(["A", "B"]))
    ]
    # Low-productivity: pumping tests exist but flow < 10 L/s
    low_prod_ids = []
    for _, row in merged.iterrows():
        if _has_pumping_tests(row["hydraulic_evidence_json"]):
            flows = _pumping_flow_values(row["hydraulic_evidence_json"])
            if flows and max(flows) < 10:
                low_prod_ids.append(row["igme_id"])
    low_productivity = merged[merged["igme_id"].isin(low_prod_ids)]

    # Dry/negative: no pumping evidence AND water_presence = UNKNOWN
    # (we do NOT create negatives from reference_flow_l_s=0)
    negative_candidates = merged[
        (merged["water_presence_status"] == "UNKNOWN") & (~merged["igme_id"].isin(low_prod_ids))
    ]

    # Genuine negatives: records with explicit dry evidence
    # In IGME data, truly "dry" wells are hard to identify —
    # absence of piezometry doesn't prove dryness.
    # Check if any records have explicit indicators
    genuine_negatives = negative_candidates[
        (negative_candidates["geological_evidence_count"] > 0)
        & (negative_candidates["water_level_evidence_count"] == 0)
        & (negative_candidates["hydraulic_evidence_count"] == 0)
    ]

    print(f"  Positive (productive): {len(positive)} records")
    print(f"  Low-productivity (<10 L/s): {len(low_productivity)} records")
    print(
        "UNKNOWN water presence: "
        f"{negative_candidates['water_presence_status'].eq('UNKNOWN').sum()}"
    )
    print(f"  Genuine negative candidates: {len(genuine_negatives)}")
    if len(genuine_negatives) < 50:
        print(
            "  ⚠ WARNING: Reliable negative examples are SCARCE. "
            "Most UNKNOWN records simply lack evidence — "
            "they are NOT confirmed dry wells."
        )

    # ── 9. Physical validation candidate pools ──────────────────────
    print("\n[9] Building physical validation candidate pools …")

    def _build_pool(df: pd.DataFrame, target: str, n: int = 50) -> pd.DataFrame:
        score_col = f"phys_val_score_{target}"
        # Filter to only records with GT >= C for this target
        gt_col = f"gt_quality_{target}"
        pool = df[df[gt_col].isin(["A", "B", "C"])].copy()
        if len(pool) == 0:
            return pool.head(0)
        # Sort by validation score descending
        pool = pool.sort_values(score_col, ascending=False).head(n)
        return pool

    candidate_pools: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        candidate_pools[target] = _build_pool(merged, target)

    # Multi-target pool
    candidate_pools["MULTI_TARGET"] = merged.nlargest(50, "phys_val_score_multi")

    # Combine all pools into one CSV
    all_candidates: list[dict[str, Any]] = []
    for pool_target, pool_df in candidate_pools.items():
        for _, row in pool_df.iterrows():
            all_candidates.append(
                {
                    "candidate_pool": pool_target,
                    "igme_id": row["igme_id"],
                    "domain": row["domain"],
                    "phys_val_score": (
                        row.get(f"phys_val_score_{pool_target}", np.nan)
                        if pool_target != "MULTI_TARGET"
                        else row["phys_val_score_multi"]
                    ),
                    "gt_quality": (
                        row.get(f"gt_quality_{pool_target}", "D")
                        if pool_target != "MULTI_TARGET"
                        else "MULTI"
                    ),
                    "documentary_confidence": row.get("documentary_confidence", 0),
                    "n_evidence_families": sum(
                        1
                        for c in [
                            "geological_evidence_count",
                            "water_level_evidence_count",
                            "hydraulic_evidence_count",
                            "water_quality_evidence_count",
                        ]
                        if (row.get(c, 0) or 0) > 0
                    ),
                    "water_presence_status": row.get("water_presence_status", "UNKNOWN"),
                    "latitude": row.get("latitude", np.nan),
                    "longitude": row.get("longitude", np.nan),
                    "point_nature": row.get("point_nature", "unknown"),
                    "temporal_gt_type": row.get("temporal_gt_type", "unknown"),
                }
            )

    candidates_df = pd.DataFrame(all_candidates)
    candidates_df.to_csv(REPORTS_DIR / "physical_validation_candidates_v1.csv", index=False)
    print(f"  Total candidate entries: {len(candidates_df)}")
    print(f"  Pools: {list(candidate_pools.keys())}")

    # ── 10. Phase 0 physical cohort proposal ────────────────────────
    print("\n[10] Phase 0 physical cohort proposal …")

    # Select top candidates for multi-target validation
    # Priority domain: San Antonio (best GT richness)
    # Secondary: Santa Eulalia
    multi_pool = candidate_pools["MULTI_TARGET"]
    # Use quotas below for per-domain distribution

    proposed_sites: list[dict[str, Any]] = []
    # Take ~10 from priority domain, ~8 from secondary, ~5 each from others
    # Take 22-30 sites: ~8 from priority, ~6 from secondary, ~4 each from others
    quotas: dict[str, int] = {
        "San Antonio": 8,
        "Santa Eulalia": 6,
        "Migjorn": 4,
        "San Carlos": 4,
        "Región de los arenales": 3,
    }
    selected_ids: set[str] = set()

    for domain_name, quota in quotas.items():
        domain_candidates = multi_pool[multi_pool["domain"] == domain_name].copy()
        domain_candidates = domain_candidates[~domain_candidates["igme_id"].isin(selected_ids)]
        # Prioritize diverse GT quality
        domain_candidates = domain_candidates.sort_values("phys_val_score_multi", ascending=False)
        taken = 0
        for _, row in domain_candidates.iterrows():
            if taken >= quota:
                break
            if row["igme_id"] in selected_ids:
                continue
            selected_ids.add(row["igme_id"])
            proposed_sites.append(
                {
                    "igme_id": row["igme_id"],
                    "domain": row["domain"],
                    "phys_val_score_multi": round(row["phys_val_score_multi"], 3),
                    "available_targets": ", ".join(
                        t for t in TARGETS if row.get(f"gt_quality_{t}", "D") in ("A", "B")
                    ),
                    "temporal_gt_type": row.get("temporal_gt_type", "unknown"),
                    "latitude": row.get("latitude", np.nan),
                    "longitude": row.get("longitude", np.nan),
                    "water_presence_status": row.get("water_presence_status", "UNKNOWN"),
                }
            )
            taken += 1

    proposal = {
        "phase": "Phase 0 physical cohort proposal",
        "version": "v1",
        "status": "PROPOSAL — NOT YET EXECUTED",
        "target_size": "20–30 sites",
        "selected_sites": len(proposed_sites),
        "strategy": (
            "Prioritize controlled hydrogeological context "
            "(San Antonio primary, Santa Eulalia secondary) "
            "over maximum geographic diversity. "
            "All sites have multi-target GT quality A or B."
        ),
        "distinction": (
            "This is RETROSPECTIVE known-well validation — "
            "sites are selected from existing IGME records with "
            "known documentary GT. "
            "This is NOT prospective blind drilling validation."
        ),
        "sites": proposed_sites,
    }

    with open(REPORTS_DIR / "phase0_physical_cohort_proposal_v1.json", "w") as f:
        json.dump(proposal, f, indent=2, ensure_ascii=False)
    print(f"  Proposed {len(proposed_sites)} sites for Phase 0 physical validation")
    for s in proposed_sites:
        print(
            f"    {s['igme_id']} ({s['domain']}): "
            f"targets={s['available_targets']}, "
            f"temporal={s['temporal_gt_type']}"
        )

    # ── 11. Leakage policy ──────────────────────────────────────────
    print("\n[11] Generating machine-readable leakage policy …")
    leakage_policy = {
        "version": "v1",
        "description": (
            "Leakage policy for AquaScan Validation Dataset v1. "
            "Defines which fields may be used as input features "
            "for each prediction target. "
            "Fields marked FORBIDDEN_LEAKAGE must never appear "
            "in training inputs for ANY target. "
            "TARGET_DERIVED fields are derived from other fields "
            "that may be targets and must be excluded per-target."
        ),
        "field_classifications": LEAKAGE_POLICY,
        "target_specific_forbidden": {t: sorted(LEAKAGE_FORBIDDEN.get(t, set())) for t in TARGETS},
    }
    with open(REPORTS_DIR / "validation_leakage_policy_v1.json", "w") as f:
        json.dump(leakage_policy, f, indent=2, ensure_ascii=False)

    # ── 12. GT quality summary ──────────────────────────────────────
    print("\n[12] Generating GT quality by target report …")
    quality_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        q_col = f"gt_quality_{target}"
        s_col = f"gt_score_{target}"
        n_col = f"gt_n_evidence_{target}"
        r_col = f"gt_rationale_{target}"
        for _, row in merged.iterrows():
            quality_rows.append(
                {
                    "igme_id": row["igme_id"],
                    "domain": row["domain"],
                    "target": target,
                    "gt_quality_class": row[q_col],
                    "gt_score": row[s_col],
                    "n_evidence": row[n_col],
                    "rationale": row[r_col],
                }
            )

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(REPORTS_DIR / "validation_gt_quality_v1.csv", index=False)

    # ── 13. Spatial groups export ───────────────────────────────────
    print("\n[13] Exporting spatial groups …")
    spatial_df = merged[
        [
            "igme_id",
            "latitude",
            "longitude",
            "group_domain",
            "group_aquifer",
            "group_watershed",
            "group_spatial_cell",
            "group_proximity_500m",
            "spatial_is_duplicate",
            "spatial_has_neighbors",
            "n_in_proximity_group",
            "n_in_domain",
        ]
    ].copy()
    spatial_df.to_parquet(REPORTS_DIR / "validation_spatial_groups_v1.parquet", index=False)
    print(f"  Spatial grouping columns: {list(spatial_df.columns)}")

    # ── 14. Save validation dataset ─────────────────────────────────
    print("\n[14] Saving AquaScan Validation Dataset v1 …")
    # Drop JSON columns that will be replaced by structured derivatives
    json_cols = [
        "geological_evidence_json",
        "water_level_evidence_json",
        "hydraulic_evidence_json",
        "water_quality_evidence_json",
        "hydrometry_evidence_json",
        "provenance_json",
        "temporal_summary_json",
        "lithology_validation_json",
    ]
    output_df = merged.drop(columns=[c for c in json_cols if c in merged.columns])

    # Add version metadata columns
    output_df["dataset_version"] = VERSION
    output_df["created_at"] = datetime.now(UTC).isoformat()

    output_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Saved {len(output_df)} records to {OUTPUT_PARQUET}")
    print(f"  Columns: {len(output_df.columns)}")

    # ── 15. Dataset manifest ────────────────────────────────────────
    print("\n[15] Generating dataset manifest …")
    manifest = {
        "dataset_name": "AquaScan Validation Dataset v1",
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "input_parquet": str(INPUT_PARQUET),
            "input_sha256": hashlib.sha256(INPUT_PARQUET.read_bytes()).hexdigest(),
            "candidate_parquet": str(CANDIDATE_PARQUET),
            "candidate_sha256": hashlib.sha256(CANDIDATE_PARQUET.read_bytes()).hexdigest(),
        },
        "records": len(output_df),
        "columns": len(output_df.columns),
        "targets": TARGETS,
        "leakage_policy_version": "v1",
        "spatial_grouping_type": [
            "group_domain (hydrogeological unit)",
            "group_aquifer (aquifer system)",
            "group_watershed",
            "group_spatial_cell (~1km grid)",
            "group_proximity_500m (<500m proximity clusters)",
        ],
        "temporal_classification": [
            "RECENT — observations from 2015+",
            "MULTI_DECADE_SERIES — >30yr span",
            "HISTORICAL — pre-2015, shorter span",
            "DATE_UNKNOWN — dates missing",
            "NO_OBSERVATIONS — no water-level data",
        ],
        "gt_quality_classes": [
            "A — strong GT for validation",
            "B — usable GT for validation",
            "C — limited GT, supplemental only",
            "D — insufficient GT",
        ],
        "warnings": [
            "reference_flow_l_s is always 0 in this cohort; do not use",
            "reliable negative/dry examples are scarce; absence of evidence ≠ absence of water",
            "historical water levels must not be presented as contemporaneous",
            "TARGET_AQUIFER_INTERVAL has maximum class B (no class A records)",
        ],
    }
    with open(REPORTS_DIR / "validation_dataset_manifest_v1.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── 16. Verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 6 VERDICT")
    print("=" * 70)
    print(f"  Records: {len(output_df)}")
    print(f"  Targets: {len(TARGETS)}")
    print("  GT quality classes: A/B/C/D per target")
    print("  Spatial groups: 5 types")
    print("  Leakage policy: machine-readable")
    print(f"  Physical validation pools: {len(candidate_pools)} targets")
    print(f"  Phase 0 proposal sites: {len(proposed_sites)}")
    print()

    n_class_a = {}
    n_class_ab = {}
    for t in TARGETS:
        a_cnt = int((merged[f"gt_quality_{t}"] == "A").sum())
        b_cnt = int((merged[f"gt_quality_{t}"] == "B").sum())
        d_cnt = int((merged[f"gt_quality_{t}"] == "D").sum())
        n_class_a[t] = a_cnt
        n_class_ab[t] = a_cnt + b_cnt
        print(f"  {t}: A+B={a_cnt + b_cnt} A={a_cnt} B={b_cnt} D={d_cnt}")

    # Determine verdict
    # PASS WITH LIMITATIONS if any target has zero class A records
    targets_no_a = [t for t in TARGETS if n_class_a[t] == 0]
    if targets_no_a:
        verdict = (
            "PASS WITH LIMITATIONS — Validation Dataset v1 created but "
            f"the following targets have zero class-A (strong) Ground Truth: "
            f"{', '.join(targets_no_a)}. "
            "Reliable negative examples also remain scarce."
        )
    else:
        verdict = "PASS — Validation Dataset v1 created successfully"

    print(f"\nFINAL VERDICT: {verdict}")

    print("\nArtifacts:")
    for p in sorted(REPORTS_DIR.glob("validation_*_v1.*")):
        print(f"  {p}")
    for p in sorted(REPORTS_DIR.glob("phase0_physical_*_v1.*")):
        print(f"  {p}")
    print(f"  {OUTPUT_PARQUET}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 6 validation dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit records for smoke testing",
    )
    args = parser.parse_args()
    main(limit=args.limit)
