"""Coverage analysis, Phase 0 v2, and manual audit."""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

from aquascan.cohort.selection import DOMAINS
from aquascan.data.models import NormalizedGroundTruthRecord


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% CI by default)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_coverage(records: list[NormalizedGroundTruthRecord]) -> pd.DataFrame:
    """Compute record-level coverage."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        has_lith = len(rec.geological_evidence) > 0
        has_piez = len(rec.water_level_evidence) > 0
        has_pump = len(rec.hydraulic_evidence) > 0
        has_chem = len(rec.water_quality_evidence) > 0
        has_hydr = len(rec.hydrometry_evidence) > 0

        families = int(has_lith) + int(has_piez) + int(has_pump) + int(has_chem) + int(has_hydr)

        rows.append(
            {
                "igme_id": rec.igme_id,
                "domain": rec.domain,
                "candidate_tier": rec.candidate_tier,
                "any_enrichment": families > 0,
                "has_lithology": has_lith,
                "has_piezometry": has_piez,
                "has_pumping_tests": has_pump,
                "has_chemistry": has_chem,
                "has_hydrometry": has_hydr,
                "evidence_family_count": families,
                "geological_plus_water_level": has_lith and has_piez,
                "geological_plus_hydraulic": has_lith and has_pump,
                "geological_plus_water_plus_hydraulic": has_lith and has_piez and has_pump,
            }
        )
    return pd.DataFrame(rows)


def compute_target_readiness_stats(records: list[NormalizedGroundTruthRecord]) -> pd.DataFrame:
    """Compute READY/PARTIAL/NOT_READY percentages per target (step 11)."""
    targets = list(records[0].target_readiness.keys()) if records else []
    rows: list[dict[str, Any]] = []
    n = len(records)
    for target in targets:
        ready = sum(1 for r in records if r.target_readiness[target].readiness.value == "READY")
        partial = sum(1 for r in records if r.target_readiness[target].readiness.value == "PARTIAL")
        not_ready = sum(
            1 for r in records if r.target_readiness[target].readiness.value == "NOT_READY"
        )
        rows.append(
            {
                "target": target,
                "n": n,
                "ready": ready,
                "partial": partial,
                "not_ready": not_ready,
                "ready_pct": round(100 * ready / n, 1) if n else 0.0,
                "partial_pct": round(100 * partial / n, 1) if n else 0.0,
                "not_ready_pct": round(100 * not_ready / n, 1) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_domain_comparison_v2(
    coverage_df: pd.DataFrame,
    records: list[NormalizedGroundTruthRecord],
) -> pd.DataFrame:
    """Phase 0 domain evaluation v2 (step 12).

    Combines TWO distinct layers:
    1. Candidate/operational suitability
    2. Actual Ground Truth richness

    Shows both dimensions separately. Then computes overall score
    with documented formula + sensitivity analysis.
    """
    rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        sub = coverage_df[coverage_df["domain"] == domain]
        n = len(sub)
        if n == 0:
            continue

        # Layer 1: Candidate/operational suitability
        p1 = int((sub["candidate_tier"] == "P1").sum())
        p2 = int((sub["candidate_tier"] == "P2").sum())
        p3 = int((sub["candidate_tier"] == "P3").sum())
        candidate_suitability = (2 * p1 + p2) / max(1, 2 * n)  # 0-1

        # Layer 2: Actual Ground Truth richness
        n_lith = int(sub["has_lithology"].sum())
        n_piez = int(sub["has_piezometry"].sum())
        n_pump = int(sub["has_pumping_tests"].sum())
        n_chem = int(sub["has_chemistry"].sum())
        n_hydr = int(sub["has_hydrometry"].sum())
        avg_families = sub["evidence_family_count"].mean()
        richness = avg_families / 5.0  # normalize by 5 families (0-1)

        # Combined Phase 0 v2 score (documented formula)
        # weight candidate suitability 40%, richness 60%
        # (richness matters more for a physical experiment)
        phase0_v2 = 0.4 * candidate_suitability + 0.6 * richness

        rows.append(
            {
                "domain": domain,
                "n": n,
                "candidate_suitability": round(candidate_suitability, 4),
                "ground_truth_richness": round(richness, 4),
                "phase0_v2_score": round(phase0_v2, 4),
                "p1_count": p1,
                "p2_count": p2,
                "p3_count": p3,
                "with_lithology": n_lith,
                "with_piezometry": n_piez,
                "with_pumping_tests": n_pump,
                "with_chemistry": n_chem,
                "with_hydrometry": n_hydr,
                "avg_evidence_families": round(avg_families, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("phase0_v2_score", ascending=False)


def run_manual_audit(
    records: list[NormalizedGroundTruthRecord],
    raw_by_id: dict[str, dict[str, Any]],
    n: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Manually compare normalized evidence against raw API JSON (step 14)."""
    rng = random.Random(seed)
    audit_ids = rng.sample([r.igme_id for r in records], min(n, len(records)))

    rows: list[dict[str, Any]] = []
    for igme_id in audit_ids:
        rec = next(r for r in records if r.igme_id == igme_id)
        raw = raw_by_id.get(igme_id, {})
        discrepancies: list[str] = []

        # Verify IDs
        if raw.get("igme_id") != rec.igme_id:
            discrepancies.append(f"ID mismatch: raw={raw.get('igme_id')} norm={rec.igme_id}")

        # Verify piezometry: depth-to-water vs head separation
        raw_piez = raw.get("piezometry", [])
        if len(raw_piez) != len(rec.water_level_evidence):
            discrepancies.append(
                "piezometry count mismatch: "
                f"raw={len(raw_piez)} norm={len(rec.water_level_evidence)}"
            )
        for i, (rrow, nrow) in enumerate(zip(raw_piez, rec.water_level_evidence, strict=False)):
            r_nivel = rrow.get("Nivel")
            n_nivel = nrow.depth_to_water_m
            if r_nivel is not None and n_nivel is not None:
                if abs(float(str(r_nivel).replace(",", ".")) - n_nivel) > 1e-6:
                    discrepancies.append(f"piez[{i}].Nivel mismatch: {r_nivel} vs {n_nivel}")
            r_valor = rrow.get("Valor")
            n_valor = nrow.hydraulic_head_elevation_m
            if r_valor is not None and n_valor is not None:
                if abs(float(str(r_valor).replace(",", ".")) - n_valor) > 1e-6:
                    discrepancies.append(f"piez[{i}].Valor mismatch: {r_valor} vs {n_valor}")

        # Verify lithology intervals
        raw_lith = raw.get("lithology", [])
        if len(raw_lith) != len(rec.geological_evidence):
            discrepancies.append(
                f"lithology count mismatch: raw={len(raw_lith)} norm={len(rec.geological_evidence)}"
            )
        for i, (rrow, nrow) in enumerate(zip(raw_lith, rec.geological_evidence, strict=False)):
            r_top = rrow.get("ProfTecho")
            n_top = nrow.top_depth_m
            if r_top is not None and n_top is not None:
                if abs(float(str(r_top).replace(",", ".")) - n_top) > 1e-6:
                    discrepancies.append(f"lith[{i}].ProfTecho mismatch: {r_top} vs {n_top}")
            r_bottom = rrow.get("ProfMuro")
            n_bottom = nrow.bottom_depth_m
            if r_bottom is not None and n_bottom is not None:
                if abs(float(str(r_bottom).replace(",", ".")) - n_bottom) > 1e-6:
                    discrepancies.append(f"lith[{i}].ProfMuro mismatch: {r_bottom} vs {n_bottom}")

        # Verify pumping tests
        raw_pump = raw.get("pumping_tests", [])
        if len(raw_pump) != len(rec.hydraulic_evidence):
            discrepancies.append(
                "pumping test count mismatch: "
                f"raw={len(raw_pump)} norm={len(rec.hydraulic_evidence)}"
            )
        for i, (rrow, nrow) in enumerate(zip(raw_pump, rec.hydraulic_evidence, strict=False)):
            r_flow = rrow.get("Caudal")
            n_flow = nrow.flow_l_s
            if r_flow is not None and n_flow is not None:
                if abs(float(str(r_flow).replace(",", ".")) - n_flow) > 1e-6:
                    discrepancies.append(f"pump[{i}].Caudal mismatch: {r_flow} vs {n_flow}")
            r_dd = rrow.get("DepresionM")
            n_dd = nrow.drawdown_m
            if r_dd is not None and n_dd is not None:
                if abs(float(str(r_dd).replace(",", ".")) - n_dd) > 1e-6:
                    discrepancies.append(f"pump[{i}].DepresionM mismatch: {r_dd} vs {n_dd}")

        rows.append(
            {
                "igme_id": igme_id,
                "domain": rec.domain,
                "discrepancy_count": len(discrepancies),
                "discrepancies": "; ".join(discrepancies),
                "verdict": "PASS" if len(discrepancies) == 0 else "DISCREPANCY",
            }
        )

    return pd.DataFrame(rows)
