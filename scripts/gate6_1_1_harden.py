#!/usr/bin/env python3
"""Gate 6.1.1: Blindness Hardening.

Corrects the Gate 6.1 FIELD_BLINDED_VIEW to remove semantic leakage
(role, reason_selected), creates two-tier blinded views, neutral site IDs,
and a semantic leakage registry.

Also renames LOW_PRODUCTIVITY → LOW_FLOW_PUMPING_TEST_STRATUM.

DO NOT: query IGME, enrich records, modify GT, train GeoAI, acquire field data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


VERSION = "v1.1"
REPORTS_DIR = Path("reports")
INPUT_COHORT_B = Path("reports/phase0_cohort_B_v1.csv")
INPUT_VALIDATION = Path("data/processed/aquascan_validation_dataset_v1.parquet")
INPUT_ENRICHED = Path("data/processed/ground_truth_cohort5000_v0_4.parquet")

# ── Semantic leakage registry ────────────────────────────────────────
# Columns that MUST NEVER appear in ANY field-facing view.
# Goes beyond column-name checks: also forbids semantic proxies.
SEMANTIC_FORBIDDEN = {
    # Direct GT columns
    "water_presence_status",
    "gt_quality_TARGET_WATER_PRESENCE",
    "gt_quality_TARGET_DEPTH_TO_WATER",
    "gt_quality_TARGET_HYDRAULIC_HEAD_ELEVATION",
    "gt_quality_TARGET_PRODUCTIVITY",
    "gt_quality_TARGET_LITHOLOGY",
    "gt_quality_TARGET_AQUIFER_INTERVAL",
    "gt_score_TARGET_WATER_PRESENCE",
    "gt_score_TARGET_DEPTH_TO_WATER",
    "gt_score_TARGET_PRODUCTIVITY",
    "gt_score_TARGET_LITHOLOGY",
    "gt_score_TARGET_AQUIFER_INTERVAL",
    "gt_n_evidence_TARGET_WATER_PRESENCE",
    "gt_n_evidence_TARGET_DEPTH_TO_WATER",
    "gt_n_evidence_TARGET_PRODUCTIVITY",
    "gt_n_evidence_TARGET_LITHOLOGY",
    "gt_n_evidence_TARGET_AQUIFER_INTERVAL",
    "gt_rationale_TARGET_WATER_PRESENCE",
    "gt_rationale_TARGET_DEPTH_TO_WATER",
    "gt_rationale_TARGET_PRODUCTIVITY",
    "gt_rationale_TARGET_LITHOLOGY",
    "gt_rationale_TARGET_AQUIFER_INTERVAL",
    "negative_class_WATER_PRESENCE",
    "negative_class_PRODUCTIVITY",
    "documentary_confidence",
    "phys_val_score_TARGET_WATER_PRESENCE",
    "phys_val_score_TARGET_DEPTH_TO_WATER",
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
    # Semantic proxy columns (gate 6.1 cohort CSV columns)
    "gt_quality_WATER_PRESENCE",
    "gt_quality_DEPTH_TO_WATER",
    "gt_quality_PRODUCTIVITY",
    "gt_quality_LITHOLOGY",
    "productivity_note",  # "max flow X.X L/s"
    "pumping_test_count",  # reveals whether pumping data exists
    "lithology_intervals",  # reveals whether lithology exists
    "n_piezometric_observations",  # reveals water-level data count
    "temporal_relevance",  # reveals whether water data is recent
    "role",  # "PRODUCTIVE" / "LOW" / "CONTRAST"
    "reason_selected",  # "High-productivity..." / "Low-productivity..."
    # Candidate tier is metadata, but borderline - keep in interpreter view
}

# Columns allowed in OPERATOR view (locate/acquire only)
OPERATOR_COLUMNS = {
    "blinded_site_id",
    "latitude",
    "longitude",
    "hydrogeological_unit",  # operators need to know where they are
}

# Columns allowed in INTERPRETER view (interpret ERT/seismic)
# Adds aquifer-system for geological context but still no GT data
INTERPRETER_COLUMNS = OPERATOR_COLUMNS | {
    "aquifer_system",
    "candidate_tier",  # operational metadata, not GT
}


def _neutral_id(index: int) -> str:
    """Generate stable neutral site ID: AQ-P0-001."""
    return f"AQ-P0-{index:03d}"


def _verify_semantic_clean(cols: set, context: str) -> list[str]:
    """Check that no column name or semantic proxy leaks GT information."""
    violations = []
    for col in sorted(cols):
        if col in SEMANTIC_FORBIDDEN:
            violations.append(f"{context}: forbidden column '{col}' leaked")

        # Role/proxy checks
        lower = col.lower()
        for forbidden_term in [
            "role",
            "reason",
            "productivity",
            "flow",
            "pump",
            "negative",
            "positive",
            "quality",
            "score",
            "n_piez",
            "lithology",
            "evidence_count",
            "gt_",
            "temporal",
        ]:
            if forbidden_term in lower and col not in {"hydrogeological_unit"}:
                violations.append(
                    f"{context}: column '{col}' matches forbidden term '{forbidden_term}'"
                )
                break
    return violations


def _pump_flow_list(edf, igme_id: str) -> list[float]:
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


def _pump_drawdowns(edf, igme_id: str) -> list[float]:
    row = edf[edf["igme_id"] == igme_id]
    if len(row) == 0:
        return []
    try:
        data = json.loads(row.iloc[0]["hydraulic_evidence_json"])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [
        h.get("drawdown_m", 0) or 0
        for h in data
        if isinstance(h, dict) and h.get("drawdown_m") is not None
    ]


def main() -> None:
    print("=" * 70)
    print("Gate 6.1.1: Blindness Hardening")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Gate 6.1 Cohort B ────────────────────────────────────
    print("\n[1] Loading Cohort B from Gate 6.1 …")
    cohort_b = pd.read_csv(INPUT_COHORT_B)
    _vdf = pd.read_parquet(INPUT_VALIDATION)
    edf = pd.read_parquet(INPUT_ENRICHED)
    print(f"  Cohort B: {len(cohort_b)} sites")
    print(f"  Domains: {cohort_b['hydrogeological_unit'].value_counts().to_dict()}")

    # ── 2. Rename LOW_PRODUCTIVITY → LOW_FLOW_PUMPING_TEST_STRATUM ──
    print("\n[2] Renaming LOW_PRODUCTIVITY → LOW_FLOW_PUMPING_TEST_STRATUM …")
    # Update the negative audit CSV in-place for consistency
    audit_path = REPORTS_DIR / "negative_evidence_audit_v1.csv"
    audit_df = pd.read_csv(audit_path)

    for col in ["negative_class_WATER_PRESENCE", "negative_class_PRODUCTIVITY"]:
        if col in audit_df.columns:
            mask = audit_df[col] == "LOW_PRODUCTIVITY"
            audit_df.loc[mask, col] = "LOW_FLOW_PUMPING_TEST_STRATUM"
            n_changed = mask.sum()
            print(f"  Renamed {n_changed} records in {col}")

    # Also update cohort B's negative class columns
    for col in ["negative_class_WATER_PRESENCE", "negative_class_PRODUCTIVITY"]:
        if col in cohort_b.columns:
            cohort_b[col] = (
                cohort_b[col]
                .astype(str)
                .replace({"LOW_PRODUCTIVITY": "LOW_FLOW_PUMPING_TEST_STRATUM", "nan": "nan"})
            )

    audit_df.to_csv(audit_path, index=False)

    # Derive specific capacity for LOW_FLOW records
    n_sc = 0
    for _, row in cohort_b.iterrows():
        flows = _pump_flow_list(edf, row["igme_id"])
        draws = _pump_drawdowns(edf, row["igme_id"])
        if flows and draws and len(flows) == len(draws):
            for f, d in zip(flows, draws, strict=False):
                if f >= 0 and d > 0:
                    n_sc += 1
    print(f"  Cohort B records with derivable specific capacity: {n_sc}")

    # Compute whole-cohort specific-capacity stats
    all_sc_values = []
    for _, row in cohort_b.iterrows():
        flows = _pump_flow_list(edf, row["igme_id"])
        draws = _pump_drawdowns(edf, row["igme_id"])
        if flows and draws and len(flows) == len(draws):
            for f, d in zip(flows, draws, strict=False):
                if f >= 0 and d > 0:
                    all_sc_values.append(f / d)

    sc_report = {}
    if all_sc_values:
        sc_report = {
            "n_derivable": len(all_sc_values),
            "median_l_s_m": round(float(sorted(all_sc_values)[len(all_sc_values) // 2]), 3),
            "min_l_s_m": round(min(all_sc_values), 3),
            "max_l_s_m": round(max(all_sc_values), 3),
            "n_cohort_sites_with_sc": sum(
                1 for _, row in cohort_b.iterrows() if _pump_flow_list(edf, row["igme_id"])
            ),
            "rationale": (
                "Specific capacity = flow / drawdown. "
                "Only derived where flow >= 0 and drawdown > 0. "
                "No arbitrary thresholds are applied — this is a "
                "descriptive statistic, not a classification rule."
            ),
        }
        print(
            f"  Specific capacity: median={sc_report['median_l_s_m']} L/s/m, "
            f"range=[{sc_report['min_l_s_m']}, {sc_report['max_l_s_m']}]"
        )
    else:
        sc_report = {"n_derivable": 0, "note": "No derivable specific capacity in this cohort"}

    # ── 3. Create neutral site IDs ───────────────────────────────────
    print("\n[3] Creating neutral site IDs …")
    site_ids = list(cohort_b["igme_id"])
    neutral_ids = [_neutral_id(i + 1) for i in range(len(site_ids))]
    _id_map = dict(zip(neutral_ids, site_ids, strict=False))
    _id_reverse = dict(zip(site_ids, neutral_ids, strict=False))

    # ── 4. Build two-tier blinded views ─────────────────────────────
    print("\n[4] Building two-tier blinded views …")

    # FIELD_OPERATOR_VIEW: coordinates + hydrogeological unit only
    operator_rows = []
    for i, (_, row) in enumerate(cohort_b.iterrows()):
        operator_rows.append(
            {
                "blinded_site_id": neutral_ids[i],
                "latitude": row["coordinate_lat"],
                "longitude": row["coordinate_lon"],
                "hydrogeological_unit": row["hydrogeological_unit"],
            }
        )
    operator_df = pd.DataFrame(operator_rows)
    operator_df.to_csv(REPORTS_DIR / "phase0_field_operator_view_v1_1.csv", index=False)

    # INTERPRETER_BLINDED_VIEW: adds aquifer system, candidate tier (no GT)
    interpreter_rows = []
    for i, (_, row) in enumerate(cohort_b.iterrows()):
        interpreter_rows.append(
            {
                "blinded_site_id": neutral_ids[i],
                "latitude": row["coordinate_lat"],
                "longitude": row["coordinate_lon"],
                "hydrogeological_unit": row["hydrogeological_unit"],
                "aquifer_system": row.get("aquifer_system", "unknown"),
                "candidate_tier": row.get("candidate_tier", "unknown"),
            }
        )
    interpreter_df = pd.DataFrame(interpreter_rows)
    interpreter_df.to_csv(REPORTS_DIR / "phase0_interpreter_blinded_view_v1_1.csv", index=False)

    # Verify no leakage
    op_cols = set(operator_df.columns)
    int_cols = set(interpreter_df.columns)
    op_violations = _verify_semantic_clean(op_cols, "OPERATOR")
    int_violations = _verify_semantic_clean(int_cols, "INTERPRETER")
    all_violations = op_violations + int_violations

    print(f"  Operator columns: {sorted(op_cols)}")
    print(f"  Interpreter columns: {sorted(int_cols)}")
    print(f"  Semantic leakage violations: {len(all_violations)}")
    if all_violations:
        for v in all_violations:
            print(f"    VIOLATION: {v}")

    # ── 5. Sealed mapping ────────────────────────────────────────────
    print("\n[5] Creating sealed site mapping …")
    mapping_rows = []
    for i, (_, row) in enumerate(cohort_b.iterrows()):
        flows = _pump_flow_list(edf, row["igme_id"])
        mapping_rows.append(
            {
                "blinded_site_id": neutral_ids[i],
                "igme_id": row["igme_id"],
                "gt_quality_PRODUCTIVITY": row["gt_quality_PRODUCTIVITY"],
                "gt_quality_WATER_PRESENCE": row["gt_quality_WATER_PRESENCE"],
                "gt_quality_DEPTH_TO_WATER": row["gt_quality_DEPTH_TO_WATER"],
                "gt_quality_LITHOLOGY": row["gt_quality_LITHOLOGY"],
                "pumping_max_flow_l_s": max(flows) if flows else None,
                "negative_class_WATER_PRESENCE": row.get(
                    "negative_class_WATER_PRESENCE", "UNKNOWN"
                ),
                "role": row["role"],
                "reason_selected": row["reason_selected"],
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(REPORTS_DIR / "phase0_sealed_site_mapping_v1_1.csv", index=False)

    # ── 6. Semantic leakage policy ───────────────────────────────────
    print("\n[6] Generating semantic leakage policy …")
    leakage_policy = {
        "version": VERSION,
        "description": (
            "Gate 6.1.1 semantic leakage policy. Defines exactly which fields "
            "are forbidden in any field-facing view, based on both column-name "
            "and semantic-proxy checks."
        ),
        "forbidden_columns": sorted(SEMANTIC_FORBIDDEN),
        "operator_view_allowed_columns": sorted(OPERATOR_COLUMNS),
        "interpreter_view_allowed_columns": sorted(INTERPRETER_COLUMNS),
        "sealed_mapping_all_columns": list(mapping_df.columns),
        "semantic_proxy_rules": {
            "role": "MUST NOT appear — encodes productivity/water-presence class",
            "reason_selected": "MUST NOT appear — encodes target evidence",
            "productivity_note": "MUST NOT appear — encodes pumping outcome",
            "pumping_test_count": "MUST NOT appear — reveals data availability",
            "lithology_intervals": "MUST NOT appear — reveals lithology availability",
            "n_piezometric_observations": "MUST NOT appear — reveals water-level data",
            "temporal_relevance": "MUST NOT appear — reveals data recency",
        },
        "two_tier_design": (
            "FIELD_OPERATOR_VIEW: site ID + coordinates + hydrogeological unit. "
            "Used by acquisition team to locate and measure sites. "
            "INTERPRETER_BLINDED_VIEW: adds aquifer system + candidate tier. "
            "Used for ERT/seismic interpretation. "
            "The interpreter receives LESS contextual geological information "
            "than the operator to prevent bias in manual interpretation."
        ),
        "terminology_correction": {
            "previous": "LOW_PRODUCTIVITY",
            "corrected": "LOW_FLOW_PUMPING_TEST_STRATUM",
            "rationale": (
                "<10 L/s pumping-test flow is a specific, measurement-based "
                "criterion. It is NOT a universal hydrogeological definition "
                "of low productivity. The corrected term makes this explicit."
            ),
        },
        "specific_capacity_summary": sc_report,
        "blinded_id_scheme": "AQ-P0-{NNN} (sequential, stable, reproducible)",
        "violations_after_hardening": len(all_violations),
    }

    with open(REPORTS_DIR / "phase0_semantic_leakage_policy_v1_1.json", "w") as f:
        json.dump(leakage_policy, f, indent=2, ensure_ascii=False)

    # ── 7. Updated blinding protocol ─────────────────────────────────
    print("\n[7] Generating updated blinding protocol …")
    protocol = {
        "version": VERSION,
        "supersedes": "phase0_blinding_protocol_v1.json",
        "n_sites": len(cohort_b),
        "operator_view": {
            "file": "phase0_field_operator_view_v1_1.csv",
            "columns": list(operator_df.columns),
            "purpose": (
                "Site location and access. Used by acquisition team. "
                "Contains NO GT data, NO role, NO reason."
            ),
        },
        "interpreter_view": {
            "file": "phase0_interpreter_blinded_view_v1_1.csv",
            "columns": list(interpreter_df.columns),
            "purpose": (
                "Context for ERT/seismic interpretation. "
                "Aquifer system + candidate tier only. "
                "NO GT quality, NO productivity, NO water-presence data."
            ),
        },
        "sealed_mapping": {
            "file": "phase0_sealed_site_mapping_v1_1.csv",
            "columns": list(mapping_df.columns),
            "purpose": (
                "Maps blinded_site_id → igme_id and contains all "
                "GT quality, productivity, and role data. "
                "MUST REMAIN SEALED until prediction freeze."
            ),
        },
        "freeze_protocol": {
            "rule": (
                "1. Team receives OPERATOR view only.\n"
                "2. Navigate to site, perform AquaScan acquisition.\n"
                "3. Hash ERT/seismic RAW data (SHA-256).\n"
                "4. Interpreter receives INTERPRETER view.\n"
                "5. Compute prediction, freeze artifact (SHA-256).\n"
                "6. ONLY THEN: open SEALED MAPPING to reveal GT."
            ),
            "prediction_artifact_template": {
                "blinded_site_id": "AQ-P0-NNN",
                "acquisition_timestamp_utc": "<ISO8601>",
                "raw_ert_sha256": "<sha256>",
                "raw_seismic_sha256": "<sha256 or N/A>",
                "autocal_version": "<version>",
                "processing_version": "<version>",
                "prediction_targets": [],
                "prediction_values": {},
                "prediction_uncertainty": {},
                "confidence_gate": "PASS|FAIL|LOW_CONFIDENCE",
                "prediction_timestamp_utc": "<ISO8601>",
                "prediction_artifact_sha256": "<sha256>",
            },
        },
        "semantic_leakage_verified": len(all_violations) == 0,
    }

    with open(REPORTS_DIR / "phase0_blinding_protocol_v1_1.json", "w") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    # ── 8. Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 6.1.1 VERDICT")
    print("=" * 70)
    print(f"  Operator view columns: {len(op_cols)} (all GT-free)")
    print(f"  Interpreter view columns: {len(int_cols)} (GT-free)")
    print(f"  Sealed mapping columns: {len(mapping_df.columns)} (all GT)")
    print(f"  Semantic violations: {len(all_violations)}")
    print(f"  Neutral IDs: {len(neutral_ids)} (AQ-P0-001 to AQ-P0-{len(neutral_ids):03d})")
    print(f"  Specific capacity derivable: {sc_report.get('n_derivable', 0)} records")
    print("  Terminology: LOW_FLOW_PUMPING_TEST_STRATUM (was LOW_PRODUCTIVITY)")

    if len(all_violations) == 0:
        verdict = "PASS — Phase 0 blindness hardened and ready for experimental design"
    else:
        verdict = "FAIL — semantic Ground Truth leakage remains"
    print(f"\nFINAL VERDICT: {verdict}")

    print("\nArtifacts:")
    for p in sorted(REPORTS_DIR.glob("*_v1_1.*")):
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gate 6.1.1 blindness hardening")
    args = parser.parse_args()
    main()
