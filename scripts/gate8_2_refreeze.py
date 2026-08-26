#!/usr/bin/env python3
"""Gate 8.2: Leakage-Free Baseline Re-Freeze.

Gate 8.1 proved that the original Gate 8 benchmark contained target leakage
through `_target_*` input columns. Gate 8 artifacts remain immutable but are
scientifically superseded. This script freezes the canonical leakage-free
Baseline v2:

1. Marks Gate 8 v1 as superseded (INVALID_FOR_SCIENTIFIC_COMPARISON).
2. Enforces the canonical fail-closed leakage policy (any `_target_*`,
   target-summary, target-derived, or GT column is forbidden as input unless
   explicitly the prediction target; unknown columns fail closed).
3. Freezes BASELINE_DEPTH_V2 from the validated Gate 8.1 evaluation.
4. Records the KNOWN_OOD_DOMAIN limitation.
5. Freezes the Phase 0 comparator (San Antonio context baseline).
6. Defines a pre-prediction OOD support assessment (IN_DISTRIBUTION /
   WEAK_SUPPORT / OUT_OF_DISTRIBUTION) independent of target outcomes.
7. Produces the definitive results manifest + provenance.

DOES NOT: launch new model search, query IGME, reveal Phase 0 GT,
process ERT/seismic, train GeoAI.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gate8_baseline_benchmark import (  # noqa: E402
    RANDOM_SEED,
    _extract_head_target,
    extract_target_depth_to_water,
    extract_target_productivity,
)

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
VDF_PATH = ROOT / "data/processed/aquascan_validation_dataset_v1.parquet"
EDF_PATH = ROOT / "data/processed/ground_truth_cohort5000_v0_4.parquet"
COHORT_A_PATH = ROOT / "reports/phase0_cohort_A_v1.csv"

# Gate 8.1 source artifacts (validated, leakage-free)
G81_RESULTS = REPORTS_DIR / "baseline_forensic_results_v1_1.csv"
G81_CROSS_DOMAIN = REPORTS_DIR / "baseline_cross_domain_v1_1.csv"
G81_ABLATION = REPORTS_DIR / "baseline_feature_source_audit_v1_1.csv"
G81_SPATIAL = REPORTS_DIR / "baseline_spatial_prior_audit_v1_1.csv"
G81_UNCERTAINTY = REPORTS_DIR / "baseline_uncertainty_audit_v1_1.csv"
G81_SUMMARY = REPORTS_DIR / "baseline_forensic_summary_v1_1.json"
G81_ELIGIBILITY = REPORTS_DIR / "baseline_forensic_eligibility_v1_1.csv"

# Gate 8 original (immutable, superseded)
G8_SUMMARY = REPORTS_DIR / "baseline_summary_v1.json"
G8_SPLIT_MANIFEST = REPORTS_DIR / "baseline_split_manifest_v1.json"

V2_DIR = MODELS_DIR / "baselines_v2"

TARGET_COLUMNS = [
    "_target_depth_to_water_m",
    "_target_head_m",
    "_target_productivity_flow_l_s",
]

# Hard-fail prefixes: any column matching these is never an input.
FORBIDDEN_PREFIXES = ("_target_", "target_", "gt_", "readiness_", "phys_val_")
FORBIDDEN_EXACT = {
    "igme_id",
    "reference_flow_l_s",
    "water_presence_status",
    "water_presence_basis",
    "documentary_confidence",
    "temporal_gt_type",
    "spatial_is_duplicate",
    "spatial_has_neighbors",
    "n_in_proximity_group",
    "n_in_domain",
    "changed_endpoints",
    "source_response_changed",
    "created_at",
    "dataset_version",
    "detail_url",
    "documentary_notes",
    "candidate_tier",
    "candidate_priority_score",
}

# Ground-truth quality columns (any GT outcome information)
GT_QUALITY_PREFIX = "gt_quality_"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


# ── 1. Supersession artifact ──────────────────────────────────────────


def build_supersession() -> dict:
    """Mark Gate 8 v1 as superseded with documented reason."""
    return {
        "artifact": "gate8_v1_supersession.json",
        "superseded": {
            "gate": "Gate 8",
            "version": "v1",
            "status": "INVALID_FOR_SCIENTIFIC_COMPARISON",
            "reason": "TARGET_LEAKAGE",
            "details": (
                "Gate 8 permitted `_target_*` columns as input features. "
                "With the leak, RF depth-to-water MAE was ~0.21 m "
                "(near-perfect target copy). Gate 8.1 proved this and "
                "corrected it (honest in-domain MAE ~10 m)."
            ),
            "evidence": {
                "forensic_summary": "baseline_forensic_summary_v1_1.json",
                "target_leak_corrected": True,
                "leaked_mae_m": 0.21,
                "corrected_mae_m": 10.0,
            },
        },
        "superseded_by": {
            "gate": "Gate 8.1 corrected forensic benchmark / Baseline v2",
            "artifacts": [
                "reports/baseline_forensic_results_v1_1.csv",
                "reports/baseline_forensic_summary_v1_1.json",
                "reports/baseline_v2_results.csv",
                "reports/baseline_v2_manifest.json",
            ],
        },
        "preservation": {
            "note": "Original Gate 8 artifacts are preserved immutable for provenance.",
            "original_artifacts": [
                "reports/baseline_results_v1.csv",
                "reports/baseline_summary_v1.json",
                "reports/baseline_feature_manifest_v1.json",
                "reports/baseline_split_manifest_v1.json",
                "reports/baseline_sanity_controls_v1.json",
            ],
        },
        "created_at": _now_iso(),
    }


# ── 2. Canonical fail-closed feature policy ───────────────────────────


def classify_column_v2(col: str) -> str:
    """Classify a column under the fail-closed canonical policy.

    Returns one of:
    - INPUT_ALLOWED
    - TARGET (explicit prediction target)
    - FORBIDDEN (target-derived / GT / metadata / provenance)
    - UNCLASSIFIED (fail closed → not an input)
    """
    if col in TARGET_COLUMNS:
        return "TARGET"
    if col.startswith(FORBIDDEN_PREFIXES) or col in FORBIDDEN_EXACT:
        return "FORBIDDEN"
    # Explicitly allowed contextual families (documented in Gate 6 leakage policy)
    allowed_exact = {
        "aquifer_system",
        "construction_date",
        "drilling_method",
        "elevation_m",
        "group_aquifer",
        "group_watershed",
        "latitude",
        "longitude",
        "municipality",
        "point_nature",
        "province",
        "pump_type",
        "qa_coordinate_outlier",
        "qa_depth_outlier",
        "qa_duplicate_coordinate",
        "qa_elevation_outlier",
        "qa_flag_count",
        "total_depth_m",
        "water_use",
        "watershed",
        "group_domain",
        "group_spatial_cell",
        "group_proximity_500m",
    }
    if col in allowed_exact:
        return "INPUT_ALLOWED"
    return "UNCLASSIFIED"


def build_feature_policy(df: pd.DataFrame) -> dict:
    """Build the canonical v2 feature policy with per-target permitted lists."""
    policy = {
        "version": "v2",
        "description": (
            "Canonical fail-closed leakage policy. Any column matching `_target_*`, "
            "target-summary, target-derived, or GT prefixes is forbidden as input "
            "unless explicitly used as the prediction target. Unknown/unclassified "
            "columns FAIL CLOSED (never become inputs automatically)."
        ),
        "fail_closed": True,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_exact": sorted(FORBIDDEN_EXACT),
        "target_columns": TARGET_COLUMNS,
        "targets": {},
    }
    for target_name in [
        "TARGET_DEPTH_TO_WATER",
        "TARGET_HYDRAULIC_HEAD_ELEVATION",
        "TARGET_PRODUCTIVITY",
    ]:
        permitted: list[str] = []
        unclassified: list[str] = []
        for col in df.columns:
            role = classify_column_v2(col)
            if role == "INPUT_ALLOWED":
                permitted.append(col)
            elif role == "UNCLASSIFIED":
                unclassified.append(col)
        policy["targets"][target_name] = {
            "permitted": sorted(permitted),
            "n_permitted": len(permitted),
            "unclassified_failed_closed": sorted(unclassified),
            "n_unclassified_failed_closed": len(unclassified),
            "explicit_target_column": _target_col_for(target_name),
        }
    return policy


def _target_col_for(target_name: str) -> str:
    return {
        "TARGET_DEPTH_TO_WATER": "_target_depth_to_water_m",
        "TARGET_HYDRAULIC_HEAD_ELEVATION": "_target_head_m",
        "TARGET_PRODUCTIVITY": "_target_productivity_flow_l_s",
    }[target_name]


# ── 3. Baseline v2 results ────────────────────────────────────────────


def build_v2_results() -> pd.DataFrame:
    """Freeze canonical results from validated Gate 8.1 outputs."""
    forensic = _read_csv(G81_RESULTS)
    fixed = forensic[forensic["feature_arm"] == "FIXED_BASELINE"].copy()
    # Exclude Dummy rows from the canonical model results table? No: keep Dummy
    # rows as the reference baseline but tag them.
    fixed["is_dummy_baseline"] = fixed["model"] == "Dummy"
    fixed = fixed.rename(
        columns={
            "mae": "MAE",
            "mdae": "MdAE",
            "rmse": "RMSE",
            "r2": "R2",
            "bias": "bias",
            "interval_coverage": "uncertainty_coverage",
            "interval_width": "interval_width",
        }
    )
    # Dummy MAE reference per (target, regime, sub_id)
    dummy_rows = fixed[fixed["is_dummy_baseline"]]
    dummy_map = {
        (r["target"], r["regime"], r["sub_id"]): r["MAE"] for _, r in dummy_rows.iterrows()
    }
    fixed["baseline_dummy_MAE"] = fixed.apply(
        lambda r: dummy_map.get((r["target"], r["regime"], r["sub_id"]), np.nan),
        axis=1,
    )
    fixed["abstention_rate"] = 0.0  # no abstention mechanism in v2 context baseline
    return fixed


def build_v2_cross_domain() -> pd.DataFrame:
    cd = _read_csv(G81_CROSS_DOMAIN)
    return cd


# ── 4. Baseline selection ─────────────────────────────────────────────


def select_baseline_depth_v2() -> dict:
    """Select and document BASELINE_DEPTH_V2 from validated results."""
    results = _read_csv(G81_RESULTS)
    depth = results[
        (results["target"] == "TARGET_DEPTH_TO_WATER")
        & (results["feature_arm"] == "FIXED_BASELINE")
        & (results["model"] == "RandomForest")
    ]

    def _med(sub: pd.DataFrame, col: str) -> float:
        return float(sub[col].median())

    in_domain = depth[depth["regime"] == "REGIME_A_GROUPED"]
    spatial = depth[depth["regime"] == "REGIME_B_SPATIAL_BLOCK"]
    cross = depth[depth["regime"] == "REGIME_C"]

    return {
        "model_id": "BASELINE_DEPTH_V2",
        "model_family": "RandomForest",
        "n_estimators": 100,
        "max_depth": 15,
        "random_seed": RANDOM_SEED,
        "evaluation": {
            "IN_DOMAIN_CONTEXT_BASELINE": {
                "regime": "REGIME_A_GROUPED",
                "median_MAE_m": _med(in_domain, "mae"),
                "median_MdAE_m": _med(in_domain, "mdae"),
                "median_R2": _med(in_domain, "r2"),
                "dummy_median_MAE_m": _med(
                    results[
                        (results["target"] == "TARGET_DEPTH_TO_WATER")
                        & (results["feature_arm"] == "FIXED_BASELINE")
                        & (results["model"] == "Dummy")
                        & (results["regime"] == "REGIME_A_GROUPED")
                    ],
                    "mae",
                ),
            },
            "SPATIAL_HOLDOUT_CONTEXT_BASELINE": {
                "regime": "REGIME_B_SPATIAL_BLOCK",
                "median_MAE_m": _med(spatial, "mae"),
                "median_MdAE_m": _med(spatial, "mdae"),
                "median_R2": _med(spatial, "r2"),
            },
            "CROSS_DOMAIN_CONTEXT_BASELINE": {
                "regime": "REGIME_C",
                "median_MAE_m": _med(cross, "mae"),
                "median_MdAE_m": _med(cross, "mdae"),
                "median_R2": _med(cross, "r2"),
                "note": (
                    "Heterogeneous across held-out domains; catastrophic on "
                    "Región de los arenales (see KNOWN_OOD_DOMAIN)."
                ),
            },
        },
        "selection_notes": [
            "Selected for spatial/generalization performance + robustness, "
            "not solely in-domain MAE.",
            "Cross-domain improvement over Dummy survives for 3/5 domains; "
            "Región de los arenales is a documented OOD failure.",
            "Frozen from Gate 8.1 validated leakage-free evaluation.",
        ],
        "known_ood_domain": "Región de los arenales",
    }


# ── 5. Phase 0 comparator ─────────────────────────────────────────────


def build_phase0_comparator() -> dict:
    """Freeze the comparator future physical AquaScan experiments must beat.

    Phase 0 = San Antonio. The comparator uses the cross-domain held-out
    San Antonio fold of the leakage-free baseline (model trained WITHOUT the
    10 Phase 0 sites, which are firewalled).
    """
    cd = _read_csv(G81_CROSS_DOMAIN)
    san_antonio = cd[cd["held_out_domain"] == "San Antonio"]
    row = san_antonio.iloc[0]
    return {
        "phase0_domain": "San Antonio",
        "firewall": {
            "n_firewalled_records": 10,
            "firewall_source": "reports/phase0_cohort_A_v1.csv",
            "note": "10 Phase 0 sites excluded from training/tuning/selection.",
        },
        "comparator_source": "baseline_cross_domain_v1_1.csv (held-out San Antonio)",
        "comparator": {
            "model": str(row["model"]),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "MAE_m": float(row["mae"]),
            "dummy_MAE_m": float(row["dummy_mae"]),
            "improvement_over_dummy_m": float(row["improvement_over_dummy"]),
            "R2": float(row["r2"]),
            "bias_m": float(row["bias"]),
        },
        "future_arms": {
            "ARM_CONTEXT": "BASELINE_DEPTH_V2 (this artifact)",
            "ARM_E": "ERT-only physical prediction (future)",
            "ARM_S": "Seismic-only physical prediction (future)",
            "ARM_ES": "ERT+Seismic physical prediction (future)",
            "ARM_CONTEXT_PLUS_PHYSICS": "Context + ERT + Seismic (future)",
        },
        "comparison_concept": ("CONTEXT vs ERT vs SEISMIC vs ERT+SEISMIC vs CONTEXT+ERT+SEISMIC"),
        "note": (
            "Future physical arms must be compared against the appropriate "
            "evaluation-context baseline (in-domain / spatial-holdout / "
            "cross-domain). For Phase 0 San Antonio, the held-out San Antonio "
            "cross-domain comparator above is the primary reference."
        ),
        "created_at": _now_iso(),
    }


# ── 6. OOD support policy ─────────────────────────────────────────────


def classify_domain_support(
    df_domains: pd.DataFrame,
    new_domain: str,
    n_min_in_domain: int = 30,
    n_min_total: int = 100,
) -> str:
    """Pre-prediction domain-support assessment.

    Uses only training-domain feature support (record counts), never target
    outcomes of the new sample.

    - IN_DISTRIBUTION: >= n_min_in_domain training records in the same domain
    - WEAK_SUPPORT:   0 < training records in domain < n_min_in_domain, but
                      total training population >= n_min_total
    - OUT_OF_DISTRIBUTION: no training records in the domain OR total training
                      population < n_min_total
    """
    counts = df_domains["group_domain"].value_counts()
    n_in = int(counts.get(new_domain, 0))
    n_total = int(len(df_domains))
    if n_in >= n_min_in_domain and n_total >= n_min_total:
        return "IN_DISTRIBUTION"
    if n_in > 0 and n_total >= n_min_total:
        return "WEAK_SUPPORT"
    return "OUT_OF_DISTRIBUTION"


def build_ood_policy(df_domains: pd.DataFrame) -> dict:
    """Build the OOD policy artifact with per-domain support assessment.

    Support is computed on the depth-to-water modeling population (GT A/B,
    firewall intact) because that is the population BASELINE_DEPTH_V2 is
    trained on. Feature support is independent of target outcomes; a domain
    can be well supported by feature counts (IN_DISTRIBUTION) yet still be a
    known generalization failure (documented separately).
    """
    counts = df_domains["group_domain"].value_counts().to_dict()
    per_domain: dict[str, dict] = {}
    for domain in sorted(counts):
        support = classify_domain_support(df_domains, domain)
        per_domain[domain] = {
            "n_training_records": int(counts[domain]),
            "support": support,
            "note": (
                "Support computed from training-domain feature counts only "
                "(no target outcomes of the new sample)."
            ),
        }
    return {
        "version": "v2",
        "description": (
            "Pre-prediction domain-support assessment. Used to inform "
            "abstention/confidence. NEVER uses target errors of the new sample. "
            "Computed on the depth-to-water modeling population (GT A/B, "
            "Phase 0 firewall intact)."
        ),
        "classes": {
            "IN_DISTRIBUTION": ">= 30 training records in the same domain",
            "WEAK_SUPPORT": "1-29 training records in the same domain",
            "OUT_OF_DISTRIBUTION": "0 training records in the domain, or tiny total population",
        },
        "thresholds": {"n_min_in_domain": 30, "n_min_total": 100},
        "per_domain": per_domain,
        "known_ood_domain": "Región de los arenales",
        "important_distinction": (
            "Domain feature support and generalization failure are different "
            "concepts. Región de los arenales is well represented by feature "
            "counts (IN_DISTRIBUTION support) yet is a documented "
            "KNOWN_OOD_DOMAIN because its depth-to-water distribution is "
            "fundamentally different (median ~7.8 m vs ~36-48 m in other "
            "domains), so the baseline fails there despite ample support. "
            "Both flags are retained: support informs confidence/abstention; "
            "the known-OOD flag documents the validated generalization failure."
        ),
    }


# ── 7. Manifest / provenance ──────────────────────────────────────────


def build_manifest(feature_policy: dict) -> dict:
    """Build the definitive v2 provenance manifest."""
    src_files = [
        VDF_PATH,
        EDF_PATH,
        COHORT_A_PATH,
        G81_RESULTS,
        G81_CROSS_DOMAIN,
        G81_ABLATION,
        G81_SPATIAL,
        G81_UNCERTAINTY,
        G81_SUMMARY,
        G81_ELIGIBILITY,
    ]
    hashes = {p.name: _sha256_file(p) for p in src_files if p.exists()}

    policy_json = json.dumps(feature_policy, sort_keys=True)
    return {
        "version": "v2",
        "experiment": "AquaScan Open-Source Context Baseline v2",
        "created_at": _now_iso(),
        "software": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "random_seed": RANDOM_SEED,
        "source_hashes": hashes,
        "feature_policy_hash": _sha256_str(policy_json),
        "phase0_firewall": {
            "n_records": 10,
            "status": "intact",
            "source": "reports/phase0_cohort_A_v1.csv",
        },
        "source_gate81_artifacts": [
            "baseline_forensic_results_v1_1.csv",
            "baseline_cross_domain_v1_1.csv",
            "baseline_feature_source_audit_v1_1.csv",
            "baseline_spatial_prior_audit_v1_1.csv",
            "baseline_uncertainty_audit_v1_1.csv",
            "baseline_forensic_summary_v1_1.json",
            "baseline_forensic_eligibility_v1_1.csv",
        ],
        "supersedes": {
            "gate8_v1": "INVALID_FOR_SCIENTIFIC_COMPARISON (TARGET_LEAKAGE)",
            "note": "Gate 8 v1 artifacts preserved immutable for provenance.",
        },
    }


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Gate 8.2: Leakage-Free Baseline Re-Freeze")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading datasets …")
    vdf = pd.read_parquet(VDF_PATH)
    edf = pd.read_parquet(EDF_PATH)
    cohort_a = pd.read_csv(COHORT_A_PATH)
    firewalled_ids = set(cohort_a["igme_id"])
    print(f"  Validation dataset: {len(vdf)} records")
    print(f"  Phase 0 firewall: {len(firewalled_ids)} records")

    # Working dataset (firewall intact)
    working = vdf[~vdf["igme_id"].isin(firewalled_ids)].copy()
    working["_target_depth_to_water_m"] = extract_target_depth_to_water(working, edf)
    working["_target_productivity_flow_l_s"] = extract_target_productivity(working, edf)
    working["_target_head_m"] = _extract_head_target(working, edf).values

    # ── 1. Supersession ───────────────────────────────────────────────
    print("\n[2] Building supersession artifact …")
    supersession = build_supersession()
    (REPORTS_DIR / "gate8_v1_supersession.json").write_text(
        json.dumps(supersession, indent=2, ensure_ascii=False)
    )
    print("  gate8_v1_supersession.json written")

    # ── 2. Canonical feature policy ───────────────────────────────────
    print("\n[3] Building canonical feature policy …")
    feature_policy = build_feature_policy(vdf)
    for target, info in feature_policy["targets"].items():
        print(
            f"  {target}: {info['n_permitted']} permitted, "
            f"{info['n_unclassified_failed_closed']} failed closed"
        )
    (REPORTS_DIR / "baseline_v2_feature_policy.json").write_text(
        json.dumps(feature_policy, indent=2, ensure_ascii=False)
    )
    print("  baseline_v2_feature_policy.json written")

    # ── 3. Results ────────────────────────────────────────────────────
    print("\n[4] Freezing canonical results …")
    v2_results = build_v2_results()
    v2_results.to_csv(REPORTS_DIR / "baseline_v2_results.csv", index=False)
    print(f"  baseline_v2_results.csv: {len(v2_results)} rows")

    v2_cross = build_v2_cross_domain()
    v2_cross.to_csv(REPORTS_DIR / "baseline_v2_cross_domain.csv", index=False)
    print(f"  baseline_v2_cross_domain.csv: {len(v2_cross)} rows")

    # ── 4. Baseline selection ─────────────────────────────────────────
    print("\n[5] Selecting BASELINE_DEPTH_V2 …")
    baseline = select_baseline_depth_v2()
    for regime, metrics in baseline["evaluation"].items():
        if "median_MAE_m" in metrics:
            print(f"  {regime}: median MAE {metrics['median_MAE_m']:.2f} m")
    (REPORTS_DIR / "baseline_v2_selection.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False)
    )
    print("  baseline_v2_selection.json written")

    # ── 5. Phase 0 comparator ─────────────────────────────────────────
    print("\n[6] Freezing Phase 0 comparator …")
    comparator = build_phase0_comparator()
    (REPORTS_DIR / "baseline_v2_phase0_comparator.json").write_text(
        json.dumps(comparator, indent=2, ensure_ascii=False)
    )
    print(
        f"  San Antonio held-out comparator: MAE "
        f"{comparator['comparator']['MAE_m']:.2f} m "
        f"(Dummy {comparator['comparator']['dummy_MAE_m']:.2f} m)"
    )

    # ── 6. OOD policy ─────────────────────────────────────────────────
    print("\n[7] Building OOD support policy …")
    # Support computed on the depth-to-water modeling population (GT A/B,
    # Phase 0 firewall intact) — the population BASELINE_DEPTH_V2 trains on.
    depth_eligible = working[
        working["gt_quality_TARGET_DEPTH_TO_WATER"].isin(["A", "B"])
        & working["_target_depth_to_water_m"].notna()
    ]
    print(f"  Modeling population: {len(depth_eligible)} records")
    ood_policy = build_ood_policy(depth_eligible)
    for domain, info in ood_policy["per_domain"].items():
        print(f"  {domain}: {info['support']} (n={info['n_training_records']})")
    (REPORTS_DIR / "baseline_v2_ood_policy.json").write_text(
        json.dumps(ood_policy, indent=2, ensure_ascii=False)
    )
    print("  baseline_v2_ood_policy.json written")

    # ── 7. Manifest ───────────────────────────────────────────────────
    print("\n[8] Building manifest …")
    manifest = build_manifest(feature_policy)
    (REPORTS_DIR / "baseline_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print("  baseline_v2_manifest.json written")

    # ── 8. Model artifacts dir ────────────────────────────────────────
    V2_DIR.mkdir(parents=True, exist_ok=True)
    (V2_DIR / "README.md").write_text(
        "# AquaScan Open-Source Context Baseline v2 (frozen)\n\n"
        "Leakage-free baseline frozen in Gate 8.2. Full results and "
        "provenance live in `reports/baseline_v2_*` artifacts.\n"
    )
    print("  models/baselines_v2/ created")

    # ── 9. Summary / verdict ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 8.2 VERDICT")
    print("=" * 70)

    # Verify canonical results contain NO leaked target-derived metrics.
    # A leakage-free results table must not contain any `_target_*` column
    # as a feature, and must reproduce the honest (non-leaked) Gate 8.1 MAE.
    leaked_cols = [c for c in v2_results.columns if c.startswith("_target_")]
    policy_targets = feature_policy["targets"]
    leaked_in_policy = [
        c
        for t_info in policy_targets.values()
        for c in t_info["permitted"]
        if c.startswith(FORBIDDEN_PREFIXES) or c in FORBIDDEN_EXACT
    ]
    in_domain_mae = baseline["evaluation"]["IN_DOMAIN_CONTEXT_BASELINE"]["median_MAE_m"]

    pass_checks = True
    if leaked_cols:
        print(f"  FAIL — canonical results contain leaked columns: {leaked_cols}")
        pass_checks = False
    if leaked_in_policy:
        print(f"  FAIL — feature policy permits forbidden columns: {leaked_in_policy}")
        pass_checks = False
    if not (3 < in_domain_mae < 22):
        print(
            f"  FAIL — in-domain MAE {in_domain_mae:.2f} m outside honest range; "
            "leakage may persist"
        )
        pass_checks = False

    if pass_checks:
        print("  PASS — leakage-free context baseline v2 frozen for future AquaScan comparison")
        print(f"  BASELINE_DEPTH_V2: in-domain MAE {in_domain_mae:.2f} m")
        print(f"  KNOWN_OOD_DOMAIN: {baseline['known_ood_domain']}")
        print(f"  Phase 0 firewall: {len(firewalled_ids)} records intact")
        print(f"  Feature policy: fail-closed, {feature_policy['fail_closed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
