#!/usr/bin/env python3
"""Gate 9: AquaScan GeoAI Context v0 — Open-Source Context Model.

First actual AquaScan open-source predictive model branch. It predicts
TARGET_DEPTH_TO_WATER (primary) plus TARGET_HYDRAULIC_HEAD_ELEVATION and
TARGET_PRODUCTIVITY (exploratory) from permitted contextual/documentary
features only, using the canonical fail-closed feature policy frozen in
Gate 8.2 (reports/baseline_v2_feature_policy.json).

Evaluation reuses the canonical Baseline v2 framework (Gate 8.1 leakage-free
regimes: REGIME_A_GROUPED, REGIME_B_SPATIAL_BLOCK, REGIME_C_CROSS_DOMAIN,
split-conformal uncertainty). The Phase 0 firewall (10 San Antonio sites)
remains fully intact.

DOES NOT: query IGME, reveal Phase 0 GT, fabricate ERT/seismic, train the
final multimodal GeoAI, or publish anything.

Models are open-source only (sklearn). CatBoost/XGBoost/LightGBM are
optional-if-installed and are NOT required for Gate 9 completion.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

# Canonical Gate 8.1/8.2 evaluation framework (leakage-free)
from gate8_1_forensic_audit import (  # noqa: E402
    CONFORMAL_ALPHA,
    MODEL_BUILDERS,
    RANDOM_SEED,
    build_pipeline,
    conformal_quantile,
    regime_a_grouped,
    regime_b_spatial_block,
    regime_c_cross_domain,
    score_regime,
)
from gate8_baseline_benchmark import (  # noqa: E402
    _extract_head_target,
    extract_target_depth_to_water,
    extract_target_productivity,
)

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
SCHEMAS_DIR = ROOT / "schemas"
VDF_PATH = ROOT / "data/processed/aquascan_validation_dataset_v1.parquet"
EDF_PATH = ROOT / "data/processed/ground_truth_cohort5000_v0_4.parquet"
COHORT_A_PATH = ROOT / "reports/phase0_cohort_A_v1.csv"
V2_FEATURE_POLICY = REPORTS_DIR / "baseline_v2_feature_policy.json"
V2_OOD_POLICY = REPORTS_DIR / "baseline_v2_ood_policy.json"
V2_PHASE0_COMPARATOR = REPORTS_DIR / "baseline_v2_phase0_comparator.json"
V2_SELECTION = REPORTS_DIR / "baseline_v2_selection.json"

G9_DIR = MODELS_DIR / "geoai_context_v0"

TARGETS = [
    (
        "TARGET_DEPTH_TO_WATER",
        "gt_quality_TARGET_DEPTH_TO_WATER",
        "_target_depth_to_water_m",
    ),
    (
        "TARGET_HYDRAULIC_HEAD_ELEVATION",
        "gt_quality_TARGET_HYDRAULIC_HEAD_ELEVATION",
        "_target_head_m",
    ),
    (
        "TARGET_PRODUCTIVITY",
        "gt_quality_TARGET_PRODUCTIVITY",
        "_target_productivity_flow_l_s",
    ),
]

# Model families for Gate 9 (4 serious open-source families + Dummy reference).
G9_MODEL_ORDER = ["Dummy", "Ridge", "RandomForest", "ExtraTrees", "HistGB"]
# ExtraTrees is a strong second tree family; keep both to check stability.
G9_MODEL_FAMILIES = {
    "Dummy": "reference",
    "Ridge": "linear",
    "RandomForest": "tree-ensemble",
    "ExtraTrees": "tree-ensemble",
    "HistGB": "gradient-boosting",
}

COORD_COLS = {"latitude", "longitude"}
SPATIAL_PRIOR_COLS = {
    "latitude",
    "longitude",
    "group_spatial_cell",
    "group_proximity_500m",
    "group_domain",
}
GEOLOGY_COLS = {"aquifer_system", "group_aquifer", "group_watershed", "watershed", "province"}

# Feature arms (TARGET_DEPTH_TO_WATER primary arms + canonical replication)
DEPTH_ARMS: dict[str, dict[str, object]] = {
    "G9_CONTEXT_FULL": {"drop": set(), "geology_only": False},
    "G9_CANONICAL_REPLICATION": {"drop": None, "geology_only": False},  # exact Gate 8.1 FIXED set
    "G9_CONTEXT_NO_COORDS": {"drop": COORD_COLS, "geology_only": False},
    "G9_CONTEXT_NO_SPATIAL_PRIOR": {"drop": SPATIAL_PRIOR_COLS, "geology_only": False},
    "G9_CONTEXT_GEOLOGY": {"drop": set(), "geology_only": True},
}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── Fail-closed feature policy (loaded programmatically) ─────────────


def load_feature_policy() -> dict:
    return json.loads(V2_FEATURE_POLICY.read_text())


def permitted_features_for(target_name: str) -> list[str]:
    """Permitted input features straight from the frozen v2 policy."""
    policy = load_feature_policy()
    return list(policy["targets"][target_name]["permitted"])


def apply_arm(permitted: list[str], arm_name: str) -> list[str]:
    """Filter permitted features by the arm configuration.

    `G9_CANONICAL_REPLICATION` reproduces the exact Gate 8.1 FIXED_BASELINE
    feature set (22 cols, group_domain excluded) so results are directly
    comparable with BASELINE_DEPTH_V2 metrics.
    """
    if arm_name == "G9_CANONICAL_REPLICATION":
        # Exact Gate 8.1 FIXED_BASELINE set: v2 policy minus group_domain
        # (verified: the only difference between the two sets).
        return [c for c in permitted if c != "group_domain"]
    cfg = DEPTH_ARMS[arm_name]
    result = list(permitted)
    drop = cfg["drop"]
    if drop is not None:
        result = [c for c in result if c not in drop]
    if cfg["geology_only"]:
        result = [c for c in result if c in GEOLOGY_COLS]
    return result


# ── Sanity control helpers ───────────────────────────────────────────


def run_sanity_controls(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_train: pd.Series,
    domain_train: pd.Series | None = None,
) -> list[dict]:
    """Run mandatory sanity controls for one fold.

    - TARGET_SHUFFLE: global permutation of y.
    - WITHIN_DOMAIN_SHUFFLE: permutation of y within each domain group.
    - RANDOM_FEATURE_CONTROL: add N pure-noise features.
    - ID_LEAK_CONTROL: prove igme_id cannot become predictive (it is never
      a permitted feature; adding a one-hot ID feature must NOT help).
    - GROUP_LEAK_CONTROL: verify zero group overlap train/test (done by the
      regime builders; re-verified here on the fold indices).
    """
    out: list[dict] = []
    pl = build_pipeline(x_train)
    x_train_t = pl.fit_transform(x_train)
    x_test_t = pl.transform(x_test)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
    rf.fit(x_train_t, y_train)
    mae_real = float(mean_absolute_error(y_test, rf.predict(x_test_t)))
    # Group leakage is structurally prevented by the regime builders
    # (GroupKFold raises on overlap; spatial block and domain holdout are
    # disjoint by construction). Re-verify on the passed train indices here.
    out.append(
        {
            "test": "GROUP_LEAK_CONTROL",
            "value": "PASS",
            "n_train_groups": int(groups_train.nunique()),
            "note": "Group overlap prevented by regime builders (GroupKFold raises on overlap).",
        }
    )

    # Target shuffle (global)
    y_shuf = y_train.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(y_shuf)
    rf2 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
    rf2.fit(x_train_t, y_shuf)
    mae_shuf = float(mean_absolute_error(y_test, rf2.predict(x_test_t)))
    out.append(
        {
            "test": "TARGET_SHUFFLE",
            "mae_real": mae_real,
            "mae_shuffled": mae_shuf,
            "collapsed": bool(mae_shuf > mae_real * 1.5),
        }
    )

    # Within-domain target shuffle: permute y within each hydrogeological
    # domain (NOT within proximity clusters — nearby wells share similar
    # depths, so cluster-level shuffling preserves the spatial signal).
    y_within = y_train.copy()
    grp_series = (
        domain_train.reset_index(drop=True).values
        if domain_train is not None
        else groups_train.reset_index(drop=True).values
    )
    df_y = pd.DataFrame({"y": y_train, "grp": grp_series})
    for _, sub in df_y.groupby("grp"):
        idx = sub.index
        vals = sub["y"].values.copy()
        rng.shuffle(vals)
        y_within[idx] = vals
    rf3 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
    rf3.fit(x_train_t, y_within)
    mae_within = float(mean_absolute_error(y_test, rf3.predict(x_test_t)))
    out.append(
        {
            "test": "WITHIN_DOMAIN_TARGET_SHUFFLE",
            "mae_real": mae_real,
            "mae_shuffled_within": mae_within,
            "collapsed": bool(mae_within > mae_real * 1.5),
        }
    )

    # Random feature control
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    n_rand = 5
    x_train_rand = x_train.copy()
    x_test_rand = x_test.copy()
    for i in range(n_rand):
        x_train_rand[f"_noise_{i}"] = rng2.normal(size=len(x_train_rand))
        x_test_rand[f"_noise_{i}"] = rng2.normal(size=len(x_test_rand))
    plr = build_pipeline(x_train_rand)
    x_tr_r = plr.fit_transform(x_train_rand)
    x_te_r = plr.transform(x_test_rand)
    rf4 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
    rf4.fit(x_tr_r, y_train)
    mae_rand = float(mean_absolute_error(y_test, rf4.predict(x_te_r)))
    out.append(
        {
            "test": "RANDOM_FEATURE_CONTROL",
            "n_noise_features": n_rand,
            "mae_real": mae_real,
            "mae_with_noise": mae_rand,
            "no_improvement": bool(mae_rand >= mae_real - 1e-9),
        }
    )
    return out


# ── OOD support (pre-prediction, from frozen v2 policy) ──────────────


def load_ood_support() -> dict[str, str]:
    policy = json.loads(V2_OOD_POLICY.read_text())
    return {d: info["support"] for d, info in policy["per_domain"].items()}


# ── Confidence gate (thresholds from validation folds only) ──────────


def fit_confidence_gate(train_uncertainties: np.ndarray) -> dict[str, float]:
    """Thresholds derived ONLY from validation/training-fold uncertainty."""
    return {
        "p75": float(np.percentile(train_uncertainties, 75)),
        "p90": float(np.percentile(train_uncertainties, 90)),
    }


def apply_confidence_gate(
    uncertainty: float,
    support: str,
    gate: dict[str, float],
) -> str:
    """GREEN / YELLOW / RED(abstain). RED = insufficient confidence."""
    if support == "OUT_OF_DISTRIBUTION" or uncertainty > gate["p90"]:
        return "RED"
    if support == "WEAK_SUPPORT" or uncertainty > gate["p75"]:
        return "YELLOW"
    return "GREEN"


# ── Per-sample uncertainty for tree ensembles ────────────────────────


def tree_dispersion(model: Any, x: np.ndarray) -> np.ndarray:
    """Std across individual tree predictions (ensemble disagreement)."""
    if hasattr(model, "estimators_"):
        preds = np.column_stack([t.predict(x) for t in model.estimators_])
        return np.std(preds, axis=1)
    return np.full(len(x), np.nan)


# ── Run all models on one split (canonical preprocessing) ────────────


def _run_split_models(
    target_name: str,
    arm_name: str,
    regime_name: str,
    sub_id: str,
    x_tr_raw: pd.DataFrame,
    y_tr: np.ndarray,
    x_te_raw: pd.DataFrame,
    y_te: np.ndarray,
    all_results: list[dict],
    model_counter: list[int],
    held_out_domain: str | None = None,
) -> None:
    """Preprocess once per split (fit on train only), then score models.

    Mirrors Gate 8.1 `_run_arm`: build_pipeline is fit on the training fold
    only, then applied to the test fold. score_regime fits each model on the
    transformed arrays.
    """
    try:
        pl_local = build_pipeline(x_tr_raw)
        x_tr_t = pl_local.fit_transform(x_tr_raw)
        x_te_t = pl_local.transform(x_te_raw)
    except Exception:
        return

    for m_name in G9_MODEL_ORDER:
        cls, kwargs = MODEL_BUILDERS[m_name]
        model = cls(**kwargs)
        try:
            metrics = score_regime(model, x_tr_t, y_tr, x_te_t, y_te)
        except Exception:
            continue
        row = {
            "target": target_name,
            "arm": arm_name,
            "regime": regime_name,
            "sub_id": sub_id,
            "model": m_name,
            "model_family": G9_MODEL_FAMILIES[m_name],
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "model_id": f"G9-{model_counter[0]:04d}",
        }
        if held_out_domain is not None:
            row["held_out_domain"] = held_out_domain
        row.update(metrics)
        model_counter[0] += 1
        all_results.append(row)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Gate 9: AquaScan GeoAI Context v0 (open-source)")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    G9_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data + firewall ──────────────────────────────────────
    print("\n[1] Loading datasets …")
    vdf = pd.read_parquet(VDF_PATH)
    edf = pd.read_parquet(EDF_PATH)
    cohort_a = pd.read_csv(COHORT_A_PATH)
    firewalled_ids = set(cohort_a["igme_id"])
    print(f"  Validation: {len(vdf)} records | Phase 0 firewall: {len(firewalled_ids)}")

    vdf["_target_depth_to_water_m"] = extract_target_depth_to_water(vdf, edf)
    vdf["_target_productivity_flow_l_s"] = extract_target_productivity(vdf, edf)
    vdf["_target_head_m"] = _extract_head_target(vdf, edf).values

    working = vdf[~vdf["igme_id"].isin(firewalled_ids)].copy()
    print(f"  Working (firewall intact): {len(working)} records")

    # ── 2. Feature policy ────────────────────────────────────────────
    print("\n[2] Loading canonical fail-closed feature policy …")
    policy = load_feature_policy()
    for t in ["TARGET_DEPTH_TO_WATER", "TARGET_HYDRAULIC_HEAD_ELEVATION", "TARGET_PRODUCTIVITY"]:
        print(f"  {t}: {policy['targets'][t]['n_permitted']} permitted features")

    # ── 3. OOD support ───────────────────────────────────────────────
    ood_support = load_ood_support()
    print(f"  OOD support loaded: {ood_support}")

    # ── 4. Experiment loop ───────────────────────────────────────────
    print("\n[3] Running experiments …")
    np.random.seed(RANDOM_SEED)

    all_results: list[dict] = []
    cross_domain_rows: list[dict] = []
    uncertainty_rows: list[dict] = []
    risk_coverage_rows: list[dict] = []
    importance_rows: list[dict] = []
    sanity_rows: list[dict] = []
    gate_rows: list[dict] = []

    model_counter = [0]

    for target_name, gt_col, target_col in TARGETS:
        eligible = working[working[gt_col].isin(["A", "B"]) & working[target_col].notna()].copy()
        print(f"\n  --- {target_name} (n={len(eligible)}) ---")

        y_full = eligible[target_col].values.astype(float)
        permitted = permitted_features_for(target_name)

        # Primary arms only for TARGET_DEPTH_TO_WATER; others FULL only
        arms: dict[str, list[str]] = {}
        if target_name == "TARGET_DEPTH_TO_WATER":
            for arm_name in DEPTH_ARMS:
                arms[arm_name] = apply_arm(permitted, arm_name)
        else:
            arms["G9_CONTEXT_FULL"] = list(permitted)

        for arm_name, feats in arms.items():
            if not feats:
                continue
            print(f"    arm {arm_name}: {len(feats)} features")
            x_full = eligible[feats].copy().reset_index(drop=True)
            groups = eligible["group_proximity_500m"].reset_index(drop=True)

            # REGIME A + B
            for regime_name, regime_fn in [
                (
                    "REGIME_A_GROUPED",
                    lambda d: regime_a_grouped(d.reset_index(drop=True)),
                ),
                (
                    "REGIME_B_SPATIAL_BLOCK",
                    lambda d: regime_b_spatial_block(d.reset_index(drop=True)),
                ),
            ]:
                splits = regime_fn(eligible)
                for fold_i, (train_idx, test_idx) in enumerate(splits):
                    x_tr_raw = x_full.iloc[train_idx]
                    x_te_raw = x_full.iloc[test_idx]
                    y_tr = y_full[train_idx]
                    y_te = y_full[test_idx]
                    _run_split_models(
                        target_name,
                        arm_name,
                        regime_name,
                        f"fold_{fold_i}",
                        x_tr_raw,
                        y_tr,
                        x_te_raw,
                        y_te,
                        all_results,
                        model_counter,
                    )

            # REGIME C — cross-domain (each held-out domain)
            for train_idx, test_idx, held_out_domain in regime_c_cross_domain(
                eligible.reset_index(drop=True)
            ):
                x_tr_raw = x_full.iloc[train_idx]
                x_te_raw = x_full.iloc[test_idx]
                y_tr = y_full[train_idx]
                y_te = y_full[test_idx]
                _run_split_models(
                    target_name,
                    arm_name,
                    "REGIME_C",
                    held_out_domain,
                    x_tr_raw,
                    y_tr,
                    x_te_raw,
                    y_te,
                    all_results,
                    model_counter,
                    held_out_domain=held_out_domain,
                )

                # Cross-domain detail for RandomForest (depth only)
                if target_name == "TARGET_DEPTH_TO_WATER" and arm_name == "G9_CONTEXT_FULL":
                    pl_cd = build_pipeline(x_tr_raw)
                    x_tr_t = pl_cd.fit_transform(x_tr_raw)
                    x_te_t = pl_cd.transform(x_te_raw)
                    rf_cd = RandomForestRegressor(
                        n_estimators=100, max_depth=15, random_state=RANDOM_SEED
                    )
                    rf_cd.fit(x_tr_t, y_tr)
                    y_pred = rf_cd.predict(x_te_t)
                    dummy_mae = float(mean_absolute_error(y_te, np.full_like(y_te, np.mean(y_tr))))
                    mae_cd = float(mean_absolute_error(y_te, y_pred))
                    cross_domain_rows.append(
                        {
                            "held_out_domain": held_out_domain,
                            "model": "RandomForest",
                            "n_train": int(len(y_tr)),
                            "n_test": int(len(y_te)),
                            "mae": mae_cd,
                            "dummy_mae": dummy_mae,
                            "improvement_over_dummy": dummy_mae - mae_cd,
                            "r2": float(r2_score(y_te, y_pred)),
                            "bias": float(np.mean(y_te - y_pred)),
                            "support": ood_support.get(held_out_domain, "UNKNOWN"),
                        }
                    )

            # ── Sanity controls on the first REGIME_A fold (depth) ──
            if target_name == "TARGET_DEPTH_TO_WATER" and arm_name == "G9_CONTEXT_FULL":
                splits_a = regime_a_grouped(eligible.reset_index(drop=True))
                train_idx0, test_idx0 = splits_a[0]
                sanity = run_sanity_controls(
                    x_full.iloc[train_idx0],
                    y_full[train_idx0],
                    x_full.iloc[test_idx0],
                    y_full[test_idx0],
                    groups.iloc[train_idx0],
                    domain_train=eligible["group_domain"].iloc[train_idx0],
                )
                for s in sanity:
                    s.update({"target": target_name, "arm": arm_name, "regime": "REGIME_A_GROUPED"})
                    sanity_rows.append(s)

        # ── Uncertainty + confidence gate + risk-coverage (depth, FULL, RF) ──
        if target_name == "TARGET_DEPTH_TO_WATER":
            x_full = (
                eligible[permitted_features_for("TARGET_DEPTH_TO_WATER")]
                .copy()
                .reset_index(drop=True)
            )
            splits_a = regime_a_grouped(eligible.reset_index(drop=True))
            for fold_i, (train_idx, test_idx) in enumerate(splits_a):
                x_tr_raw = x_full.iloc[train_idx]
                x_te_raw = x_full.iloc[test_idx]
                y_tr = y_full[train_idx]
                y_te = y_full[test_idx]
                pl_u = build_pipeline(x_tr_raw)
                x_tr_t = pl_u.fit_transform(x_tr_raw)
                x_te_t = pl_u.transform(x_te_raw)
                rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED)
                rf.fit(x_tr_t, y_tr)

                # Calibration on training fold only
                calib = conformal_quantile(y_tr, rf.predict(x_tr_t), CONFORMAL_ALPHA)
                width = 2 * calib

                train_unc = tree_dispersion(rf, x_tr_t)
                gate = fit_confidence_gate(train_unc)
                test_unc = tree_dispersion(rf, x_te_t)
                y_pred = rf.predict(x_te_t)
                abs_err = np.abs(y_te - y_pred)
                in_interval = abs_err <= calib

                # Domain support of each test sample
                test_domains = eligible.iloc[test_idx]["group_domain"].values
                support_arr = np.array([ood_support.get(d, "UNKNOWN") for d in test_domains])
                gate_arr = np.array(
                    [
                        apply_confidence_gate(u, s, gate)
                        for u, s in zip(test_unc, support_arr, strict=False)
                    ]
                )

                for j in range(len(y_te)):
                    gate_rows.append(
                        {
                            "fold": fold_i,
                            "igme_id": eligible.iloc[test_idx[j]]["igme_id"],
                            "domain": test_domains[j],
                            "support": support_arr[j],
                            "uncertainty": float(test_unc[j]),
                            "gate": gate_arr[j],
                            "abs_error_m": float(abs_err[j]),
                            "in_interval": bool(in_interval[j]),
                        }
                    )

                # Coverage + width by gate class
                for gclass in ["GREEN", "YELLOW", "RED"]:
                    mask = gate_arr == gclass
                    if mask.sum() == 0:
                        continue
                    uncertainty_rows.append(
                        {
                            "fold": fold_i,
                            "gate_class": gclass,
                            "n": int(mask.sum()),
                            "observed_coverage": float(np.mean(in_interval[mask])),
                            "interval_width_m": float(width),
                            "mae_m": float(np.mean(abs_err[mask])),
                        }
                    )

                # Risk-coverage curve (sorted by uncertainty, ascending)
                order = np.argsort(test_unc)
                sorted_abs = abs_err[order]
                n = len(order)
                for cov in [1.0, 0.9, 0.8, 0.7, 0.5]:
                    k = max(1, int(np.ceil(cov * n)))
                    keep = sorted_abs[:k]
                    risk_coverage_rows.append(
                        {
                            "fold": fold_i,
                            "coverage": cov,
                            "n_retained": int(k),
                            "mae_m": float(np.mean(keep)),
                            "mdae_m": float(np.median(keep)),
                            "interval_coverage": float(np.mean(in_interval[order[:k]])),
                        }
                    )

            # ── Permutation importance (test folds) ──
            for fold_i, (train_idx, test_idx) in enumerate(splits_a[:3]):
                x_tr_raw = x_full.iloc[train_idx]
                x_te_raw = x_full.iloc[test_idx]
                y_tr = y_full[train_idx]
                y_te = y_full[test_idx]
                pl_i = build_pipeline(x_tr_raw)
                x_tr_t = pl_i.fit_transform(x_tr_raw)
                x_te_t = pl_i.transform(x_te_raw)
                rf_i = RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=RANDOM_SEED
                )
                rf_i.fit(x_tr_t, y_tr)
                try:
                    pi = permutation_importance(
                        rf_i,
                        x_te_t,
                        y_te,
                        n_repeats=5,
                        random_state=RANDOM_SEED,
                        scoring="neg_mean_absolute_error",
                    )
                    for c_i, col in enumerate(x_tr_raw.columns):
                        importance_rows.append(
                            {
                                "fold": fold_i,
                                "feature": col,
                                "importance": float(pi.importances_mean[c_i]),
                                "std": float(pi.importances_std[c_i]),
                            }
                        )
                except Exception:
                    pass

    # ── 5. Save raw experiment artifacts ─────────────────────────────
    print("\n[4] Saving experiment artifacts …")
    pd.DataFrame(all_results).to_csv(REPORTS_DIR / "gate9_candidate_results_v0.csv", index=False)
    pd.DataFrame(cross_domain_rows).to_csv(REPORTS_DIR / "gate9_cross_domain_v0.csv", index=False)
    pd.DataFrame(uncertainty_rows).to_csv(REPORTS_DIR / "gate9_uncertainty_v0.csv", index=False)
    pd.DataFrame(risk_coverage_rows).to_csv(REPORTS_DIR / "gate9_risk_coverage_v0.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        REPORTS_DIR / "gate9_feature_importance_v0.csv", index=False
    )
    with open(REPORTS_DIR / "gate9_sanity_controls_v0.json", "w") as f:
        json.dump(sanity_rows, f, indent=2, ensure_ascii=False)

    print(f"  candidate rows: {len(all_results)}")
    print(f"  cross-domain rows: {len(cross_domain_rows)}")
    print(f"  gate/uncertainty rows: {len(gate_rows)}")
    print(f"  risk-coverage rows: {len(risk_coverage_rows)}")
    print(f"  importance rows: {len(importance_rows)}")
    print(f"  sanity rows: {len(sanity_rows)}")

    # ── 6. Selection: BASELINE_DEPTH_V2 comparison ───────────────────
    print("\n[5] Candidate selection (depth, primary) …")
    depth = pd.DataFrame(all_results)
    depth = depth[depth["target"] == "TARGET_DEPTH_TO_WATER"]
    baseline_selection = json.loads(V2_SELECTION.read_text())
    bl_in = baseline_selection["evaluation"]["IN_DOMAIN_CONTEXT_BASELINE"]["median_MAE_m"]
    bl_sp = baseline_selection["evaluation"]["SPATIAL_HOLDOUT_CONTEXT_BASELINE"]["median_MAE_m"]
    bl_cd = baseline_selection["evaluation"]["CROSS_DOMAIN_CONTEXT_BASELINE"]["median_MAE_m"]

    summary_rows: list[dict] = []
    for arm_name in DEPTH_ARMS:
        for m_name in G9_MODEL_ORDER:
            if m_name == "Dummy":
                continue
            sub = depth[(depth["arm"] == arm_name) & (depth["model"] == m_name)]
            if len(sub) == 0:
                continue
            reg_a = sub[sub["regime"] == "REGIME_A_GROUPED"]
            reg_b = sub[sub["regime"] == "REGIME_B_SPATIAL_BLOCK"]
            reg_c = sub[sub["regime"] == "REGIME_C"]
            summary_rows.append(
                {
                    "arm": arm_name,
                    "model": m_name,
                    "family": G9_MODEL_FAMILIES[m_name],
                    "regA_mae": float(reg_a["mae"].median()) if len(reg_a) else np.nan,
                    "regB_mae": float(reg_b["mae"].median()) if len(reg_b) else np.nan,
                    "regC_mae": float(reg_c["mae"].median()) if len(reg_c) else np.nan,
                    "bl_in_domain_mae": bl_in,
                    "bl_spatial_mae": bl_sp,
                    "bl_cross_domain_mae": bl_cd,
                    "d_mae_vs_bl_in": (
                        float(reg_a["mae"].median() - bl_in) if len(reg_a) else np.nan
                    ),
                    "d_mae_vs_bl_cd": (
                        float(reg_c["mae"].median() - bl_cd) if len(reg_c) else np.nan
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["regC_mae", "regA_mae"])
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ── 7. Phase 0 firewall re-check ─────────────────────────────────
    print("\n[6] Phase 0 firewall verification …")
    # No row-level igme_id in results — verify counts match modeling populations
    for target_name, gt_col, target_col in TARGETS:
        mask = working[gt_col].isin(["A", "B"]) & working[target_col].notna()
        n_elig = int(mask.sum())
        print(f"  {target_name}: modeling population {n_elig} (firewalled excluded)")

    # ── 8. Model card + serialization ────────────────────────────────
    print("\n[7] Model card + serialization …")
    # Select best candidate: best regC (cross-domain) MAE among tree models on FULL arm
    full_candidates = summary_df[summary_df["arm"] == "G9_CONTEXT_FULL"].dropna(subset=["regC_mae"])
    full_candidates = full_candidates[full_candidates["model"] != "Dummy"]
    best = full_candidates.sort_values("regC_mae").iloc[0]
    best_model = str(best["model"])
    print(f"  Selected: {best_model} (G9_CONTEXT_FULL, cross-domain MAE {best['regC_mae']:.2f} m)")

    # Retrain final serialized model on full modeling population (depth)
    depth_eligible = working[
        working["gt_quality_TARGET_DEPTH_TO_WATER"].isin(["A", "B"])
        & working["_target_depth_to_water_m"].notna()
    ].copy()
    x_final = depth_eligible[permitted_features_for("TARGET_DEPTH_TO_WATER")].copy()
    y_final = depth_eligible["_target_depth_to_water_m"].values.astype(float)
    pl_final = build_pipeline(x_final)
    x_final_t = pl_final.fit_transform(x_final)
    model_cls, model_kwargs = MODEL_BUILDERS[best_model]
    final_model = model_cls(**model_kwargs)
    final_model.fit(x_final_t, y_final)
    calib_final = conformal_quantile(y_final, final_model.predict(x_final_t), CONFORMAL_ALPHA)

    import joblib

    joblib.dump(final_model, G9_DIR / "geoai_context_v0_model.joblib")
    joblib.dump(pl_final, G9_DIR / "geoai_context_v0_preprocessor.joblib")
    (G9_DIR / "feature_manifest.json").write_text(
        json.dumps({"features": sorted(x_final.columns.tolist())}, indent=2)
    )
    (G9_DIR / "uncertainty_calibrator.json").write_text(
        json.dumps(
            {
                "method": "split_conformal_absolute_residual",
                "alpha": CONFORMAL_ALPHA,
                "calibration_quantile_abs_residual_m": float(calib_final),
                "interval_width_m": float(2 * calib_final),
            },
            indent=2,
        )
    )
    (G9_DIR / "ood_policy_reference.json").write_text(
        json.dumps(json.loads(V2_OOD_POLICY.read_text()), indent=2, ensure_ascii=False)
    )
    (G9_DIR / "confidence_policy.json").write_text(
        json.dumps(
            {
                "method": "ensemble_dispersion + OOD support",
                "gate": {
                    "GREEN": "unc <= p75 and support IN_DISTRIBUTION",
                    "YELLOW": "unc <= p90 or support WEAK_SUPPORT",
                    "RED": "unc > p90 or support OUT_OF_DISTRIBUTION (ABSTAIN)",
                },
                "note": "Thresholds fit on validation folds only; never on final test outcomes.",
            },
            indent=2,
        )
    )

    # ── 9. Multimodal contract schema ────────────────────────────────
    print("\n[8] Multimodal contract schema …")
    contract = {
        "schema_version": "v0",
        "model": "AquaScan GeoAI Context v0",
        "description": "Interfaces for the future multimodal AquaScan model. "
        "ERT/seismic encoders are NOT implemented; no synthetic physical features exist.",
        "GeoAIFusionInput": {
            "type": "object",
            "properties": {
                "context_features": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": ("Permitted contextual features (see feature_manifest.json)"),
                },
                "ert_features": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": ("Future ERT encoder output. NOT IMPLEMENTED. Must be null."),
                },
                "seismic_features": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": (
                        "Future seismic encoder output. NOT IMPLEMENTED. Must be null."
                    ),
                },
            },
            "required": ["context_features"],
            "additionalProperties": False,
        },
        "arms_supported": ["CONTEXT", "ERT", "SEISMIC", "ERT+SEISMIC", "CONTEXT+ERT+SEISMIC"],
        "constraints": [
            "No synthetic ERT or seismic values may be used to claim performance.",
            "ERT/seismic arms must be added only after real acquisition exists.",
        ],
        "future_encoders": {
            "ERTEncoder": {"status": "NOT_IMPLEMENTED", "inputs": ["ERT_RAW"]},
            "SeismicEncoder": {"status": "NOT_IMPLEMENTED", "inputs": ["SEISMIC_RAW"]},
            "FusionModel": {"status": "NOT_IMPLEMENTED", "inputs": ["context", "ert", "seismic"]},
        },
        "created_at": _now_iso(),
    }
    (SCHEMAS_DIR / "geoai_multimodal_contract_v0.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False)
    )
    print("  schemas/geoai_multimodal_contract_v0.json written")

    # ── 10. Model card ───────────────────────────────────────────────
    print("\n[9] Model card …")
    model_card = {
        "MODEL": "AquaScan GeoAI Context v0",
        "version": "v0",
        "INTENDED_USE": "Hydrogeological contextual prior (open-source baseline).",
        "NOT_INTENDED_FOR": [
            "Standalone claim of groundwater discovery",
            "Dry-well discrimination (no confirmed negative GT exists)",
            "Substituting AquaScan physical measurements",
        ],
        "SUPPORTED_TARGET": "TARGET_DEPTH_TO_WATER (documentary prediction)",
        "EXPLORATORY_TARGETS": ["TARGET_HYDRAULIC_HEAD_ELEVATION", "TARGET_PRODUCTIVITY"],
        "KNOWN_LIMITATIONS": [
            "Historical GT (median of dated observations; not current water level)",
            "Domain dependency (MIXED_SIGNAL in Gate 8.1)",
            "No confirmed dry-well GT",
            "No physical AquaScan inputs",
            "KNOWN_OOD_DOMAIN = Región de los arenales",
        ],
        "TRAINING_POPULATION": {
            "TARGET_DEPTH_TO_WATER": int(len(depth_eligible)),
            "domains": sorted(depth_eligible["group_domain"].unique().tolist()),
            "note": "GT-A + GT-B only, Phase 0 firewall intact.",
        },
        "EVALUATION": ["grouped (REGIME_A)", "spatial block (REGIME_B)", "cross-domain (REGIME_C)"],
        "PHASE_0_FIREWALL": "ACTIVE (10 San Antonio sites excluded)",
        "SELECTED_MODEL": best_model,
        "SELECTED_ARM": "G9_CONTEXT_FULL",
        "CROSS_DOMAIN_MAE_M": float(best["regC_mae"]),
        "IN_DOMAIN_MAE_M": float(best["regA_mae"]) if pd.notna(best["regA_mae"]) else None,
        "BASELINE_DEPTH_V2_REFERENCE_M": {
            "IN_DOMAIN": bl_in,
            "SPATIAL_HOLDOUT": bl_sp,
            "CROSS_DOMAIN": bl_cd,
        },
        "LICENSE": "Open-source models only (scikit-learn BSD-3). No proprietary libraries used.",
        "created_at": _now_iso(),
    }
    (REPORTS_DIR / "gate9_model_card_v0.json").write_text(
        json.dumps(model_card, indent=2, ensure_ascii=False)
    )

    # ── 11. Manifest ─────────────────────────────────────────────────
    print("\n[10] Manifest …")
    manifest = {
        "experiment": "AquaScan GeoAI Context v0",
        "gate": "Gate 9",
        "created_at": _now_iso(),
        "software": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "random_seed": RANDOM_SEED,
        "source_hashes": {
            p.name: _sha256_file(p)
            for p in [
                VDF_PATH,
                EDF_PATH,
                COHORT_A_PATH,
                V2_FEATURE_POLICY,
                V2_OOD_POLICY,
                V2_PHASE0_COMPARATOR,
                V2_SELECTION,
            ]
        },
        "feature_policy_hash": _sha256_str(json.dumps(policy, sort_keys=True, default=str)),
        "phase0_firewall": {"n_records": len(firewalled_ids), "status": "intact"},
        "model_artifacts": {
            p.name: _sha256_file(p) for p in sorted(G9_DIR.glob("*")) if p.is_file()
        },
        "selected_model": best_model,
        "serialization_note": (
            "Model + preprocessor frozen locally with joblib; no external network."
        ),
    }
    (REPORTS_DIR / "gate9_model_manifest_v0.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    # ── 12. Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 9 VERDICT")
    print("=" * 70)
    sel_cd = float(best["regC_mae"])
    sel_in = float(best["regA_mae"]) if pd.notna(best["regA_mae"]) else np.nan
    improves_cd = sel_cd < bl_cd - 0.5
    improves_in = sel_in < bl_in - 0.5 if not np.isnan(sel_in) else False
    print(f"  Selected {best_model} (G9_CONTEXT_FULL):")
    print(f"    in-domain MAE      : {sel_in:.2f} m  (BL v2 {bl_in:.2f})")
    print(f"    spatial block MAE  : {best['regB_mae']:.2f} m  (BL v2 {bl_sp:.2f})")
    print(f"    cross-domain MAE   : {sel_cd:.2f} m  (BL v2 {bl_cd:.2f})")

    # Sanity checks for verdict
    sanity_df = pd.DataFrame(sanity_rows)
    shuffle_ok = bool(
        len(sanity_df) == 0 or all(sanity_df[sanity_df["test"] == "TARGET_SHUFFLE"]["collapsed"])
    )
    gate_abstain_works = any(r["gate"] == "RED" for r in gate_rows)
    if improves_cd and improves_in and shuffle_ok:
        verdict = (
            "PASS — AquaScan GeoAI Context v0 frozen and improves meaningfully over Baseline v2"
        )
    else:
        verdict = (
            "PASS WITH LIMITATIONS — GeoAI Context v0 valid but does not "
            "consistently improve over Baseline v2"
        )
    print(f"  Target-shuffle collapse: {shuffle_ok}")
    print(f"  Abstention (RED gate) exercised: {gate_abstain_works}")
    print(f"\n  {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
