#!/usr/bin/env python3
"""Train and package the AquaScan national depth-to-water model v1.

The script has two evaluation modes:

* ``--evaluate``: rerun the full leakage-safe evaluation (grouped CV by 500 m
  proximity, province holdout, domain holdout) and refresh the cached reports.
* ``--reuse-evaluation``: reuse the cached reports, but verify that their OOF
  predictions, targets and grouping labels match the current modeling frame
  exactly.

In both cases the final RandomForest is trained on all modeling rows, and only
model weights, policies, aggregate metrics and the model card are written.
Record-level IGME data is never copied into the artifact directory.

Usage:
    python scripts/train_national_model.py --reuse-evaluation
    python scripts/train_national_model.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aquascan.modeling.national_depth import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    FOLD_GROUP_COLUMN,
    NUMERIC_FEATURES,
    SPATIAL_CELL_COLUMN,
    TARGET_COLUMN,
    build_confidence_policy,
    build_national_depth_frame,
    build_ood_policy,
    conformal_quantile,
    run_domain_holdout,
    run_grouped_cv,
    sha256_file,
    train_final_model,
)

DATASET_PATH = ROOT / "data/processed/aquascan_national_dataset_v1.parquet"
FIREWALL_PATH = ROOT / "reports/phase0_cohort_A_v1.csv"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUT_DIR = ROOT / "models/national_v1"
CV_REPORT = REPORTS_DIR / "national_model_v1_cv.csv"
DOMAIN_REPORT = REPORTS_DIR / "national_model_v1_domain_holdout.json"
OOF_REPORT = REPORTS_DIR / "national_model_v1_oof.npz"
NATIONAL_MANIFEST = REPORTS_DIR / "national_enrichment_manifest_v1.json"

MODEL_ORDER = ["Dummy", "Ridge", "RandomForest", "ExtraTrees"]
SPLITS = [
    ("group_proximity_500m", "Grouped CV by 500 m proximity"),
    ("province", "Province holdout"),
]
ALPHA = 0.1
DOMAIN_MIN_RECORDS = 150


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--firewall", type=Path, default=FIREWALL_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Rerun the full evaluation and refresh the cached reports.",
    )
    mode.add_argument(
        "--reuse-evaluation",
        action="store_true",
        help="Reuse cached reports after verifying they match the modeling frame.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--domain-min-records", type=int, default=DOMAIN_MIN_RECORDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def _validate_cached_evaluation(
    cv: pd.DataFrame,
    domains: list[dict[str, Any]],
    oof_payload: dict[str, np.ndarray],
    data: Any,
) -> None:
    """Fail closed when cached reports do not match the current frame."""
    frame = data.frame
    required_cv = {"n", "mae", "medae", "rmse", "r2", "fold", "model", "split"}
    missing_cv = required_cv - set(cv.columns)
    if missing_cv:
        raise ValueError(f"cached CV report is missing columns: {sorted(missing_cv)}")
    if not domains:
        raise ValueError("cached domain holdout report is empty")

    if set(oof_payload) != {"oof", "y", "groups"}:
        raise ValueError(f"unexpected OOF keys: {sorted(oof_payload)}")
    oof = np.asarray(oof_payload["oof"], dtype=float)
    target = np.asarray(oof_payload["y"], dtype=float)
    groups = np.asarray(oof_payload["groups"]).astype(str)
    expected_target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    expected_groups = frame[FOLD_GROUP_COLUMN].astype(str).to_numpy()

    if len(oof) != len(frame):
        raise ValueError(
            f"cached OOF has {len(oof)} rows but the modeling frame has {len(frame)}. "
            "Run with --evaluate."
        )
    if np.isnan(oof).any():
        raise ValueError("cached OOF contains NaN predictions. Run with --evaluate.")
    if not np.allclose(target, expected_target, rtol=0.0, atol=1e-9):
        raise ValueError("cached target does not match the modeling frame. Run with --evaluate.")
    if not np.array_equal(groups, expected_groups):
        raise ValueError(
            "cached proximity groups do not match the modeling frame. Run with --evaluate."
        )


def _evaluate(data: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict], dict]:
    """Run the full evaluation protocol and write the cached reports."""
    frame = data.frame
    rows: list[pd.DataFrame] = []
    oof_payload: dict[str, np.ndarray] = {}

    for split_column, label in SPLITS:
        print(f"\n[evaluation] {label}")
        for model_name in MODEL_ORDER:
            print(f"  {model_name} ...", flush=True)
            fold_metrics, oof = run_grouped_cv(
                frame,
                split_column,
                model_name,
                n_splits=5,
                seed=args.seed,
            )
            rows.append(fold_metrics)
            print(
                f"    median MAE={fold_metrics['mae'].median():.3f} m | "
                f"mean MAE={fold_metrics['mae'].mean():.3f} m",
                flush=True,
            )
            if split_column == "group_proximity_500m" and model_name == "RandomForest":
                oof_payload = {
                    "oof": oof,
                    "y": frame[TARGET_COLUMN].to_numpy(dtype=float),
                    "groups": frame[FOLD_GROUP_COLUMN].astype(str).to_numpy(),
                }

    cv = pd.concat(rows, ignore_index=True)
    CV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(CV_REPORT, index=False)
    print(f"\n[evaluation] wrote {CV_REPORT.relative_to(ROOT)}")

    print(f"\n[evaluation] leave-one-domain-out (>= {args.domain_min_records} records)")
    domain_rows = run_domain_holdout(
        frame,
        min_records=args.domain_min_records,
        seed=args.seed,
    )
    DOMAIN_REPORT.write_text(
        json.dumps(domain_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[evaluation] wrote {DOMAIN_REPORT.relative_to(ROOT)}")

    np.savez_compressed(OOF_REPORT, **oof_payload)
    print(f"[evaluation] wrote {OOF_REPORT.relative_to(ROOT)}")
    return cv, domain_rows, oof_payload


def _load_cached_evaluation(
    args: argparse.Namespace,
    data: Any,
) -> tuple[pd.DataFrame, list[dict], dict]:
    missing = [path for path in (CV_REPORT, DOMAIN_REPORT, OOF_REPORT) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing cached evaluation reports: "
            + ", ".join(str(path) for path in missing)
            + ". Run with --evaluate."
        )
    cv = pd.read_csv(CV_REPORT)
    domain_rows = json.loads(DOMAIN_REPORT.read_text(encoding="utf-8"))
    with np.load(OOF_REPORT, allow_pickle=True) as payload:
        oof_payload = {key: payload[key] for key in payload.files}
    _validate_cached_evaluation(cv, domain_rows, oof_payload, data)
    print("[evaluation] cached reports verified against the modeling frame")
    return cv, domain_rows, oof_payload


def _summarize_cv(cv: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_column, label in SPLITS:
        split_summary: dict[str, Any] = {"label": label, "models": {}}
        for model_name in MODEL_ORDER:
            subset = cv[(cv["split"] == split_column) & (cv["model"] == model_name)]
            if subset.empty:
                continue
            split_summary["models"][model_name] = {
                "n_folds": int(len(subset)),
                "mae_median_m": float(subset["mae"].median()),
                "mae_mean_m": float(subset["mae"].mean()),
                "mae_min_m": float(subset["mae"].min()),
                "mae_max_m": float(subset["mae"].max()),
                "medae_median_m": float(subset["medae"].median()),
                "rmse_median_m": float(subset["rmse"].median()),
                "r2_median": float(subset["r2"].median()),
                "folds": subset[["fold", "n", "mae", "medae", "rmse", "r2", "seconds"]].to_dict(
                    orient="records"
                ),
            }
        summary[split_column] = split_summary
    return summary


def _summarize_domain_holdout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.asarray([row["n_test"] for row in rows], dtype=float)
    summary: dict[str, Any] = {
        "n_domains": len(rows),
        "n_records": int(weights.sum()),
        "domains": rows,
    }
    for model_name in ("rf", "dummy"):
        mae = np.asarray([row[model_name]["mae"] for row in rows], dtype=float)
        rmse = np.asarray([row[model_name]["rmse"] for row in rows], dtype=float)
        r2 = np.asarray([row[model_name]["r2"] for row in rows], dtype=float)
        summary[model_name] = {
            "mae_median_m": float(np.median(mae)),
            "mae_mean_m": float(np.mean(mae)),
            "mae_weighted_m": float(np.average(mae, weights=weights)),
            "mae_min_m": float(np.min(mae)),
            "mae_max_m": float(np.max(mae)),
            "rmse_median_m": float(np.median(rmse)),
            "rmse_weighted_m": float(np.sqrt(np.average(rmse**2, weights=weights))),
            "r2_median": float(np.median(r2)),
        }
    summary["rf_better_than_dummy_in_domains"] = int(
        sum(row["rf"]["mae"] < row["dummy"]["mae"] for row in rows)
    )
    summary["domains_without_rf_improvement"] = [
        row["domain"] for row in rows if row["rf"]["mae"] >= row["dummy"]["mae"]
    ]
    return summary


def _conformal_summary(oof_payload: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    oof = np.asarray(oof_payload["oof"], dtype=float)
    target = np.asarray(oof_payload["y"], dtype=float)
    residuals = np.abs(target - oof)
    quantile = conformal_quantile(target, oof, alpha)
    return {
        "method": "split_conformal_absolute_residual",
        "alpha": float(alpha),
        "calibration_set": "out-of-fold predictions from 5-fold GroupKFold by 500 m proximity",
        "n_calibration": int(len(target)),
        "calibration_quantile_abs_residual_m": quantile,
        "interval_half_width_m": quantile,
        "interval_width_m": 2.0 * quantile,
        "empirical_coverage": float(np.mean(residuals <= quantile)),
        "oof_mae_m": float(np.mean(residuals)),
        "oof_median_abs_error_m": float(np.median(residuals)),
        "oof_rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "oof_r2": float(1.0 - np.sum(residuals**2) / np.sum((target - target.mean()) ** 2)),
        "note": "Marginal interval over the national modeling population; not per-site.",
    }


def _build_metrics(
    data: Any,
    cv_summary: dict[str, Any],
    domain_summary: dict[str, Any],
    conformal: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return {
        "model": "AquaScan National Groundwater Depth Model v1",
        "model_id": "aquascan-national-depth-v1",
        "created_at": created_at,
        "target": data.metadata["target"],
        "dataset": data.metadata["dataset"],
        "phase0_firewall": data.metadata["phase0_firewall"],
        "modeling_population": data.metadata["modeling_population"],
        "readiness_quality": data.metadata["readiness_quality"],
        "feature_contract": {
            "n_features": len(FEATURES),
            "features": FEATURES,
            "fold_grouping_columns": [FOLD_GROUP_COLUMN, SPATIAL_CELL_COLUMN],
            "fold_grouping_note": (
                "Grouping columns are not model inputs. They are used only to "
                "prevent same-site leakage in evaluation."
            ),
        },
        "evaluation": {
            "grouped_cv_proximity_500m": cv_summary["group_proximity_500m"],
            "province_holdout": cv_summary["province"],
            "domain_holdout": domain_summary,
            "conformal": conformal,
        },
    }


def _build_training_manifest(
    data: Any,
    args: argparse.Namespace,
    created_at: str,
    artifact_hashes: dict[str, str],
    reports: list[Path],
) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        ROOT / "src/aquascan/modeling/national_depth.py",
        Path(args.dataset).resolve(),
        Path(args.firewall).resolve(),
        CV_REPORT,
        DOMAIN_REPORT,
        OOF_REPORT,
    ]
    if NATIONAL_MANIFEST.exists():
        source_files.append(NATIONAL_MANIFEST)
    source_hashes = {}
    for path in source_files:
        if path.exists():
            try:
                key = str(path.relative_to(ROOT))
            except ValueError:
                key = str(path)
            source_hashes[key] = sha256_file(path)

    try:
        parent_artifact = str(Path(args.dataset).resolve().relative_to(ROOT))
    except ValueError:
        parent_artifact = Path(args.dataset).name

    return {
        "experiment_id": "aquascan-national-depth-v1",
        "created_at": created_at,
        "git_commit": None,
        "git_commit_note": (
            "The working tree is not a git repository; source-file hashes below "
            "identify the exact code used."
        ),
        "software": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "random_seed": args.seed,
        "dataset": data.metadata["dataset"],
        "phase0_firewall": data.metadata["phase0_firewall"],
        "modeling_population": data.metadata["modeling_population"],
        "feature_contract": {
            "n_features": len(FEATURES),
            "features": FEATURES,
            "fold_grouping_columns": [FOLD_GROUP_COLUMN, SPATIAL_CELL_COLUMN],
        },
        "training": {
            "model_class": "sklearn.ensemble.RandomForestRegressor",
            "hyperparameters": {
                "n_estimators": 150,
                "max_depth": 20,
                "min_samples_leaf": 5,
                "random_state": args.seed,
                "n_jobs": -1,
            },
            "n_training_records": int(len(data.frame)),
            "target": TARGET_COLUMN,
        },
        "evaluation_protocol": {
            "grouped_cv": {
                "group_column": FOLD_GROUP_COLUMN,
                "n_splits": 5,
                "models": MODEL_ORDER,
            },
            "province_holdout": {"group_column": "province", "n_splits": 5},
            "domain_holdout": {
                "min_records": args.domain_min_records,
                "n_domains_scored": int(len(json.loads(DOMAIN_REPORT.read_text(encoding="utf-8")))),
            },
            "conformal": {
                "alpha": args.alpha,
                "calibration": "out-of-fold GroupKFold by 500 m proximity",
            },
        },
        "provenance": {
            "source_record_id": (
                "not_applicable: aggregate model artifact; no record-level identifiers"
            ),
            "source_uri": ("IGME bdaguas national snapshot via AquaScan national enrichment v1"),
            "source_hash": data.metadata["dataset"]["sha256"],
            "processing_step": "national depth-to-water model training",
            "processing_version": "national_v1",
            "timestamp": created_at,
            "parent_artifact": parent_artifact,
        },
        "source_files": source_hashes,
        "reports": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in reports if path.exists()
        },
        "model_artifacts": artifact_hashes,
        "record_level_data_published": False,
    }


def _write_example(out_dir: Path, half_width_m: float) -> None:
    content = f'''"""Minimal synthetic example for the AquaScan National model v1.

The input row is synthetic: it is not a real site and the prediction has no
scientific meaning. It only demonstrates the input contract and the API.

The support gate would mark this row RED because its domain and province are
"unknown". The prediction is still computed to demonstrate the estimator.

Usage:
    pip install -r requirements.txt
    python example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFORMAL_HALF_WIDTH_M = {half_width_m!r}  # alpha = 0.1, from uncertainty_calibrator.json


def main() -> None:
    model = joblib.load(HERE / "model.joblib")
    preprocessor = joblib.load(HERE / "preprocessor.joblib")
    features = json.loads((HERE / "feature_manifest.json").read_text())["features"]

    # Synthetic row. Categorical values are deliberately "unknown" so the
    # example does not imply a real location or a real site.
    row = {{feature: "unknown" for feature in features}}
    row.update(
        {{
            "elevation_m": 100.0,
            "latitude": 39.5,
            "longitude": -3.7,
            "qa_coordinate_outlier": 0,
            "qa_depth_outlier": 0,
            "qa_duplicate_coordinate": 0,
            "qa_elevation_outlier": 0,
            "qa_flag_count": 0,
            "total_depth_m": 100.0,
        }}
    )

    x = pd.DataFrame([row], columns=features)
    prediction_m = float(model.predict(preprocessor.transform(x))[0])

    print("WARNING: synthetic input, not a real site. Output has no scientific meaning.")
    print(f"Predicted historical depth to water: {{prediction_m:.2f}} m")
    print(
        "90% conformal interval: "
        f"[{{prediction_m - CONFORMAL_HALF_WIDTH_M:.2f}}, "
        f"{{prediction_m + CONFORMAL_HALF_WIDTH_M:.2f}}] m"
    )
    print("This is a research baseline, not a groundwater detector.")


if __name__ == "__main__":
    main()
'''
    (out_dir / "example.py").write_text(content, encoding="utf-8")


def _write_requirements(out_dir: Path) -> None:
    content = (
        "# Runtime versions used to serialize the artifacts.\n"
        "# scikit-learn does not guarantee pickle compatibility across versions.\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"joblib=={joblib.__version__}\n"
    )
    (out_dir / "requirements.txt").write_text(content, encoding="utf-8")


def _write_model_card(
    out_dir: Path,
    data: Any,
    cv_summary: dict[str, Any],
    domain_summary: dict[str, Any],
    conformal: dict[str, Any],
    ood_policy: dict[str, Any],
    created_at: str,
    seed: int,
) -> None:
    metadata = data.metadata
    population = metadata["modeling_population"]
    target = metadata["target"]
    dataset = metadata["dataset"]
    firewall = metadata["phase0_firewall"]
    grouped = cv_summary["group_proximity_500m"]["models"]
    province = cv_summary["province"]["models"]
    rf_grouped = grouped["RandomForest"]
    dummy_grouped = grouped["Dummy"]
    rf_province = province["RandomForest"]
    dummy_province = province["Dummy"]
    domain_rf = domain_summary["rf"]
    domain_dummy = domain_summary["dummy"]

    worst_domains = sorted(
        domain_summary["domains"], key=lambda row: row["rf"]["mae"], reverse=True
    )[:5]
    worst_lines = "\n".join(
        f"- `{row['domain']}`: {_fmt(row['rf']['mae'])} m MAE "
        f"(n={row['n_test']}, dummy {_fmt(row['dummy']['mae'])} m)"
        for row in worst_domains
    )
    no_improvement = domain_summary["domains_without_rf_improvement"]
    if not no_improvement:
        no_improvement_text = "RandomForest beats the median dummy in every scored domain."
    elif len(no_improvement) == 1:
        no_improvement_text = (
            "The only domain where RandomForest does not beat the median dummy is "
            f"`{no_improvement[0]}`."
        )
    else:
        no_improvement_text = (
            "The domains where RandomForest does not beat the median dummy are "
            + ", ".join(f"`{name}`" for name in no_improvement)
            + "."
        )

    missingness = metadata["source_missingness"]
    n_records = population["records"]
    missing_lines = "\n".join(
        f"| `{column.replace('source_', '')}` | {_pct(missingness[column] / n_records)} |"
        for column in (
            "source_construction_date",
            "source_total_depth_m",
            "source_aquifer_system",
            "source_watershed",
        )
    )

    summary_table = "\n".join(
        [
            "| Item | Value |",
            "|---|---|",
            "| Model name | AquaScan National Groundwater Depth Model v1 |",
            "| Task | Tabular regression: `TARGET_DEPTH_TO_WATER` (metres) |",
            (
                "| Model | `RandomForestRegressor(n_estimators=150, max_depth=20, "
                "min_samples_leaf=5, random_state=42)` |"
            ),
            (
                "| Preprocessing | Median imputation + `StandardScaler` for numeric; "
                "most-frequent imputation + "
                '`OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` '
                "for categorical |"
            ),
            "| Inputs | 21 documentary/contextual features |",
            f"| Training population | {population['records']:,} READY depth records |",
            (
                f"| Geographic scope | Spain — {population['provinces']} of "
                f"{dataset['snapshot_provinces']} provinces and "
                f"{population['domains']} of {dataset['snapshot_domains']} domains "
                f"present in the {dataset['snapshot_records']:,}-record national snapshot |"
            ),
            (
                f"| Phase 0 firewall | Active — {firewall['n_ids']} sealed San Antonio "
                f"sites excluded ({firewall['n_ready_excluded']} READY records) |"
            ),
            f"| Framework | scikit-learn {sklearn.__version__} |",
            "| License | Apache-2.0 (code and weights) |",
        ]
    )

    evaluation_table = "\n".join(
        [
            "| Evaluation regime | RF MAE | Dummy (median) MAE | RF RMSE | RF R² |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Grouped 5-fold CV (500 m proximity) | "
                f"{_fmt(rf_grouped['mae_median_m'])} m | "
                f"{_fmt(dummy_grouped['mae_median_m'])} m | "
                f"{_fmt(rf_grouped['rmse_median_m'])} m | "
                f"{_fmt(rf_grouped['r2_median'], 3)} |"
            ),
            (
                f"| Province holdout (5 folds; {population['provinces']} provinces) | "
                f"{_fmt(rf_province['mae_median_m'])} m | "
                f"{_fmt(dummy_province['mae_median_m'])} m | "
                f"{_fmt(rf_province['rmse_median_m'])} m | "
                f"{_fmt(rf_province['r2_median'], 3)} |"
            ),
            (
                f"| Domain holdout ({domain_summary['n_domains']} domains ≥150 records; "
                f"weighted) | {_fmt(domain_rf['mae_weighted_m'])} m | "
                f"{_fmt(domain_dummy['mae_weighted_m'])} m | "
                f"{_fmt(domain_rf['rmse_weighted_m'])} m | "
                f"median {_fmt(domain_rf['r2_median'], 3)} |"
            ),
        ]
    )

    card = f"""---
language:
  - en
license: apache-2.0
library_name: scikit-learn
pretty_name: AquaScan National Groundwater Depth Model v1
tags:
  - groundwater
  - hydrogeology
  - geoscience
  - tabular-regression
  - random-forest
  - scikit-learn
  - spain
  - national
---

# AquaScan National Groundwater Depth Model v1

Research baseline that predicts **historical depth to water (metres)** from
documentary and contextual groundwater features across Spain. It is part of
the [AquaScan Open Science](https://github.com/angelopezvpb/aquascan-open-science)
project.

> **This is not a groundwater detector.** It is a tabular regression baseline
> trained on historical documentary records. It has **not been validated in the field**
> and does **not** classify dry wells, and must not be used as the sole basis
> for drilling, investment or safety-critical decisions.

This is the **canonical AquaScan depth-to-water model** in
[`angellopez/aquascan-open-science-model`](https://huggingface.co/angellopez/aquascan-open-science-model).
It supersedes the earlier five-domain GeoAI Context v0 baseline, which is
preserved at the
[`v0-5domain` tag](https://huggingface.co/angellopez/aquascan-open-science-model/tree/v0-5domain).
The five-domain baseline was trained on 563 records from five domains; this
model is trained on the national READY depth-to-water population described
below.

## Model summary

{summary_table}

## Intended use

- Research baseline for leakage-safe national groundwater contextual modelling.
- Reproducing the AquaScan national depth-to-water evaluation.
- A **prior** to compare against physical measurements (ERT, seismic) in
  future experiments.
- Screening and prioritisation, with explicit uncertainty and support flags.

## Out-of-scope use

- Claiming groundwater discovery or borehole success.
- Dry-well discrimination: this model does not predict water presence.
- Replacing a hydrogeological study or a field campaign.
- Operational, commercial or safety-critical decisions.
- Treating the prediction as a current water level.

## Geographic scope and support

The national snapshot contains **{dataset["snapshot_records"]:,} records**,
**{dataset["snapshot_provinces"]} provinces** and **{dataset["snapshot_domains"]} domains**.
After requiring a READY depth-to-water target and applying the Phase 0
firewall, the modeling population is **{population["records"]:,} records** from
**{population["provinces"]} provinces** and **{population["domains"]} domains**.

This is a national model in coverage, **not a uniformly validated model for
every province or domain**. Of the {population["domains"]} modeling domains,
**{ood_policy["summary"]["n_domains_in_distribution"]}** have at least 30 training
records, **{ood_policy["summary"]["n_domains_weak_support"]}** have 1–29 records,
and the domain-holdout evaluation only scores the
**{domain_summary["n_domains"]}** domains with at least 150 records. Predictions
for weak-support domains carry no domain-level validation.

The 10 sealed Phase 0 San Antonio sites are excluded before grouping,
evaluation and training. They are not part of the published training
population.

## Evaluation

All metrics are for `TARGET_DEPTH_TO_WATER` and use leakage-safe groups. The
same 500 m proximity cluster, province or domain is never split across train
and test.

{evaluation_table}

Figures are medians across folds except the domain-holdout row: MAE and RMSE
are weighted by each domain's test-record count, and R² is the median across
domains. The RandomForest improves over the median dummy in the grouped,
province and domain regimes. The province holdout is the hardest regime:
performance depends strongly on the held-out province.

Domain-holdout detail: RandomForest beats the median dummy in
**{domain_summary["rf_better_than_dummy_in_domains"]} of
{domain_summary["n_domains"]}** scored domains. {no_improvement_text}
The highest domain-holdout errors are:

{worst_lines}

These are aggregate domain results; they are not site-specific guarantees.

## Uncertainty

Split-conformal absolute residual at `alpha = {conformal["alpha"]}` gives a
**±{_fmt(conformal["interval_half_width_m"])} m** 90% interval
(`interval_width_m = {_fmt(conformal["interval_width_m"])} m`) with
**{_pct(conformal["empirical_coverage"])}** empirical coverage on the
out-of-fold predictions. This is a **marginal** interval over the national
modeling population, not a per-site guarantee. It is wide because the target
spans {_fmt(target["min_m"])}–{_fmt(target["max_m"])} m and the model is
trained across heterogeneous hydrogeological domains.

The `confidence_policy.json` gate additionally flags unseen or weak-support
domains and provinces. A RED gate means **abstain**, not that the point
prediction is valid.

## Limitations

- **No field validation.** No ERT/seismic/GNSS groundwater validation has been
  performed.
- **Historical target.** The target is the median of dated observations; it is
  not the current water level.
- **No water-presence or dry-well classification.** The model predicts depth
  only. It must not be used to claim that a well will be productive.
- **Uneven support.** {ood_policy["summary"]["n_domains_weak_support"]} of
  {population["domains"]} domains have fewer than 30 training records, and only
  {domain_summary["n_domains"]} domains have a dedicated holdout evaluation.
- **High-error domains.** Even with support, some domains have large holdout
  errors (see above), so aggregate national metrics hide local failures.
- **Distribution shift.** Performance outside Spain is not established.
- **Marginal uncertainty.** The conformal interval is global and wide; it does
  not model site-specific uncertainty.
- **No physical inputs.** The model uses documentary/contextual features only.
- **Missing documentary data.** `construction_date` is missing for
  {_pct(missingness["source_construction_date"] / n_records)} of records;
  imputation is used. Missingness for key source fields:

| Field | Missing |
|---|---:|
{missing_lines}

## Features (21)

```
{", ".join(FEATURES)}
```

- Numeric (9): `elevation_m`, `latitude`, `longitude`, the four `qa_*`
  flags, `qa_flag_count`, `total_depth_m`.
- Categorical (12): the remaining features.

`group_proximity_500m` and `group_spatial_cell` are **not** model inputs. They
are built only to define leakage-safe evaluation folds. Target-derived columns
(`temporal_summary_json`, `readiness_*`, `gt_*`) are never inputs.

The exact contract is in [`feature_manifest.json`](feature_manifest.json).
Rows must contain **all 21 columns**; missing columns raise an error.

## How to use

```bash
pip install -r requirements.txt
python example.py
```

```python
import json
import joblib
import pandas as pd

model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
features = json.load(open("feature_manifest.json"))["features"]

row = {{feature: "unknown" for feature in features}}
row.update({{
    "elevation_m": 100.0,
    "latitude": 39.5,
    "longitude": -3.7,
    "qa_coordinate_outlier": 0,
    "qa_depth_outlier": 0,
    "qa_duplicate_coordinate": 0,
    "qa_elevation_outlier": 0,
    "qa_flag_count": 0,
    "total_depth_m": 100.0,
}})

X = pd.DataFrame([row], columns=features)
prediction_m = model.predict(preprocessor.transform(X))[0]
print(f"Predicted historical depth to water: {{prediction_m:.2f}} m")
print("90% conformal interval: +/- {_fmt(conformal["interval_half_width_m"])} m")
```

The row above is **synthetic** and only demonstrates the API. It is not a
real site and its output has no scientific meaning. For real data, preserve
the same feature dtypes used in training and check the support policy before
using a prediction.

## Files

| File | Description |
|---|---|
| `model.joblib` | Fitted `RandomForestRegressor` |
| `preprocessor.joblib` | Fitted preprocessing pipeline |
| `feature_manifest.json` | Exact input contract (21 features) |
| `confidence_policy.json` | GREEN/YELLOW/RED support and interval policy |
| `uncertainty_calibrator.json` | Split-conformal calibration |
| `ood_policy_reference.json` | Domain/province support policy |
| `metrics.json` | Aggregate evaluation metrics |
| `training_manifest.json` | Provenance, source hashes and hyperparameters |
| `example.py` | Minimal synthetic example |
| `requirements.txt` | Pinned runtime versions |
| `SHA256SUMS` | Artifact checksums |

## Data provenance and license

The model was trained on IGME-derived documentary records from the national
AquaScan enrichment (dataset SHA-256
`{dataset["sha256"]}`). The **record-level dataset is not redistributed**:
the IGME reuse terms are still under review and the dataset may contain
record-level information. Only aggregate counts, the model card, the policies
and the fitted weights are published.

Code and weights are released under **Apache-2.0**. This does not grant any
rights over the underlying IGME data. Users are responsible for complying with
the source-data terms.

## Reproducibility

- `experiment_id`: `aquascan-national-depth-v1`
- Random seed: `{seed}`
- Dataset SHA-256: `{dataset["sha256"]}`
- Phase 0 firewall: `{firewall["n_ids"]}` sealed IDs excluded
- Created at: `{created_at}`

The training/evaluation entry point is
`scripts/train_national_model.py`; the modeling code is
`src/aquascan/modeling/national_depth.py`. The working tree is not a git
repository, so `training_manifest.json` records SHA-256 hashes of the exact
source and report files instead of a commit hash.

> **Synthetic inputs are not field validation.** The example row is synthetic
> and cannot validate real-world performance.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def _write_sha256sums(out_dir: Path) -> None:
    lines = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  ./{path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    created_at = _now_iso()

    print("=" * 72)
    print("AquaScan National Groundwater Depth Model v1")
    print("=" * 72)

    data = build_national_depth_frame(args.dataset, args.firewall)
    metadata = data.metadata
    population = metadata["modeling_population"]
    dataset = metadata["dataset"]
    print(
        f"\n[data] snapshot={dataset['snapshot_records']:,} | "
        f"modeling={population['records']:,} | "
        f"provinces={population['provinces']} | domains={population['domains']} | "
        f"proximity_groups={population['proximity_groups']:,}"
    )
    print(
        f"[data] Phase 0 firewall: {metadata['phase0_firewall']['n_ready_excluded']} "
        f"READY records excluded"
    )

    if args.evaluate:
        cv, domain_rows, oof_payload = _evaluate(data, args)
    else:
        if not args.reuse_evaluation:
            print(
                "\nNo mode selected: reusing cached evaluation. "
                "Pass --evaluate to rerun the full protocol."
            )
        cv, domain_rows, oof_payload = _load_cached_evaluation(args, data)

    cv_summary = _summarize_cv(cv)
    domain_summary = _summarize_domain_holdout(domain_rows)
    conformal = _conformal_summary(oof_payload, args.alpha)
    ood_policy = build_ood_policy(data.frame)
    confidence_policy = build_confidence_policy(
        conformal=conformal,
        ood_policy=ood_policy,
        target_min_m=metadata["target"]["min_m"],
        target_max_m=metadata["target"]["max_m"],
    )

    print("\n[metrics]")
    for split_column, label in SPLITS:
        row = cv_summary[split_column]["models"]["RandomForest"]
        dummy = cv_summary[split_column]["models"]["Dummy"]
        print(
            f"  {label}: RF median MAE={row['mae_median_m']:.3f} m | "
            f"Dummy={dummy['mae_median_m']:.3f} m"
        )
    print(
        f"  Domain holdout: RF weighted MAE={domain_summary['rf']['mae_weighted_m']:.3f} m | "
        f"Dummy={domain_summary['dummy']['mae_weighted_m']:.3f} m | "
        f"RF better in {domain_summary['rf_better_than_dummy_in_domains']}/"
        f"{domain_summary['n_domains']}"
    )
    print(
        f"  Conformal: +/- {conformal['interval_half_width_m']:.3f} m "
        f"(alpha={args.alpha}, coverage={conformal['empirical_coverage']:.4f})"
    )

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} already exists. Pass --force to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"\n[fit] training final RandomForest on {len(data.frame):,} records ...")
    model, preprocessor = train_final_model(data.frame, seed=args.seed)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    feature_manifest = {
        "features": FEATURES,
        "n_features": len(FEATURES),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "target_source": "temporal_summary_json.median_depth_to_water_m",
        "fold_grouping_columns": [FOLD_GROUP_COLUMN, SPATIAL_CELL_COLUMN],
        "fold_grouping_note": ("Not model inputs. Used only for leakage-safe evaluation."),
    }
    (out_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    uncertainty_calibrator = {
        "method": conformal["method"],
        "alpha": conformal["alpha"],
        "calibration_set": conformal["calibration_set"],
        "n_calibration": conformal["n_calibration"],
        "calibration_quantile_abs_residual_m": conformal["calibration_quantile_abs_residual_m"],
        "interval_half_width_m": conformal["interval_half_width_m"],
        "interval_width_m": conformal["interval_width_m"],
        "empirical_coverage": conformal["empirical_coverage"],
        "note": conformal["note"],
    }
    (out_dir / "uncertainty_calibrator.json").write_text(
        json.dumps(uncertainty_calibrator, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "ood_policy_reference.json").write_text(
        json.dumps(ood_policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "confidence_policy.json").write_text(
        json.dumps(confidence_policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = _build_metrics(data, cv_summary, domain_summary, conformal, created_at)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_example(out_dir, conformal["interval_half_width_m"])
    _write_requirements(out_dir)
    (out_dir / ".gitattributes").write_text(
        "*.joblib filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    shutil.copyfile(ROOT / "LICENSE", out_dir / "LICENSE")

    _write_model_card(
        out_dir,
        data,
        cv_summary,
        domain_summary,
        conformal,
        ood_policy,
        created_at,
        args.seed,
    )

    artifact_names = [
        "model.joblib",
        "preprocessor.joblib",
        "feature_manifest.json",
        "uncertainty_calibrator.json",
        "ood_policy_reference.json",
        "confidence_policy.json",
        "metrics.json",
        "example.py",
        "requirements.txt",
        "README.md",
        ".gitattributes",
        "LICENSE",
    ]
    artifact_hashes = {name: sha256_file(out_dir / name) for name in artifact_names}
    training_manifest = _build_training_manifest(
        data,
        args,
        created_at,
        artifact_hashes,
        [CV_REPORT, DOMAIN_REPORT, OOF_REPORT],
    )
    (out_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_sha256sums(out_dir)

    print(f"\n[artifacts] written to {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")
    print("\n" + "=" * 72)
    print("DONE — review the model card before uploading.")
    print("=" * 72)


if __name__ == "__main__":
    main()
