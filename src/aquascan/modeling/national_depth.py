"""National depth-to-water model (AquaScan National v1).

This module builds the national modeling frame from the IGME national
dataset, defines the leakage-safe evaluation protocol, and trains the
published RandomForest.

Scientific contract
-------------------
* Target: median historical depth to water from READY documentary records.
* Inputs: 21 documentary/contextual features. Target-derived columns
  (``temporal_summary_json``, ``readiness_*``) are never used as inputs.
* Evaluation groups: 500 m proximity clusters, province, and hydrogeological
  domain. The same site never appears in both train and test.
* Phase 0 firewall: the 10 sealed San Antonio sites in
  ``reports/phase0_cohort_A_v1.csv`` are excluded before grouping and
  training.
* Uncertainty: split-conformal absolute residual on out-of-fold predictions.

The model is a research baseline, not a groundwater detector.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET_READINESS = "READY"
READINESS_COLUMN = "readiness_TARGET_DEPTH_TO_WATER"
READINESS_QUALITY_COLUMN = "readiness_TARGET_DEPTH_TO_WATER_quality"
TARGET_JSON_COLUMN = "temporal_summary_json"
TARGET_COLUMN = "_target_depth_to_water_m"
FOLD_GROUP_COLUMN = "group_proximity_500m"
SPATIAL_CELL_COLUMN = "group_spatial_cell"

# 21-feature contract. The two grouping identifiers (proximity cluster and
# spatial cell) are built for leakage-safe evaluation but are NOT inputs.
FEATURES = [
    "aquifer_system",
    "construction_date",
    "drilling_method",
    "elevation_m",
    "group_aquifer",
    "group_domain",
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
]

NUMERIC_FEATURES = [
    "elevation_m",
    "latitude",
    "longitude",
    "qa_coordinate_outlier",
    "qa_depth_outlier",
    "qa_duplicate_coordinate",
    "qa_elevation_outlier",
    "qa_flag_count",
    "total_depth_m",
]
CATEGORICAL_FEATURES = [feature for feature in FEATURES if feature not in NUMERIC_FEATURES]

SOURCE_COLUMNS = [
    "igme_id",
    "readiness_TARGET_DEPTH_TO_WATER",
    "readiness_TARGET_DEPTH_TO_WATER_quality",
    "temporal_summary_json",
    "domain",
    "source_aquifer_system",
    "source_construction_date",
    "source_drilling_method",
    "source_elevation_m",
    "source_latitude",
    "source_longitude",
    "source_municipality",
    "source_point_nature",
    "source_province",
    "source_pump_type",
    "source_qa_flag_count",
    "source_total_depth_m",
    "source_water_use",
    "source_watershed",
]


@dataclass(frozen=True)
class NationalDepthData:
    """Modeling frame plus auditable provenance metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_from_temporal_summary(value: Any) -> float:
    """Extract the median depth-to-water from a temporal summary JSON string."""
    if not isinstance(value, str):
        return math.nan
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return math.nan
    if not isinstance(payload, dict):
        return math.nan
    target = payload.get("median_depth_to_water_m")
    try:
        return float(target)
    except (TypeError, ValueError):
        return math.nan


def _cluster_labels(n_points: int, pairs: np.ndarray) -> list[int]:
    """Sequential cluster labels from union-find over connected pairs."""
    parent = list(range(n_points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for left, right in pairs:
        union(int(left), int(right))

    roots: dict[int, int] = {}
    labels: list[int] = []
    for position in range(n_points):
        root = find(position)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels


def proximity_groups(frame: pd.DataFrame, radius_m: float = 500.0) -> pd.Series:
    """Cluster records within ``radius_m`` metres, separately per domain.

    The implementation is a deterministic union-find over a KD-tree. Cluster
    identifiers are sequential in order of first appearance, which is the
    exact convention used by the frozen national evaluation. Records with
    missing coordinates become unique singleton groups rather than being
    silently merged.

    Grouping is an evaluation device only: nearby or duplicate records must
    never be split across train and test.
    """
    required = {"group_domain", "latitude", "longitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"proximity grouping requires columns: {sorted(missing)}")

    result = pd.Series(index=frame.index, dtype=object)
    for domain, part in frame.groupby("group_domain", sort=False):
        n_rows = len(part)
        latitude = pd.to_numeric(part["latitude"], errors="coerce").to_numpy(dtype=float)
        longitude = pd.to_numeric(part["longitude"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(latitude) & np.isfinite(longitude)
        labels: list[str | None] = [None] * n_rows

        valid_positions = np.flatnonzero(valid)
        if len(valid_positions) == 1:
            labels[int(valid_positions[0])] = f"{domain}_p0"
        elif len(valid_positions) > 1:
            mean_lat = float(np.mean(latitude[valid_positions]))
            x = longitude[valid_positions] * 111_320.0 * math.cos(math.radians(mean_lat))
            y = latitude[valid_positions] * 110_540.0
            tree = cKDTree(np.column_stack([x, y]))
            pairs = tree.query_pairs(r=radius_m, output_type="ndarray")
            cluster_ids = _cluster_labels(len(valid_positions), pairs)
            for position, cluster_id in zip(valid_positions, cluster_ids, strict=True):
                labels[int(position)] = f"{domain}_p{cluster_id}"

        # Missing coordinates cannot be safely assigned to a spatial cluster.
        for position in np.flatnonzero(~valid):
            labels[int(position)] = f"{domain}_p_missing_{int(position)}"

        result.loc[part.index] = labels

    return result


def build_national_depth_frame(
    dataset_path: Path,
    firewall_path: Path | None = None,
) -> NationalDepthData:
    """Build the 21-feature national depth-to-water frame.

    Args:
        dataset_path: National parquet written by the national enrichment
            finalizer.
        firewall_path: CSV with the sealed Phase 0 cohort (``igme_id``
            column). When provided, those records are excluded before
            grouping, evaluation and training.

    Returns:
        ``NationalDepthData`` with the modeling frame and provenance metadata.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    snapshot = pd.read_parquet(dataset_path, columns=SOURCE_COLUMNS)
    snapshot_records = int(len(snapshot))

    firewall_ids: set[str] = set()
    firewall_meta: dict[str, Any] = {
        "status": "not_applied",
        "source": None,
        "n_ids": 0,
        "n_ready_excluded": 0,
    }
    if firewall_path is not None:
        firewall_path = Path(firewall_path)
        if not firewall_path.exists():
            raise FileNotFoundError(firewall_path)
        firewall_ids = set(
            pd.read_csv(firewall_path, usecols=["igme_id"])["igme_id"].astype(str).tolist()
        )
        firewall_meta = {
            "status": "active",
            "source": firewall_path.name,
            "source_sha256": sha256_file(firewall_path),
            "n_ids": len(firewall_ids),
            "n_ready_excluded": 0,
        }

    ready_mask = snapshot[READINESS_COLUMN] == TARGET_READINESS
    ready_records = int(ready_mask.sum())
    firewalled_mask = snapshot["igme_id"].astype(str).isin(firewall_ids)
    firewalled_ready = int((ready_mask & firewalled_mask).sum())
    firewall_meta["n_ready_excluded"] = firewalled_ready

    selected = snapshot[ready_mask & ~firewalled_mask].copy()
    selected[TARGET_COLUMN] = selected[TARGET_JSON_COLUMN].map(_target_from_temporal_summary)
    missing_target = int(selected[TARGET_COLUMN].isna().sum())
    selected = selected[selected[TARGET_COLUMN].notna()].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("no READY depth-to-water records available after filtering")

    # Duplicate coordinates are evaluated over the full snapshot, not only the
    # modeling rows, so the flag does not depend on the filter.
    duplicate_mask = snapshot.duplicated(subset=["source_latitude", "source_longitude"], keep=False)
    duplicate_ids = set(snapshot.loc[duplicate_mask, "igme_id"].astype(str).tolist())

    out = pd.DataFrame(index=selected.index)
    # Audit-only identifier. It is never part of FEATURES and never written to
    # the published artifacts.
    out["igme_id"] = selected["igme_id"].astype(str)
    out["aquifer_system"] = selected["source_aquifer_system"].fillna("unknown").astype(str)
    out["construction_date"] = selected["source_construction_date"].fillna("unknown").astype(str)
    out["drilling_method"] = selected["source_drilling_method"].fillna("unknown").astype(str)
    out["elevation_m"] = pd.to_numeric(selected["source_elevation_m"], errors="coerce")
    out["group_aquifer"] = out["aquifer_system"]
    out["group_domain"] = selected["domain"].fillna("unknown").astype(str)
    out["group_watershed"] = selected["source_watershed"].fillna("unknown").astype(str)
    out["latitude"] = pd.to_numeric(selected["source_latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(selected["source_longitude"], errors="coerce")
    out["municipality"] = selected["source_municipality"].fillna("unknown").astype(str)
    out["point_nature"] = selected["source_point_nature"].fillna("unknown").astype(str)
    out["province"] = selected["source_province"].fillna("unknown").astype(str)
    out["pump_type"] = selected["source_pump_type"].fillna("unknown").astype(str)
    out["qa_coordinate_outlier"] = (
        (out["longitude"] < -19)
        | (out["longitude"] > 5)
        | (out["latitude"] < 27)
        | (out["latitude"] > 44.5)
    ).astype(int)
    out["qa_elevation_outlier"] = (
        (out["elevation_m"] < -100) | (out["elevation_m"] > 4000)
    ).astype(int)
    out["qa_depth_outlier"] = (
        pd.to_numeric(selected["source_total_depth_m"], errors="coerce") > 3000
    ).astype(int)
    out["qa_duplicate_coordinate"] = selected["igme_id"].astype(str).isin(duplicate_ids).astype(int)
    out["qa_flag_count"] = (
        pd.to_numeric(selected["source_qa_flag_count"], errors="coerce").fillna(0).astype(int)
    )
    out["total_depth_m"] = pd.to_numeric(selected["source_total_depth_m"], errors="coerce")
    out["water_use"] = selected["source_water_use"].fillna("unknown").astype(str)
    out["watershed"] = out["group_watershed"]
    out[SPATIAL_CELL_COLUMN] = (
        out["latitude"].round(2).astype(str) + "_" + out["longitude"].round(2).astype(str)
    )
    out[FOLD_GROUP_COLUMN] = proximity_groups(out)
    out[TARGET_COLUMN] = selected[TARGET_COLUMN].astype(float)

    target = out[TARGET_COLUMN].to_numpy(dtype=float)
    readiness_quality = selected[READINESS_QUALITY_COLUMN].value_counts(dropna=False)
    metadata: dict[str, Any] = {
        "dataset": {
            "path": dataset_path.name,
            "sha256": sha256_file(dataset_path),
            "snapshot_records": snapshot_records,
            "snapshot_provinces": int(snapshot["source_province"].nunique(dropna=True)),
            "snapshot_domains": int(snapshot["domain"].nunique(dropna=True)),
        },
        "phase0_firewall": firewall_meta,
        "modeling_population": {
            "records": int(len(out)),
            "provinces": int(out["province"].nunique(dropna=True)),
            "domains": int(out["group_domain"].nunique(dropna=True)),
            "proximity_groups": int(out[FOLD_GROUP_COLUMN].nunique(dropna=True)),
            "spatial_cells": int(out[SPATIAL_CELL_COLUMN].nunique(dropna=True)),
            "ready_records_before_firewall": ready_records,
            "firewalled_ready_records": firewalled_ready,
            "missing_target_records": missing_target,
        },
        "target": {
            "column": TARGET_COLUMN,
            "source": "temporal_summary_json.median_depth_to_water_m",
            "readiness": TARGET_READINESS,
            "n": int(len(target)),
            "min_m": float(np.min(target)),
            "q25_m": float(np.quantile(target, 0.25)),
            "median_m": float(np.median(target)),
            "q75_m": float(np.quantile(target, 0.75)),
            "max_m": float(np.max(target)),
            "mean_m": float(np.mean(target)),
            "std_m": float(np.std(target, ddof=1)),
        },
        "readiness_quality": {str(key): int(value) for key, value in readiness_quality.items()},
        "source_missingness": {
            column: int(selected[column].isna().sum()) for column in SOURCE_COLUMNS
        },
    }
    return NationalDepthData(frame=out, metadata=metadata)


def make_preprocessor() -> ColumnTransformer:
    """Canonical preprocessing: median/scale numeric, one-hot categorical."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def make_model(name: str) -> Any:
    """Build an unfitted model by name for the evaluation protocol."""
    if name == "Dummy":
        return DummyRegressor(strategy="median")
    if name == "Ridge":
        return Ridge(alpha=10.0)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Point-regression metrics used throughout the evaluation."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(residuals)),
        "mae": float(np.mean(np.abs(residuals))),
        "medae": float(np.median(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _make_model_with_seed(name: str, seed: int) -> Any:
    model = make_model(name)
    if seed != SEED and "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model


def run_grouped_cv(
    frame: pd.DataFrame,
    group_column: str,
    model_name: str,
    *,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run GroupKFold cross-validation and return fold metrics plus OOF."""
    if group_column not in frame.columns:
        raise ValueError(f"missing group column: {group_column}")
    features = frame[FEATURES]
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    groups = frame[group_column].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    oof = np.full(len(frame), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target, groups=groups)):
        started = time.perf_counter()
        preprocessor = make_preprocessor()
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_test = preprocessor.transform(features.iloc[test_idx])
        model = _make_model_with_seed(model_name, seed)
        model.fit(x_train, target[train_idx])
        prediction = model.predict(x_test)
        oof[test_idx] = prediction
        metrics = regression_metrics(target[test_idx], prediction)
        metrics.update(
            {
                "fold": fold,
                "model": model_name,
                "split": group_column,
                "seconds": round(time.perf_counter() - started, 1),
                "seed": seed,
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows), oof


def run_domain_holdout(
    frame: pd.DataFrame,
    *,
    min_records: int = 150,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Leave-one-domain-out evaluation for domains with enough records.

    Domains below ``min_records`` are not individually scored: their test set
    would be too small to support a domain-level claim. They remain in the
    national training population and are reported as weak support in the OOD
    policy.
    """
    domain_counts = frame["group_domain"].value_counts()
    domains = domain_counts[domain_counts >= min_records].index.tolist()
    features = frame[FEATURES]
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for domain in domains:
        test_mask = (frame["group_domain"] == domain).to_numpy()
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)

        preprocessor = make_preprocessor()
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_test = preprocessor.transform(features.iloc[test_idx])

        random_forest = _make_model_with_seed("RandomForest", seed)
        random_forest.fit(x_train, target[train_idx])
        dummy = _make_model_with_seed("Dummy", seed)
        dummy.fit(x_train, target[train_idx])

        rows.append(
            {
                "domain": str(domain),
                "n_test": int(len(test_idx)),
                "rf": regression_metrics(target[test_idx], random_forest.predict(x_test)),
                "dummy": regression_metrics(target[test_idx], dummy.predict(x_test)),
                "seed": seed,
            }
        )

    return rows


def conformal_quantile(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal absolute-residual quantile at ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return float(np.quantile(residuals, 1.0 - alpha, method="higher"))


def build_ood_policy(frame: pd.DataFrame, *, min_support: int = 30) -> dict[str, Any]:
    """Build a pre-prediction domain/province support policy.

    Support is computed from feature counts only. It never uses the target
    outcomes of the sample being predicted.
    """

    def support_class(n_records: int) -> str:
        if n_records >= min_support:
            return "IN_DISTRIBUTION"
        if n_records > 0:
            return "WEAK_SUPPORT"
        return "OUT_OF_DISTRIBUTION"

    domain_counts = frame["group_domain"].value_counts()
    province_counts = frame["province"].value_counts()

    per_domain = {
        str(domain): {
            "n_training_records": int(count),
            "support": support_class(int(count)),
        }
        for domain, count in domain_counts.items()
    }
    per_province = {
        str(province): {
            "n_training_records": int(count),
            "support": support_class(int(count)),
        }
        for province, count in province_counts.items()
    }

    return {
        "version": "national_v1",
        "description": (
            "Pre-prediction domain/province support assessment for the national "
            "depth-to-water model. Support is computed from training feature "
            "counts only; it never uses target errors of the new sample."
        ),
        "classes": {
            "IN_DISTRIBUTION": f">= {min_support} training records",
            "WEAK_SUPPORT": f"1-{min_support - 1} training records",
            "OUT_OF_DISTRIBUTION": "0 training records (unseen category)",
        },
        "thresholds": {
            "n_min_in_domain": min_support,
            "n_min_in_province": min_support,
        },
        "summary": {
            "n_domains": int(len(per_domain)),
            "n_domains_in_distribution": int(
                sum(v["support"] == "IN_DISTRIBUTION" for v in per_domain.values())
            ),
            "n_domains_weak_support": int(
                sum(v["support"] == "WEAK_SUPPORT" for v in per_domain.values())
            ),
            "n_domains_out_of_distribution": int(
                sum(v["support"] == "OUT_OF_DISTRIBUTION" for v in per_domain.values())
            ),
            "n_provinces": int(len(per_province)),
        },
        "per_domain": per_domain,
        "per_province": per_province,
        "note": (
            "Domain support is not the same as validated performance. The "
            "domain-holdout evaluation only covers domains with >= 150 records; "
            "weak-support domains have no domain-level validation."
        ),
    }


def build_confidence_policy(
    *,
    conformal: dict[str, Any],
    ood_policy: dict[str, Any],
    target_min_m: float,
    target_max_m: float,
) -> dict[str, Any]:
    """Define the GREEN/YELLOW/RED triage gate.

    The gate deliberately uses only feature support, the prediction and the
    training target range. Tree-dispersion thresholds are not invented here;
    the split-conformal interval is the quantified uncertainty.
    """
    return {
        "version": "national_v1",
        "method": ("OOD support + training-range extrapolation guard + split-conformal interval"),
        "gate": {
            "GREEN": (
                "domain support IN_DISTRIBUTION, province support "
                "IN_DISTRIBUTION, and prediction inside the training target range"
            ),
            "YELLOW": ("domain WEAK_SUPPORT or prediction outside the training target range"),
            "RED": "domain or province OUT_OF_DISTRIBUTION (ABSTAIN)",
        },
        "conformal": conformal,
        "thresholds": ood_policy["thresholds"],
        "training_target_range_m": [float(target_min_m), float(target_max_m)],
        "note": (
            "The gate never uses observed target errors of the new sample. "
            "It is a support triage rule, not a calibrated probability. "
            "The conformal interval is marginal (global), not per-site."
        ),
    }


def train_final_model(
    frame: pd.DataFrame,
    *,
    seed: int = SEED,
) -> tuple[RandomForestRegressor, ColumnTransformer]:
    """Fit the published preprocessor and RandomForest on all modeling rows."""
    features = frame[FEATURES]
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(features)
    model = make_model("RandomForest")
    if seed != SEED:
        model.set_params(random_state=seed)
    model.fit(x_train, target)
    return model, preprocessor
