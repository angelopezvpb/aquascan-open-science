"""No-leakage and provenance tests for the published national model v1.

These tests verify the properties that matter scientifically:

* the Phase 0 firewall is applied before grouping/training;
* no target-derived column is a model input;
* proximity, province and domain groups never cross train/test;
* cached evaluation metrics and conformal calibration are reproducible;
* published artifacts contain no record-level IGME data;
* the model card states the known limitations and scope.

They are intentionally independent of the final metric values: metrics are
recomputed from the cached reports and compared, never asserted from memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from aquascan.modeling.national_depth import (
    FEATURES,
    FOLD_GROUP_COLUMN,
    SPATIAL_CELL_COLUMN,
    TARGET_COLUMN,
    build_national_depth_frame,
    conformal_quantile,
    proximity_groups,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/national_v1"
DATASET = ROOT / "data/processed/aquascan_national_dataset_v1.parquet"
FIREWALL = ROOT / "reports/phase0_cohort_A_v1.csv"
CV_REPORT = ROOT / "reports/national_model_v1_cv.csv"
DOMAIN_REPORT = ROOT / "reports/national_model_v1_domain_holdout.json"
OOF_REPORT = ROOT / "reports/national_model_v1_oof.npz"
NATIONAL_MANIFEST = ROOT / "reports/national_enrichment_manifest_v1.json"

# The national dataset and fitted weights are intentionally not redistributed.
# Public CI must skip this module instead of failing on missing private data.
_REQUIRED_ARTIFACTS = (
    DATASET,
    MODEL_DIR / "model.joblib",
    MODEL_DIR / "preprocessor.joblib",
    CV_REPORT,
    DOMAIN_REPORT,
    OOF_REPORT,
)
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED_ARTIFACTS),
    reason="national model tests require the private IGME snapshot and generated artifacts",
)

FORBIDDEN_PREFIXES = (
    "_target_",
    "target_",
    "gt_",
    "readiness_",
    "phys_val_",
    "candidate_",
)
FORBIDDEN_EXACT = {
    "temporal_summary_json",
    "igme_id",
    "group_proximity_500m",
    "group_spatial_cell",
    "n_in_domain",
    "n_in_proximity_group",
    "candidate_priority_score",
    "candidate_tier",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return build_national_depth_frame(DATASET, FIREWALL)


@pytest.fixture(scope="module")
def frame(data):
    return data.frame


@pytest.fixture(scope="module")
def metrics() -> dict:
    return _load_json(MODEL_DIR / "metrics.json")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _load_json(MODEL_DIR / "training_manifest.json")


@pytest.fixture(scope="module")
def ood_policy() -> dict:
    return _load_json(MODEL_DIR / "ood_policy_reference.json")


@pytest.fixture(scope="module")
def card() -> str:
    return (MODEL_DIR / "README.md").read_text(encoding="utf-8")


# ── 1. Phase 0 firewall ───────────────────────────────────────────────


def test_phase0_firewall_is_active_and_relevant(data) -> None:
    firewall_ids = set(pd.read_csv(FIREWALL)["igme_id"].astype(str))
    snapshot = pd.read_parquet(
        DATASET,
        columns=["igme_id", "readiness_TARGET_DEPTH_TO_WATER"],
    )
    ready_ids = set(
        snapshot.loc[snapshot["readiness_TARGET_DEPTH_TO_WATER"] == "READY", "igme_id"].astype(str)
    )
    # The firewall would otherwise be part of the READY modeling population.
    assert firewall_ids <= ready_ids
    assert data.metadata["phase0_firewall"]["status"] == "active"
    assert data.metadata["phase0_firewall"]["n_ids"] == len(firewall_ids)
    assert data.metadata["phase0_firewall"]["n_ready_excluded"] == len(firewall_ids)


def test_no_firewalled_ids_in_modeling_frame(frame) -> None:
    firewall_ids = set(pd.read_csv(FIREWALL)["igme_id"].astype(str))
    assert not set(frame["igme_id"]) & firewall_ids
    assert len(frame) == 16_722


def test_modeling_population_reconciles(frame, data) -> None:
    population = data.metadata["modeling_population"]
    assert population["records"] == len(frame)
    assert (
        population["ready_records_before_firewall"] - population["firewalled_ready_records"]
        == population["records"]
    )
    assert population["provinces"] == 44
    assert population["domains"] == 323


# ── 2. Feature contract ───────────────────────────────────────────────


def test_feature_contract_is_exact() -> None:
    manifest = _load_json(MODEL_DIR / "feature_manifest.json")
    assert manifest["n_features"] == 21
    assert manifest["features"] == FEATURES
    assert manifest["target"] == TARGET_COLUMN
    assert manifest["fold_grouping_columns"] == [FOLD_GROUP_COLUMN, SPATIAL_CELL_COLUMN]


def test_no_forbidden_columns_in_features() -> None:
    features = set(FEATURES)
    assert len(features) == 21
    for column in features:
        assert not column.startswith(FORBIDDEN_PREFIXES), column
        assert column not in FORBIDDEN_EXACT, column
    assert "group_proximity_500m" not in features
    assert "group_spatial_cell" not in features
    assert "igme_id" not in features
    assert "temporal_summary_json" not in features


def test_preprocessor_accepts_only_the_contract(frame) -> None:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    transformed = preprocessor.transform(frame[FEATURES].iloc[[0]])
    assert transformed.shape[0] == 1
    with pytest.raises(Exception):
        preprocessor.transform(frame[FEATURES].iloc[[0]].drop(columns=["province"]))


# ── 3. Group isolation ────────────────────────────────────────────────


def test_proximity_grouping_merges_nearby_points() -> None:
    points = pd.DataFrame(
        {
            "group_domain": ["A", "A", "A", "B", "A"],
            "latitude": [0.0, 0.0005, 0.01, 0.0, np.nan],
            "longitude": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    groups = proximity_groups(points)
    assert groups.iloc[0] == groups.iloc[1]  # ~55 m apart, same domain
    assert groups.iloc[0] != groups.iloc[2]  # ~1.1 km apart
    assert groups.iloc[0] != groups.iloc[3]  # same coordinates, different domain
    assert groups.iloc[4] != groups.iloc[0]  # missing coordinates are singleton
    assert groups.iloc[4] != groups.iloc[3]


def test_grouped_cv_has_no_proximity_overlap(frame) -> None:
    splitter = GroupKFold(n_splits=5)
    groups = frame[FOLD_GROUP_COLUMN].astype(str).to_numpy()
    y = frame[TARGET_COLUMN].to_numpy()
    seen = np.zeros(len(frame), dtype=int)
    for train_idx, test_idx in splitter.split(frame[FEATURES], y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert not train_groups & test_groups
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_province_holdout_has_no_province_overlap(frame) -> None:
    splitter = GroupKFold(n_splits=5)
    provinces = frame["province"].astype(str).to_numpy()
    y = frame[TARGET_COLUMN].to_numpy()
    seen = np.zeros(len(frame), dtype=int)
    for train_idx, test_idx in splitter.split(frame[FEATURES], y, groups=provinces):
        assert not set(provinces[train_idx]) & set(provinces[test_idx])
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_domain_holdout_protocol_matches_report(frame, metrics) -> None:
    domain_summary = metrics["evaluation"]["domain_holdout"]
    scored = domain_summary["domains"]
    expected_domains = set(
        frame["group_domain"].value_counts().loc[lambda s: s >= 150].index.astype(str)
    )
    assert {row["domain"] for row in scored} == expected_domains
    for row in scored:
        assert row["n_test"] == int((frame["group_domain"] == row["domain"]).sum())
        assert row["n_test"] >= 150
    assert domain_summary["n_records"] == sum(row["n_test"] for row in scored)


# ── 4. Cached evaluation reproducibility ──────────────────────────────


def test_oof_matches_modeling_frame(frame) -> None:
    with np.load(OOF_REPORT, allow_pickle=True) as payload:
        oof = payload["oof"]
        target = payload["y"]
        groups = payload["groups"].astype(str)
    assert len(oof) == len(frame)
    assert np.isfinite(oof).all()
    assert np.allclose(target, frame[TARGET_COLUMN].to_numpy(), rtol=0.0, atol=1e-9)
    assert np.array_equal(groups, frame[FOLD_GROUP_COLUMN].astype(str).to_numpy())


def test_cv_report_has_expected_design() -> None:
    cv = pd.read_csv(CV_REPORT)
    assert set(cv["split"]) == {"group_proximity_500m", "province"}
    assert set(cv["model"]) == {"Dummy", "Ridge", "RandomForest", "ExtraTrees"}
    for (split, model), group in cv.groupby(["split", "model"]):
        assert len(group) == 5, (split, model)
        assert set(group["fold"]) == set(range(5))


def test_metrics_reproduce_cv_report(metrics) -> None:
    cv = pd.read_csv(CV_REPORT)
    evaluation = metrics["evaluation"]
    for split, key in (
        ("group_proximity_500m", "grouped_cv_proximity_500m"),
        ("province", "province_holdout"),
    ):
        for model in ("Dummy", "Ridge", "RandomForest", "ExtraTrees"):
            subset = cv[(cv["split"] == split) & (cv["model"] == model)]
            reported = evaluation[key]["models"][model]
            assert reported["n_folds"] == 5
            assert abs(reported["mae_median_m"] - subset["mae"].median()) < 1e-9
            assert abs(reported["mae_mean_m"] - subset["mae"].mean()) < 1e-9
            assert abs(reported["rmse_median_m"] - subset["rmse"].median()) < 1e-9


def test_metrics_reproduce_domain_holdout(metrics) -> None:
    report = json.loads(DOMAIN_REPORT.read_text(encoding="utf-8"))
    summary = metrics["evaluation"]["domain_holdout"]
    assert [row["domain"] for row in summary["domains"]] == [row["domain"] for row in report]
    weights = np.asarray([row["n_test"] for row in report], dtype=float)
    rf_mae = np.asarray([row["rf"]["mae"] for row in report], dtype=float)
    dummy_mae = np.asarray([row["dummy"]["mae"] for row in report], dtype=float)
    assert abs(summary["rf"]["mae_weighted_m"] - np.average(rf_mae, weights=weights)) < 1e-9
    assert abs(summary["dummy"]["mae_weighted_m"] - np.average(dummy_mae, weights=weights)) < 1e-9
    assert summary["rf_better_than_dummy_in_domains"] == int(np.sum(rf_mae < dummy_mae))


def test_conformal_calibration_is_reproducible() -> None:
    calibrator = _load_json(MODEL_DIR / "uncertainty_calibrator.json")
    with np.load(OOF_REPORT, allow_pickle=True) as payload:
        oof = payload["oof"]
        target = payload["y"]
    residuals = np.abs(target - oof)
    expected = conformal_quantile(target, oof, calibrator["alpha"])
    assert abs(calibrator["calibration_quantile_abs_residual_m"] - expected) < 1e-9
    assert abs(calibrator["interval_width_m"] - 2.0 * expected) < 1e-9
    coverage = float(np.mean(residuals <= expected))
    assert abs(calibrator["empirical_coverage"] - coverage) < 1e-12
    assert 0.895 <= coverage <= 0.905
    assert calibrator["n_calibration"] == len(target)


def test_confidence_policy_uses_the_same_calibration() -> None:
    calibrator = _load_json(MODEL_DIR / "uncertainty_calibrator.json")
    policy = _load_json(MODEL_DIR / "confidence_policy.json")
    assert policy["conformal"]["alpha"] == calibrator["alpha"]
    assert (
        policy["conformal"]["calibration_quantile_abs_residual_m"]
        == calibrator["calibration_quantile_abs_residual_m"]
    )
    assert "never uses observed target errors" in policy["note"]


# ── 5. OOD policy ─────────────────────────────────────────────────────


def test_ood_policy_reconciles_with_training_population(frame, ood_policy) -> None:
    assert ood_policy["version"] == "national_v1"
    assert ood_policy["summary"]["n_domains"] == 323
    assert ood_policy["summary"]["n_provinces"] == 44
    assert sum(value["n_training_records"] for value in ood_policy["per_domain"].values()) == len(
        frame
    )
    assert sum(value["n_training_records"] for value in ood_policy["per_province"].values()) == len(
        frame
    )
    for name, value in ood_policy["per_domain"].items():
        assert name in set(frame["group_domain"])
        if value["n_training_records"] >= 30:
            assert value["support"] == "IN_DISTRIBUTION"
        elif value["n_training_records"] > 0:
            assert value["support"] == "WEAK_SUPPORT"
        else:
            assert value["support"] == "OUT_OF_DISTRIBUTION"


def test_ood_summary_is_consistent(ood_policy) -> None:
    summary = ood_policy["summary"]
    assert summary["n_domains_in_distribution"] + summary["n_domains_weak_support"] == 323
    assert summary["n_domains_out_of_distribution"] == 0
    assert ood_policy["classes"]["IN_DISTRIBUTION"] == ">= 30 training records"


# ── 6. Serialization / provenance ────────────────────────────────────


def test_required_artifacts_exist() -> None:
    required = {
        "model.joblib",
        "preprocessor.joblib",
        "feature_manifest.json",
        "confidence_policy.json",
        "uncertainty_calibrator.json",
        "ood_policy_reference.json",
        "metrics.json",
        "training_manifest.json",
        "example.py",
        "requirements.txt",
        "SHA256SUMS",
        "README.md",
        ".gitattributes",
        "LICENSE",
    }
    assert required <= {path.name for path in MODEL_DIR.iterdir()}


def test_sha256sums_verify() -> None:
    lines = (MODEL_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines
    pattern = re.compile(r"^([0-9a-f]{64})  \./(.+)$")
    seen = set()
    for line in lines:
        match = pattern.match(line)
        assert match, line
        digest, name = match.groups()
        assert name not in seen, name
        seen.add(name)
        path = MODEL_DIR / name
        assert path.exists(), name
        assert _sha256_file(path) == digest, name
    assert "SHA256SUMS" not in seen


def test_training_manifest_hashes_artifacts(manifest) -> None:
    for name, digest in manifest["model_artifacts"].items():
        assert (MODEL_DIR / name).exists(), name
        assert _sha256_file(MODEL_DIR / name) == digest, name


def test_training_manifest_provenance(manifest, frame) -> None:
    actual_dataset_hash = _sha256_file(DATASET)
    assert manifest["dataset"]["sha256"] == actual_dataset_hash
    assert manifest["dataset"]["snapshot_records"] == 139_697
    assert manifest["dataset"]["snapshot_provinces"] == 48
    assert manifest["dataset"]["snapshot_domains"] == 401
    assert manifest["modeling_population"]["records"] == len(frame)
    assert manifest["modeling_population"]["records"] == 16_722
    assert manifest["phase0_firewall"]["status"] == "active"
    assert manifest["phase0_firewall"]["n_ids"] == 10
    assert manifest["phase0_firewall"]["n_ready_excluded"] == 10
    assert manifest["random_seed"] == 42
    assert manifest["record_level_data_published"] is False
    assert manifest["provenance"]["source_hash"] == actual_dataset_hash
    assert manifest["feature_contract"]["n_features"] == 21
    assert manifest["training"]["n_training_records"] == len(frame)
    assert manifest["training"]["hyperparameters"]["random_state"] == 42
    national = _load_json(NATIONAL_MANIFEST)
    assert national["dataset_sha256"] == actual_dataset_hash


def test_source_hashes_verify(manifest) -> None:
    for name, digest in manifest["source_files"].items():
        path = ROOT / name
        assert path.exists(), name
        assert _sha256_file(path) == digest, name


# ── 7. Model artifact works ───────────────────────────────────────────


def test_model_predicts_from_contract(frame) -> None:
    model = joblib.load(MODEL_DIR / "model.joblib")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    prediction = model.predict(preprocessor.transform(frame[FEATURES].iloc[[0]]))
    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_example_runs() -> None:
    result = subprocess.run(
        [sys.executable, "example.py"],
        cwd=MODEL_DIR,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "synthetic input" in result.stdout.lower()
    assert "Predicted historical depth to water" in result.stdout
    assert "not a groundwater detector" in result.stdout


# ── 8. No record-level data in published artifacts ───────────────────


def test_no_record_level_files() -> None:
    suffixes = {".parquet", ".csv", ".sqlite", ".db", ".feather", ".xlsx"}
    assert not [path for path in MODEL_DIR.iterdir() if path.suffix in suffixes]


def test_no_firewall_ids_in_text_artifacts() -> None:
    firewall_ids = set(pd.read_csv(FIREWALL)["igme_id"].astype(str))
    text_suffixes = {".json", ".md", ".py", ".txt", ""}
    for path in MODEL_DIR.iterdir():
        if path.name in {"model.joblib", "preprocessor.joblib"}:
            continue
        if path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for igme_id in firewall_ids:
            assert igme_id not in text, f"{igme_id} leaked into {path.name}"


def test_no_igme_ids_in_metrics(metrics) -> None:
    serialized = json.dumps(metrics, ensure_ascii=False)
    assert "igme_id" not in serialized
    assert "coordinate_lat" not in serialized
    assert "coordinate_lon" not in serialized


# ── 9. Model card requirements ───────────────────────────────────────


def test_model_card_frontmatter(card) -> None:
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card
    assert "library_name: scikit-learn" in card
    assert "pretty_name: AquaScan National Groundwater Depth Model v1" in card


def test_model_card_documents_scope_and_firewall(card) -> None:
    assert "16,722" in card
    assert "44 of 48 provinces" in card
    assert "323 of 401 domains" in card
    assert "Phase 0 firewall" in card
    assert "10 sealed San Antonio sites" in card
    assert "historical depth to water" in card.lower()


def test_model_card_documents_limitations(card) -> None:
    lower = card.lower()
    assert "not a groundwater detector" in lower
    assert "not been validated in the field" in lower
    assert "not the current water level" in lower
    assert "marginal" in lower
    assert "not redistributed" in lower
    assert "apache-2.0" in lower
    assert "documentary/contextual features only" in lower
    assert "synthetic inputs are not field validation" in lower


def test_model_card_reports_computed_uncertainty(card, metrics) -> None:
    half_width = metrics["evaluation"]["conformal"]["interval_half_width_m"]
    assert f"{half_width:.2f} m" in card
    assert f"{100.0 * metrics['evaluation']['conformal']['empirical_coverage']:.1f}%" in card
