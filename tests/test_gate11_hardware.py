"""Gate 11: Hardware Integration & AQ-TEST-001 tests.

Covers:
- hardware inventory honesty (never mark absent hardware as AVAILABLE)
- protocol compatibility honesty (UNVERIFIED without hardware)
- acceptance matrix honesty (physical subsystems NOT_TESTED)
- no fabricated measurements / no groundwater claims
- software pipeline runs on SYNTHETIC fixtures only
- RAW immutability + tamper detection
- AQ-TEST-001 freeze lifecycle (NO_SCIENTIFIC_GT_AVAILABLE)
- Phase 0 cohort untouched
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

SYNTHETIC = "SYNTHETIC_ENGINEERING_FIXTURE"

INVENTORY = REPORTS_DIR / "gate11_hardware_inventory_v1.csv"
COMPAT = REPORTS_DIR / "gate11_protocol_compatibility_v1.csv"
ERT_ACC = REPORTS_DIR / "gate11_ert_acceptance_v1.json"
SEIS_ACC = REPORTS_DIR / "gate11_seismic_acceptance_v1.json"
GNSS_ACC = REPORTS_DIR / "gate11_gnss_acceptance_v1.json"
POWER_ACC = REPORTS_DIR / "gate11_power_acceptance_v1.json"
EMC_ACC = REPORTS_DIR / "gate11_emc_acceptance_v1.json"
REPEAT = REPORTS_DIR / "gate11_repeatability_v1.json"
AQ_TEST = REPORTS_DIR / "gate11_aq_test001_v1.json"
FREEZE = REPORTS_DIR / "gate11_test_prediction_freeze_v1.json"
MATRIX = REPORTS_DIR / "gate11_acceptance_matrix_v1.json"
PIPELINE = REPORTS_DIR / "gate11_data_pipeline_acceptance_v1.json"


def test_inventory_never_marks_absent_hardware_available() -> None:
    df = pd.read_csv(INVENTORY, na_filter=False)
    assert len(df) >= 15
    assert (
        df["status"] == "MISSING"
    ).all(), "Hardware must be MISSING when not physically available"
    # model/serial must not be fabricated for missing hardware
    assert (df["serial"] == "N/A").all()


def test_inventory_contains_critical_components() -> None:
    df = pd.read_csv(INVENTORY)
    components = set(df["component"])
    for critical in [
        "ERT current source",
        "Seismic acquisition",
        "GNSS RTK",
        "CAL-BOX",
        "Power",
    ]:
        assert critical in components, f"missing critical component {critical}"


def test_protocol_compatibility_unverified() -> None:
    df = pd.read_csv(COMPAT)
    assert (df["classification"] == "UNVERIFIED").all()
    assert "NOT_VERIFIED" in df["actual_capability"].iloc[0]


def test_physical_acceptance_not_tested() -> None:
    for f in [ERT_ACC, SEIS_ACC, GNSS_ACC, POWER_ACC, EMC_ACC]:
        report = json.loads(f.read_text())
        assert report["status"] == "NOT_TESTED", f
        assert report["reason"]


def test_repeatability_not_tested() -> None:
    rep = json.loads(REPEAT.read_text())
    assert rep["status"] == "NOT_TESTED"


def test_aq_test001_not_executed() -> None:
    rec = json.loads(AQ_TEST.read_text())
    assert rec["test_site"] == "AQ-TEST-001"
    assert rec["execution_status"] == "NOT_EXECUTED"
    # All SOP steps must be NOT_EXECUTED — never fabricate acquisition
    for step, status in rec["sop_steps"].items():
        assert status == "NOT_EXECUTED", f"fabricated step {step}"


def test_aq_test001_not_in_cohort() -> None:
    rec = json.loads(AQ_TEST.read_text())
    assert "NOT in Phase 0 blinded cohort" in rec["cohort_membership"]


def test_freeze_lifecycle_no_gt() -> None:
    freeze = json.loads(FREEZE.read_text())
    assert freeze["blinded_site_id"] == "AQ-TEST-001"
    assert freeze["ground_truth_status"] == "NO_SCIENTIFIC_GT_AVAILABLE"
    assert freeze["confidence_gate"] == "RED"
    assert freeze["predicted_target"] == "NONE"
    # The artifact hash must match a recomputation of the payload
    import hashlib

    payload = {k: v for k, v in freeze.items() if k != "prediction_artifact_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert recomputed == freeze["prediction_artifact_sha256"]


def test_data_pipeline_synthetic_only() -> None:
    pipeline = json.loads(PIPELINE.read_text())
    assert pipeline["warning"] == SYNTHETIC
    assert "NOT physical acceptance" in pipeline["note"]
    assert pipeline["status"] == "PASS_WITH_LIMITATIONS"
    # Stages executed by the software stack
    assert len(pipeline["ert_stages"]) == 7
    assert len(pipeline["seismic_stages"]) == 8


def test_data_pipeline_no_real_import_claim() -> None:
    pipeline = json.loads(PIPELINE.read_text())
    assert "NOT_TESTED — no real instrument file format available" in pipeline["real_data_import"]


def test_immutability_tamper_detection() -> None:
    """Copies verify; a tampered copy must fail integrity verification."""
    import hashlib

    raw_dir = DATA_DIR / "raw" / "aq_test_001_immutability_test"
    raw_file = raw_dir / "RAW_synthetic.bin"
    archive = raw_dir / "central_archive_copy.bin"
    assert raw_file.exists()
    raw_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    # The archive copy was deliberately tampered by the gate11 script
    assert archive_hash != raw_hash, "tampered copy must not match RAW"


def test_canonical_raw_not_modified_by_test() -> None:
    """The immutability test only tampers a COPY — RAW stays pristine."""
    raw_dir = DATA_DIR / "raw" / "aq_test_001_immutability_test"
    raw_file = raw_dir / "RAW_synthetic.bin"
    local = raw_dir / "local_nvme_copy.bin"
    assert raw_file.read_bytes() == local.read_bytes()


def test_acceptance_matrix_verdict() -> None:
    matrix = json.loads(MATRIX.read_text())
    assert matrix["verdict"] == "NOT READY — critical physical acceptance remains untested"
    critical = [
        "ERT_ACCEPTANCE",
        "SEISMIC_ACCEPTANCE",
        "GNSS_ACCEPTANCE",
        "AUTOCAL_ACCEPTANCE",
        "POWER_ACCEPTANCE",
        "EMC_ACCEPTANCE",
        "REPEATABILITY_ACCEPTANCE",
        "SOP_ACCEPTANCE",
    ]
    for k in critical:
        assert matrix["matrix"][k] == "NOT_TESTED", f"{k} must be NOT_TESTED"
    # Global rule documented
    assert "no critical subsystem in FAIL or NOT_TESTED" in matrix["global_rule"]


def test_acceptance_matrix_warning() -> None:
    matrix = json.loads(MATRIX.read_text())
    assert matrix["warning"] == SYNTHETIC


def test_no_groundwater_performance_claims() -> None:
    """Gate 11 outputs must never claim detection performance."""

    for f in [AQ_TEST, FREEZE, PIPELINE, MATRIX]:
        text = f.read_text()
        assert "detects groundwater" not in text.lower()
        assert "groundwater detection" not in text.lower()


def test_phase0_cohort_untouched() -> None:
    """AQ-TEST-001 must never collide with AQ-P0-001..010."""
    for f in [AQ_TEST, FREEZE]:
        text = f.read_text()
        assert "AQ-P0-001" not in text or "untouched" in text.lower()
        assert "AQ-TEST-001" in text
