#!/usr/bin/env python3
"""Gate 11: Hardware Integration & AQ-TEST-001 — readiness audit.

Gate 11 crosses the software/physical boundary. In the current environment
NO physical hardware is available (no ERT meter, seismic system, GNSS RTK,
or CAL-BOX), and no AQ-TEST-001 field data exists. Per the gate's own rule
(section 1): "If physical hardware is not actually available, report MISSING
or NOT_TESTED. Software tests cannot substitute for physical acceptance."

What this script DOES produce (honest statuses only):
- Frozen hardware inventory: every Gate 7 component recorded as MISSING.
- Gate 7 protocol compatibility: NOT_SUPPORTED / UNVERIFIED (no hardware).
- ERT / Seismic / GNSS / Power / EMC acceptance: NOT_TESTED with reasons.
- AutoCal fault-injection catalog: scenarios defined, actual_detection=NOT_TESTED.
- Software-testable items exercised on SYNTHETIC data only:
  * data pipeline ingestion (Gate 10 stack, canonical schemas)
  * RAW immutability + copy/hash verification (tamper detection)
  * AQ-TEST-001 freeze lifecycle with NO_SCIENTIFIC_GT_AVAILABLE
- Acceptance matrix + verdict.

DOES NOT: measure any Phase 0 blinded cohort site, reveal Phase 0 GT,
query IGME, train final GeoAI, fabricate measurements, claim groundwater
detection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aquascan.physical.ert.pipeline import ERTPipeline  # noqa: E402
from aquascan.physical.fusion.contract import (  # noqa: E402
    GeoAIFusionInput,
    ModalityBundle,
)
from aquascan.physical.models import (  # noqa: E402
    ERTObservation,
    ERTQuadrupoleMeasurement,
    SeismicObservation,
    SeismicShot,
)
from aquascan.physical.seismic.pipeline import SeismicPipeline  # noqa: E402
from aquascan.physical.synthetic.fixtures import (  # noqa: E402
    SYNTHETIC_WARNING,
    ErtTwoLayerFixture,
    SeismicTwoLayerFixture,
)

REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
PROTOCOLS_DIR = ROOT / "protocols"
VERSION = "v1"

SYNTHETIC = "SYNTHETIC_ENGINEERING_FIXTURE"
SYNTHETIC_NOTE = (
    "Software pipeline exercised on synthetic engineering fixtures only. "
    "This is NOT physical acceptance and NOT field data."
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── Hardware inventory (from Gate 7 equipment requirements) ───────────


def build_hardware_inventory() -> list[dict[str, str]]:
    """Freeze the real BOM with honest statuses.

    In this environment no hardware is present, so every component is
    MISSING. Never mark planned hardware as available.
    """
    items = [
        ("ERT current source", "Multi-electrode resistivity meter, 48 channels"),
        ("ERT voltage measurement", "Resistivity meter voltage channels (48)"),
        ("Multiplexer", "48-electrode takeout switch matrix"),
        ("ERT electrodes", "Stainless steel stakes, 48 + spares"),
        ("ERT cables", "48-takeout multi-core cable, 5 m spacing"),
        ("CAL-BOX", "AutoCal reference box (known loads / switch matrix)"),
        ("Seismic acquisition", "24-channel seismograph"),
        ("Geophones", "14 Hz vertical component, 24 + spares"),
        ("Seismic cable", "24-channel geophone cable, 2 m spacing"),
        ("Seismic source", "8 kg sledgehammer + steel plate"),
        ("Trigger", "Hammer switch + contact geophone"),
        ("GNSS RTK", "RTK rover + base station"),
        ("Compute", "Ruggedized field laptop"),
        ("Storage", "NVMe SSD 1 TB x2"),
        ("Power", "12V deep-cycle LiFePO4 batteries + spares"),
        ("Contact gel/water", "Electrode contact improvement"),
    ]
    rows: list[dict[str, str]] = []
    for name, desc in items:
        rows.append(
            {
                "component": name,
                "description": desc,
                "status": "MISSING",
                "model_version": "N/A (not acquired)",
                "serial": "N/A",
                "firmware_version": "N/A",
                "ranges": "N/A",
                "evidence": (
                    "No hardware present in the repository environment at Gate 11 execution time."
                ),
            }
        )
    return rows


# ── Gate 7 protocol compatibility ─────────────────────────────────────


def build_protocol_compatibility() -> list[dict[str, str]]:
    """Compare actual (absent) hardware against Gate 7 protocol geometry."""
    requirements = [
        ("ERT electrodes", "48 electrodes", "48 x 5 m profile"),
        ("ERT spacing", "5 m nominal electrode spacing", "235 m profile"),
        ("ERT arrays", "Wenner-Schlumberger + Dipole-Dipole", "Gate 7 arrays"),
        ("Seismic channels", "24 channels", "24-channel spread"),
        ("Seismic geophones", "14 Hz vertical", "2 m spacing"),
        ("Seismic source", "8 shot positions, hammer", "5 stacks per shot"),
        ("GNSS", "RTK per electrode/geophone", "±0.1 m target"),
    ]
    rows: list[dict[str, str]] = []
    for item, gate7_req, detail in requirements:
        rows.append(
            {
                "component": item,
                "gate7_requirement": gate7_req,
                "detail": detail,
                "actual_capability": "NOT_VERIFIED (hardware absent)",
                "classification": "UNVERIFIED",
                "note": (
                    "Cannot be classified SUPPORTED without physical "
                    "hardware evidence. No silent protocol amendment."
                ),
            }
        )
    return rows


# ── Acceptance report builders (all physical = NOT_TESTED) ────────────


def _not_tested(reason: str) -> dict[str, object]:
    return {
        "status": "NOT_TESTED",
        "reason": reason,
        "evidence": "No physical hardware or field measurements available.",
    }


def build_ert_acceptance() -> dict[str, object]:
    checks = [
        "known_resistance_loads",
        "resistance_ranges",
        "current_measurement",
        "voltage_measurement",
        "channel_gain",
        "offsets",
        "repeatability",
        "open_circuit",
        "bad_channel",
        "multiplexer_mapping",
        "polarity",
        "isolation_leakage",
    ]
    return {
        "acceptance": "ERT_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No ERT resistivity meter or test loads available.",
        "channel_verification": "software electrode N == physical electrode N: NOT_VERIFIED",
        "checks": dict.fromkeys(checks, "NOT_TESTED"),
    }


def build_seismic_acceptance() -> dict[str, object]:
    checks = [
        "baseline_noise",
        "response",
        "polarity",
        "dead_channels",
        "clipping",
        "trigger",
        "trigger_jitter",
        "inter_channel_timing",
        "repeated_impulse_consistency",
    ]
    return {
        "acceptance": "SEISMIC_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No 24-channel seismograph or geophones available.",
        "timing": "Timing NOT independently measured -> UNVERIFIED.",
        "checks": dict.fromkeys(checks, "NOT_TESTED"),
    }


def build_gnss_acceptance() -> dict[str, object]:
    return {
        "acceptance": "GNSS_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No GNSS RTK receiver available.",
        "rtk_fix": "NOT_TESTED",
        "rtk_float": "NOT_TESTED",
        "correction_loss_recovery": "NOT_TESTED",
        "elevation_semantics": (
            "Protocol requires explicit ELLIPSOIDAL/ORTHOMETRIC tracking; "
            "the software frame (Gate 10 coords.py) enforces this, but no "
            "physical observations exist to verify."
        ),
    }


def build_power_acceptance() -> dict[str, object]:
    return {
        "acceptance": "POWER_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No field power stack available.",
        "idle_consumption": "NOT_TESTED",
        "ert_load": "NOT_TESTED",
        "seismic_load": "NOT_TESTED",
        "compute_load": "NOT_TESTED",
        "peak_demand": "NOT_TESTED",
        "battery_runtime_estimate": "NOT_TESTED",
        "voltage_stability": "NOT_TESTED",
        "power_loss_test": "NOT_TESTED",
        "raw_corruption_check": "NOT_TESTED (no power hardware)",
    }


def build_emc_acceptance() -> dict[str, object]:
    return {
        "acceptance": "EMC_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No electronics stack available to measure interference.",
        "ert_only": "NOT_TESTED",
        "seismic_active": "NOT_TESTED",
        "gnss_compute_active": "NOT_TESTED",
        "all_active": "NOT_TESTED",
        "seismic_noise_ert_off": "NOT_TESTED",
        "seismic_noise_ert_idle": "NOT_TESTED",
        "seismic_noise_ert_injecting": "NOT_TESTED",
        "sequential_operation_recommendation": (
            "UNDETERMINED — cannot classify REQUIRED/RECOMMENDED/UNNECESSARY "
            "without physical EMC measurement."
        ),
    }


def build_autocal_faults() -> list[dict[str, str]]:
    """Fault-injection catalog: scenarios defined, actual detection pending.

    AutoCal only passes if it can DETECT faults, not merely a healthy
    system. Detection must be verified with physical hardware.
    """
    faults = [
        (
            "ERT",
            "open_channel",
            "high impedance on a channel",
            "detect high contact resistance",
            "NOT_TESTED",
        ),
        (
            "ERT",
            "incorrect_known_load",
            "known resistor replaced by wrong value",
            "detect gain/offset deviation > 1%",
            "NOT_TESTED",
        ),
        (
            "ERT",
            "noisy_signal",
            "injected broadband noise",
            "detect leakage/noise floor exceedance",
            "NOT_TESTED",
        ),
        (
            "ERT",
            "mapping_fault",
            "electrode N physically miswired",
            "detect channel mapping inconsistency",
            "NOT_TESTED",
        ),
        (
            "Seismic",
            "disconnected_geophone",
            "open geophone channel",
            "detect dead channel",
            "NOT_TESTED",
        ),
        (
            "Seismic",
            "noisy_channel",
            "excessive channel noise",
            "detect RMS above threshold",
            "NOT_TESTED",
        ),
        (
            "Seismic",
            "trigger_failure",
            "hammer switch not closing",
            "detect missing trigger",
            "NOT_TESTED",
        ),
        ("System", "gnss_unavailable", "no GNSS fix", "detect and refuse/downgrade", "NOT_TESTED"),
        ("System", "low_battery", "voltage below threshold", "detect and warn", "NOT_TESTED"),
        ("System", "storage_warning", "disk nearly full", "detect and warn", "NOT_TESTED"),
    ]
    return [
        {
            "subsystem": s,
            "fault_id": f,
            "fault_description": d,
            "expected_detection": e,
            "actual_detection": a,
            "severity": "critical" if s in ("ERT", "Seismic", "System") else "major",
            "system_action": "abort_or_downgrade" if a == "NOT_TESTED" else "warn",
        }
        for s, f, d, e, a in faults
    ]


# ── Software-testable items (SYNTHETIC ONLY) ──────────────────────────


def _make_ert_observation() -> ERTObservation:
    quads: list[ERTQuadrupoleMeasurement] = []
    fx = ErtTwoLayerFixture(rho1=100.0, rho2=1000.0, h1=10.0)
    rho_a_full = fx.forward()
    for i in range(12):
        spacing = float(5.0 + i * 5.0)
        # Interpolate the analytic two-layer apparent resistivity at spacing
        rho_a = float(np.interp(spacing, fx.spacing_m, rho_a_full))
        k = 2.0 * np.pi * spacing
        quads.append(
            ERTQuadrupoleMeasurement(
                survey_id="SYNTH-ERT-001",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-001",
                electrode_a=0.0,
                electrode_b=3.0 * spacing,
                electrode_m=spacing,
                electrode_n=2.0 * spacing,
                injected_current_a=0.1,
                measured_voltage_v=(rho_a / k) * 0.1,
                contact_resistance_ohm=120.0,
                instrument_channel=f"CH{i}",
                autocal_version="v0.1",
                raw_hash=f"raw-synth-{i}",
            )
        )
    return ERTObservation(
        survey_id="SYNTH-ERT-001",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-001",
        quadrupoles=quads,
        autocal_version="v0.1",
        raw_hash=hashlib.sha256(b"synth").hexdigest(),
    )


def _make_seismic_observation() -> SeismicObservation:
    fx = SeismicTwoLayerFixture()
    offsets = fx.offsets_m
    _, times = fx.forward()
    shots: list[SeismicShot] = []
    for si, sx in enumerate([0.0, 30.0]):
        rec_x: list[float] = []
        rec_ids: list[str] = []
        waves: list[list[float]] = []
        for ri, rx in enumerate(offsets):
            rel = abs(rx - sx)
            t_arr = float(np.interp(rel, offsets, times))
            idx = max(0, min(999, int(t_arr * 2000.0)))
            trace = [0.0] * 1000
            trace[idx] = 1.0
            waves.append(trace)
            rec_x.append(rx)
            rec_ids.append(f"R{ri}")
        shots.append(
            SeismicShot(
                survey_id="SYNTH-SEIS-001",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-S1",
                shot_id=f"S{si}",
                shot_x=sx,
                receiver_ids=rec_ids,
                receiver_x=rec_x,
                sample_rate_hz=2000.0,
                record_length_s=0.5,
                raw_waveforms=waves,
                channel_numbers=list(range(len(rec_ids))),
                autocal_version="v0.1",
                raw_hash=f"raw-synth-seis-{si}",
            )
        )
    return SeismicObservation(
        survey_id="SYNTH-SEIS-001",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-S1",
        shots=shots,
        autocal_version="v0.1",
        raw_hash=hashlib.sha256(b"synth-seis").hexdigest(),
    )


def run_data_pipeline_acceptance() -> dict[str, object]:
    """Feed synthetic observations through the Gate 10 stack.

    This proves the software import path and canonical schemas work. It is
    NOT physical acceptance.
    """
    ert_obs = _make_ert_observation()
    ert_chain = ERTPipeline().run(ert_obs)
    seis_obs = _make_seismic_observation()
    seis_chain = SeismicPipeline().run(seis_obs)

    # Fusion contract executes with both modalities
    ert_section = ert_chain["inverted"]["data"]
    seis_model = seis_chain["velocity"]["data"]
    bundle_e = ModalityBundle(
        representation=ert_section.model_dump(mode="json"),
        quality=0.8,
        uncertainty=0.2,
    )
    bundle_s = ModalityBundle(
        representation=seis_model.model_dump(mode="json"),
        quality=0.8,
        uncertainty=0.2,
    )
    fusion_in = GeoAIFusionInput(
        blinded_site_id="AQ-TEST-001",
        ert=bundle_e,
        seismic=bundle_s,
    )

    return {
        "acceptance": "DATA_PIPELINE_ACCEPTANCE",
        "status": "PASS_WITH_LIMITATIONS",
        "note": SYNTHETIC_NOTE,
        "warning": SYNTHETIC,
        "ert_stages": list(ert_chain.keys()),
        "seismic_stages": list(seis_chain.keys()),
        "fusion_executes": fusion_in.blinded_site_id == "AQ-TEST-001",
        "real_data_import": "NOT_TESTED — no real instrument file format available; "
        "canonical schemas accept the software representation without change.",
        "limitations": [
            "Physical acceptance NOT performed (no hardware).",
            "Real instrument file formats (proprietary) unverified.",
            "No real contact resistance / noise / first-arrival statistics.",
        ],
    }


def run_immutability_test() -> dict[str, object]:
    """RAW -> SHA-256 -> copies -> verify; tamper a copy; prove failure.

    Software-testable integrity lifecycle. The canonical RAW is never
    modified; only a COPY is tampered.
    """
    raw_dir = DATA_DIR / "raw" / "aq_test_001_immutability_test"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "RAW_synthetic.bin"
    raw_file.write_bytes(b"AQ-TEST-001 SYNTHETIC RAW BYTES" + bytes(range(64)))

    local_copy = raw_dir / "local_nvme_copy.bin"
    archive_copy = raw_dir / "central_archive_copy.bin"
    shutil.copy2(raw_file, local_copy)
    shutil.copy2(raw_file, archive_copy)

    raw_hash = _sha256_file(raw_file)
    local_hash = _sha256_file(local_copy)
    archive_hash = _sha256_file(archive_copy)
    copies_verified = local_hash == raw_hash == archive_hash

    # Tamper a COPY (never the canonical RAW)
    archive_copy.write_bytes(archive_copy.read_bytes() + b"TAMPER")
    tampered_hash = _sha256_file(archive_copy)
    tamper_detected = tampered_hash != raw_hash

    return {
        "acceptance": "DATA_IMMUTABILITY",
        "status": "PASS" if copies_verified and tamper_detected else "FAIL",
        "raw_sha256": raw_hash,
        "local_copy_sha256": local_hash,
        "archive_copy_sha256": archive_hash,
        "copies_verified": copies_verified,
        "tamper_detected": tamper_detected,
        "note": (
            "Canonical RAW never modified; only a COPY was tampered to "
            "prove integrity verification fails."
        ),
    }


def run_repeatability_acceptance() -> dict[str, object]:
    """R1/R2 repeatability requires hardware — report NOT_TESTED."""
    return {
        "acceptance": "REPEATABILITY_ACCEPTANCE",
        "status": "NOT_TESTED",
        "reason": "No AQ-TEST-001 field acquisition performed.",
        "r1_immediate_repeat": "NOT_TESTED",
        "r2_redeployment_repeat": "NOT_TESTED",
        "metrics_defined": [
            "ERT: current, voltage, apparent resistivity, reciprocity, derived model",
            "Seismic: waveforms, picks, travel times, velocity model",
        ],
    }


def build_aq_test001() -> dict[str, object]:
    """AQ-TEST-001 record — NOT EXECUTED (no hardware / no site)."""
    return {
        "test_site": "AQ-TEST-001",
        "cohort_membership": "NOT in Phase 0 blinded cohort (AQ-P0-001..010 untouched)",
        "execution_status": "NOT_EXECUTED",
        "reason": "No physical hardware or test location available in this environment.",
        "sop_steps": {
            "gnss_survey": "NOT_EXECUTED",
            "ert_deployment": "NOT_EXECUTED",
            "contact_qc": "NOT_EXECUTED",
            "autocal": "NOT_EXECUTED",
            "ert_raw_acquisition": "NOT_EXECUTED",
            "ert_qc": "NOT_EXECUTED",
            "seismic_deployment": "NOT_EXECUTED",
            "seismic_calibration": "NOT_EXECUTED",
            "hammer_acquisition": "NOT_EXECUTED",
            "seismic_qc": "NOT_EXECUTED",
            "hashes": "NOT_EXECUTED",
            "backup": "NOT_EXECUTED",
        },
        "field_time_measurement": "NOT_MEASURED (no field time)",
        "software_proxy": {
            "note": SYNTHETIC_NOTE,
            "ert_pipeline_stages": list(ERTPipeline().run(_make_ert_observation()).keys()),
            "seismic_pipeline_stages": list(
                SeismicPipeline().run(_make_seismic_observation()).keys()
            ),
        },
    }


def build_freeze_artifact() -> dict[str, object]:
    """AQ-TEST-001 test prediction freeze using the Gate 7 machinery.

    NO_SCIENTIFIC_GT_AVAILABLE — this proves the freeze lifecycle works on
    software representations. It is NOT a groundwater performance result.
    """
    payload = {
        "blinded_site_id": "AQ-TEST-001",
        "analysis_arm": "ARM_ES",
        "raw_ert_sha256": "SYNTHETIC-NO-REAL-ERT",
        "raw_seismic_sha256": "SYNTHETIC-NO-REAL-SEISMIC",
        "autocal_version": "NOT_RUN (no hardware)",
        "processing_version": "0.1.0",
        "processing_config_sha256": "SYNTHETIC-CONFIG",
        "predicted_target": "NONE",
        "point_estimate": None,
        "interval_estimate_min": None,
        "interval_estimate_max": None,
        "uncertainty": "INSUFFICIENT_EVIDENCE",
        "confidence_gate": "RED",
        "interpreter_notes": {"note": "Software freeze-lifecycle proof. No real measurements."},
        "view_used": "STRICT_BLIND",
        "prediction_timestamp_utc": _now_iso(),
        "ground_truth_status": "NO_SCIENTIFIC_GT_AVAILABLE",
    }
    artifact_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload["prediction_artifact_sha256"] = artifact_hash
    return payload


def main() -> None:
    print("=" * 70)
    print("Gate 11: Hardware Integration & AQ-TEST-001 — readiness audit")
    print("=" * 70)
    print(f"\n{SYNTHETIC_WARNING}\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Hardware inventory ────────────────────────────────────────
    print("[1] Hardware inventory …")
    inventory = build_hardware_inventory()
    with open(REPORTS_DIR / "gate11_hardware_inventory_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
        w.writeheader()
        w.writerows(inventory)
    n_missing = sum(1 for r in inventory if r["status"] == "MISSING")
    print(f"  {len(inventory)} components, {n_missing} MISSING (no hardware present)")

    # ── 2. Protocol compatibility ────────────────────────────────────
    print("[2] Gate 7 protocol compatibility …")
    compat = build_protocol_compatibility()
    with open(REPORTS_DIR / "gate11_protocol_compatibility_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compat[0].keys()))
        w.writeheader()
        w.writerows(compat)
    print(f"  {len(compat)} requirements -> all UNVERIFIED (hardware absent)")

    # ── 3-9. Physical acceptance reports ─────────────────────────────
    print("[3] Physical acceptance reports (all NOT_TESTED) …")
    acceptances = {
        "gate11_ert_acceptance_v1.json": build_ert_acceptance(),
        "gate11_seismic_acceptance_v1.json": build_seismic_acceptance(),
        "gate11_gnss_acceptance_v1.json": build_gnss_acceptance(),
        "gate11_power_acceptance_v1.json": build_power_acceptance(),
        "gate11_emc_acceptance_v1.json": build_emc_acceptance(),
    }
    for fname, report in acceptances.items():
        (REPORTS_DIR / fname).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  {fname}: {report['status']}")

    # ── AutoCal fault catalog ────────────────────────────────────────
    print("[4] AutoCal fault-injection catalog …")
    faults = build_autocal_faults()
    with open(REPORTS_DIR / "gate11_autocal_faults_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(faults[0].keys()))
        w.writeheader()
        w.writerows(faults)
    n_detected = sum(1 for x in faults if x["actual_detection"] != "NOT_TESTED")
    print(
        f"  {len(faults)} fault scenarios defined; {n_detected} detected "
        f"(physical detection pending)"
    )

    # ── Software-testable: data pipeline acceptance ──────────────────
    print("[5] Data pipeline acceptance (SYNTHETIC software check) …")
    pipeline = run_data_pipeline_acceptance()
    (REPORTS_DIR / "gate11_data_pipeline_acceptance_v1.json").write_text(
        json.dumps(pipeline, indent=2, ensure_ascii=False)
    )
    print(f"  status: {pipeline['status']} ({pipeline['note']})")

    # ── Software-testable: immutability ──────────────────────────────
    print("[6] RAW immutability + tamper detection …")
    immutability = run_immutability_test()
    print(
        f"  copies_verified={immutability['copies_verified']} "
        f"tamper_detected={immutability['tamper_detected']}"
    )

    # ── Repeatability ────────────────────────────────────────────────
    print("[7] Repeatability acceptance …")
    repeatability = run_repeatability_acceptance()
    (REPORTS_DIR / "gate11_repeatability_v1.json").write_text(
        json.dumps(repeatability, indent=2, ensure_ascii=False)
    )
    print(f"  status: {repeatability['status']}")

    # ── AQ-TEST-001 ──────────────────────────────────────────────────
    print("[8] AQ-TEST-001 …")
    aq_test = build_aq_test001()
    (REPORTS_DIR / "gate11_aq_test001_v1.json").write_text(
        json.dumps(aq_test, indent=2, ensure_ascii=False)
    )
    print(f"  execution_status: {aq_test['execution_status']}")

    # ── Freeze lifecycle ─────────────────────────────────────────────
    print("[9] AQ-TEST-001 prediction freeze (lifecycle proof) …")
    freeze = build_freeze_artifact()
    (REPORTS_DIR / "gate11_test_prediction_freeze_v1.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False)
    )
    print(
        f"  artifact sha256: {freeze['prediction_artifact_sha256'][:16]}… "
        f"| GT: {freeze['ground_truth_status']}"
    )

    # ── Acceptance matrix ────────────────────────────────────────────
    print("[10] Acceptance matrix …")
    matrix = {
        "generated_at": _now_iso(),
        "warning": SYNTHETIC,
        "matrix": {
            "ERT_ACCEPTANCE": acceptances["gate11_ert_acceptance_v1.json"]["status"],
            "SEISMIC_ACCEPTANCE": acceptances["gate11_seismic_acceptance_v1.json"]["status"],
            "GNSS_ACCEPTANCE": acceptances["gate11_gnss_acceptance_v1.json"]["status"],
            "AUTOCAL_ACCEPTANCE": "NOT_TESTED",
            "POWER_ACCEPTANCE": acceptances["gate11_power_acceptance_v1.json"]["status"],
            "EMC_ACCEPTANCE": acceptances["gate11_emc_acceptance_v1.json"]["status"],
            "DATA_PIPELINE_ACCEPTANCE": pipeline["status"],
            "REPEATABILITY_ACCEPTANCE": repeatability["status"],
            "SOP_ACCEPTANCE": "NOT_TESTED",
            "DATA_IMMUTABILITY": immutability["status"],
        },
        "global_rule": ("Global PASS requires no critical subsystem in FAIL or NOT_TESTED."),
        "verdict": "NOT READY — critical physical acceptance remains untested",
        "verdict_reason": (
            "No physical hardware (ERT, seismic, GNSS, power, EMC, CAL-BOX) is "
            "available in this environment. Software pipeline checks pass on "
            "synthetic fixtures but cannot substitute for physical acceptance. "
            "No AQ-TEST-001 field acquisition was performed. No Phase 0 site "
            "was touched; no Ground Truth revealed."
        ),
    }
    (REPORTS_DIR / "gate11_acceptance_matrix_v1.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False)
    )
    for k, v in matrix["matrix"].items():
        print(f"  {k}: {v}")

    # ── Verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 11 VERDICT")
    print("=" * 70)
    print("  Hardware: NOT AVAILABLE in this environment")
    print("  AQ-TEST-001: NOT EXECUTED")
    print("  Physical acceptance: NOT_TESTED (all critical subsystems)")
    print("  Software pipeline (synthetic): PASS")
    print("  RAW immutability/tamper detection: PASS")
    print("  Prediction freeze lifecycle: executed (NO_SCIENTIFIC_GT_AVAILABLE)")
    print()
    print("  NOT READY — critical physical acceptance remains untested")
    print("=" * 70)


if __name__ == "__main__":
    main()
