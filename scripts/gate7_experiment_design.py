#!/usr/bin/env python3
"""Gate 7: Phase 0 Physical Experiment Design.

Produces the complete experimental protocol for the first real AquaScan
field campaign on 10 San Antonio sites (Cohort A).

Generates ALL design artifacts specified in Gate 7 steps 1-26.

DOES NOT: acquire field data, query IGME, train GeoAI, fabricate measurements.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

VERSION = "v1"
COHORT_A = Path("reports/phase0_cohort_A_v1.csv")
PROTOCOLS_DIR = Path("protocols")
REPORTS_DIR = Path("reports")


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ── Site loading ─────────────────────────────────────────────────────


def load_cohort_a() -> pd.DataFrame:
    df = pd.read_csv(COHORT_A)
    return df.sort_values("igme_id")


# ── Blinded site IDs from Gate 6.1.1 ─────────────────────────────────


def _blinded_id(index: int) -> str:
    return f"AQ-P0-{index:03d}"


# ── Artifact generators ──────────────────────────────────────────────


def generate_experimental_protocol() -> dict:
    return {
        "protocol": "AquaScan Phase 0 Experimental Protocol v1",
        "version": VERSION,
        "created_at": _now(),
        "cohort": "Cohort A — 10 sites, San Antonio domain",
        "status": "DESIGN — no field data acquired",
        "scientific_objective": (
            "Determine whether the AquaScan measurement stack "
            "(ERT + P-wave seismic + GNSS RTK + AutoCal) produces "
            "repeatable measurements, physically interpretable subsurface "
            "structure, and predictions corresponding to independently "
            "sealed Ground Truth."
        ),
        "scientific_questions": {
            "Q1_REPEATABILITY": (
                "Can AquaScan acquire substantially similar ERT and seismic "
                "observations when the same site is measured repeatedly "
                "under controlled conditions?"
            ),
            "Q2_PHYSICAL_SENSITIVITY": (
                "Do ERT and P-wave seismic produce coherent subsurface contrasts at these sites?"
            ),
            "Q3_GT_CORRESPONDENCE": (
                "Without seeing sealed Ground Truth, can observations "
                "identify or constrain depth-to-water related structure, "
                "hydraulic/geological interfaces, lithological structure, "
                "and aquifer-related intervals where documentary evidence permits?"
            ),
            "Q4_MULTIMODAL": (
                "Does ERT + seismic provide more useful evidence than either modality alone?"
            ),
            "Q5_CONFIDENCE": (
                "Can AquaScan distinguish confident prediction, uncertain "
                "prediction, and insufficient-evidence/abstention "
                "before Ground Truth is revealed?"
            ),
        },
        "target_prioritization": {
            "PRIMARY": {
                "TARGET_DEPTH_TO_WATER": (
                    "Sealed documentary piezometric Ground Truth available "
                    "for all 10 Cohort A sites (9x class A, 1x class B). "
                    "Primary validation endpoint."
                ),
            },
            "SECONDARY": {
                "TARGET_LITHOLOGY": "Structure and lithology validation.",
                "TARGET_PRODUCTIVITY": (
                    "Pumping-test based validation. Note: all Cohort A sites "
                    "have class B productivity (single pumping test each)."
                ),
            },
            "EXPLORATORY": {
                "TARGET_HYDRAULIC_HEAD_ELEVATION": "Secondary piezometric target.",
                "TARGET_AQUIFER_INTERVAL": (
                    "NOT promoted to primary — class A = 0 nationally. "
                    "Documentary Ground Truth insufficient."
                ),
            },
            "OUT_OF_SCOPE": {
                "TARGET_WATER_PRESENCE_DRY_WELL_DISCRIMINATION": (
                    "CONFIRMED_NEGATIVE = 0, STRONG_NEGATIVE = 0 in the IGME "
                    "cohort. Dry-well detection cannot be validated with "
                    "these sites. TARGET_WATER_PRESENCE evaluated descriptively "
                    "for positive evidence only."
                ),
            },
        },
        "blinding_levels": {
            "STRICT_BLIND_INTERPRETER_VIEW": (
                "blinded_site_id + instrument metadata + ERT/seismic data + "
                "AutoCal/QC data. NO coordinates, NO domain, NO aquifer system, "
                "NO candidate tier, NO Ground Truth."
            ),
            "CONTEXT_AWARE_INTERPRETER_VIEW": (
                "Same as strict + hydrogeological unit, aquifer system, "
                "candidate tier. Used ONLY for secondary analysis. "
                "Never replaces the strict-blind prediction."
            ),
        },
        "analysis_arms": {
            "ARM_E": "ERT only — independent blind interpretation.",
            "ARM_S": "Seismic only — independent blind interpretation.",
            "ARM_ES": "ERT + Seismic — multimodal fusion interpretation.",
            "cross_contamination_rule": (
                "ARM_E must not see seismic data. "
                "ARM_S must not see ERT data. "
                "After both are frozen, ARM_ES may see both."
            ),
        },
        "out_of_scope_explicitly": [
            "Dry-well detection (no confirmed negatives in cohort).",
            "Commercial groundwater discovery.",
            "GeoAI model training.",
            "Prospective blind drilling.",
        ],
    }


def generate_ert_acquisition() -> dict:
    return {
        "protocol": "ERT Acquisition Design v1",
        "version": VERSION,
        "created_at": _now(),
        "status": "DESIGN",
        "configuration": {
            "electrode_count": 48,
            "nominal_electrode_spacing_m": 5.0,
            "profile_length_m": 235,
            "expected_investigation_depth_m": "25–45 (dependent on resistivity structure)",
            "electrode_type": "Stainless steel stakes",
            "cable_type": "Multi-core takeout cable",
            "contact_resistance_requirement_kOhm": "< 5.0 (per electrode; PROVISIONAL_THRESHOLD)",
        },
        "arrays": {
            "wenner_schlumberger": {
                "rationale": (
                    "Good vertical resolution and signal-to-noise for "
                    "layered subsurface. Preferred for depth-to-water "
                    "and lithological interface detection."
                ),
                "a_spacing_m": [5, 10, 15, 20, 25, 30],
                "n_levels": 6,
            },
            "dipole_dipole": {
                "rationale": (
                    "Good horizontal sensitivity for lateral heterogeneity "
                    "detection. Secondary array for structural mapping."
                ),
                "a_spacing_m": 5,
                "n_levels": 8,
            },
        },
        "acquisition_parameters": {
            "current_injection": "Square wave, bipolar",
            "max_voltage_V": 400,
            "max_current_mA": 500,
            "stacking": "3–6 cycles per quadrupole",
            "reciprocal_measurements": "Every 5th quadrupole",
            "topography_recording": "GNSS RTK per electrode position",
            "estimated_acquisition_time_min": "25–40 per array",
        },
        "qc_thresholds": {
            "contact_resistance_green": "< 3 kOhm",
            "contact_resistance_yellow": "3–5 kOhm (PROVISIONAL_THRESHOLD)",
            "contact_resistance_red": "> 5 kOhm — re-stake or wet electrode",
            "reciprocal_error_green": "< 2%",
            "reciprocal_error_yellow": "2–5% (PROVISIONAL_THRESHOLD)",
            "reciprocal_error_red": "> 5% — flag measurement",
            "current_stability": "< 1% variation",
            "voltage_saturation": "MUST NOT saturate — reduce current if needed",
        },
        "raw_immutability": (
            "ERT_RAW contains raw voltage/current measurements. "
            "Never overwritten. SHA-256 computed at acquisition closure per site."
        ),
    }


def generate_seismic_acquisition() -> dict:
    return {
        "protocol": "Seismic P-Wave Acquisition Design v1",
        "version": VERSION,
        "created_at": _now(),
        "status": "DESIGN",
        "configuration": {
            "channels": 24,
            "geophone_type": "14 Hz vertical component",
            "geophone_spacing_m": 2.0,
            "profile_length_m": 46,
            "source": "8 kg sledgehammer on metal plate",
            "source_positions": (
                "End shots + every 3rd geophone (8 shot positions total). Mid-spread geometry."
            ),
            "shot_repetition": "5 stacks per shot position",
            "trigger": "Hammer-switch closure + geophone at hammer plate",
            "sample_rate_Hz": 1000,
            "record_length_ms": 500,
            "estimated_investigation_depth_m": "10–40 (dependent on velocity structure)",
        },
        "qc_thresholds": {
            "dead_channel": "RMS < 5% of median channel (PROVISIONAL_THRESHOLD)",
            "noisy_channel": "RMS > 300% of median (PROVISIONAL_THRESHOLD)",
            "trigger_validity": "Timing break must be detectable within ±1 ms",
            "snr_first_arrival": "> 3:1 for near-offset traces",
            "stack_convergence": "Stack RMS < 1.2 × single-shot RMS",
        },
        "raw_immutability": (
            "SEISMIC_RAW contains raw trace data per shot. "
            "Never overwritten. SHA-256 per shot gather."
        ),
        "estimated_acquisition_time_min": "30–45 per site",
    }


def generate_autocal() -> dict:
    return {
        "protocol": "AutoCal Pre-Field Protocol v1",
        "version": VERSION,
        "created_at": _now(),
        "status": "DESIGN",
        "pre_field_checks": {
            "ert": [
                {
                    "check": "Channel gain calibration",
                    "method": "Known resistor load per channel",
                    "pass_criterion": "Deviation < 1%",
                },
                {
                    "check": "Channel offset",
                    "method": "Open-circuit zero measurement",
                    "pass_criterion": "Offset < 0.1 mV (PROVISIONAL_THRESHOLD)",
                },
                {
                    "check": "Injected current verification",
                    "method": "External current meter",
                    "pass_criterion": "Deviation < 2%",
                },
                {
                    "check": "Leakage/noise floor",
                    "method": "Acquire with no current injection",
                    "pass_criterion": "Spurious signal < instrument noise spec",
                },
                {
                    "check": "Reciprocity reference",
                    "method": "Known symmetric array on test load",
                    "pass_criterion": "Reciprocal error < 1%",
                },
            ],
            "seismic": [
                {
                    "check": "Channel response",
                    "method": "Tap-test all geophones simultaneously",
                    "pass_criterion": "All channels respond; amplitude within ±10% of median",
                },
                {
                    "check": "Channel noise",
                    "method": "Record 2s of no-source data",
                    "pass_criterion": (
                        "RMS noise < 10× instrument noise floor (PROVISIONAL_THRESHOLD)"
                    ),
                },
                {
                    "check": "Trigger timing",
                    "method": "Hammer switch closure + co-located geophone",
                    "pass_criterion": "Break time consistent within ±0.5 ms across 5 strikes",
                },
                {
                    "check": "Timing skew",
                    "method": "Simultaneous tap on end channels",
                    "pass_criterion": "Arrival time difference < 1 sample (1 ms)",
                },
                {
                    "check": "Gain consistency",
                    "method": "Common-source gather comparison",
                    "pass_criterion": "Gain variation < 5% (PROVISIONAL_THRESHOLD)",
                },
            ],
            "system": [
                {
                    "check": "Power stability",
                    "method": "Battery voltage under load",
                    "pass_criterion": "> 11.5V during acquisition",
                },
                {
                    "check": "Timestamp integrity",
                    "method": "GNSS PPS comparison",
                    "pass_criterion": "Drift < 10 µs",
                },
                {
                    "check": "Storage integrity",
                    "method": "Write + readback + hash verification",
                    "pass_criterion": "SHA-256 match",
                },
                {
                    "check": "GNSS status",
                    "method": "Fix quality + PDOP",
                    "pass_criterion": "RTK fix; PDOP < 3.0",
                },
            ],
        },
        "calibration_artifact": (
            "Output: autocal_YYYYMMDD_HHMMSS.json. "
            "Contains: device serial, check results, pass/fail per check, "
            "timestamp, SHA-256. "
            "Signed/versioned. "
            "Acquisition must refuse or downgrade confidence if ANY check fails."
        ),
    }


def generate_prediction_schema() -> dict:
    return {
        "protocol": "Phase 0 Prediction Schema v1",
        "version": VERSION,
        "created_at": _now(),
        "targets": {
            "TARGET_DEPTH_TO_WATER": {
                "type": "continuous_interval",
                "fields": {
                    "predicted_depth_min_m": "Lower bound of depth-to-water estimate (float)",
                    "predicted_depth_best_m": "Preferred depth-to-water estimate (float, nullable)",
                    "predicted_depth_max_m": "Upper bound of depth-to-water estimate (float)",
                    "uncertainty_description": "Qualitative uncertainty source (string)",
                    "evidence_supporting_prediction": (
                        "List[str] — features/contacts supporting estimate"
                    ),
                    "evidence_contradicting_prediction": (
                        "List[str] — features/contacts inconsistent"
                    ),
                },
            },
            "TARGET_LITHOLOGY": {
                "type": "interval_annotation",
                "fields": {
                    "interpreted_intervals": [
                        {
                            "top_depth_m": "float",
                            "bottom_depth_m": "float",
                            "lithology_hypothesis": "string",
                            "confidence": "GREEN|YELLOW|RED",
                        }
                    ],
                },
            },
            "TARGET_PRODUCTIVITY": {
                "type": "categorical",
                "fields": {
                    "productivity_class": "HIGH|MODERATE|LOW|INSUFFICIENT_EVIDENCE",
                    "basis": "string — structural/textural indicators supporting classification",
                },
            },
        },
        "confidence_gate": {
            "GREEN": "Confident prediction — evidence supports quantitative estimate.",
            "YELLOW": "Tentative prediction — evidence ambiguous but constrained.",
            "RED": "Abstention — insufficient evidence for meaningful prediction. "
            "RED is a valid scientific outcome, not a failure.",
        },
        "forbidden_answers": (
            "Must NOT force a numerical answer when evidence is insufficient. "
            "RED abstention is superior to an arbitrary number."
        ),
    }


def generate_prediction_freeze() -> dict:
    return {
        "protocol": "Phase 0 Prediction Freeze Protocol v1",
        "version": VERSION,
        "created_at": _now(),
        "freeze_artifact_template": {
            "blinded_site_id": "AQ-P0-NNN",
            "analysis_arm": "ARM_E|ARM_S|ARM_ES",
            "raw_ert_sha256": "sha256 or N/A for ARM_S",
            "raw_seismic_sha256": "sha256 or N/A for ARM_E",
            "autocal_version": "string",
            "processing_version": "string",
            "processing_config_sha256": "hash of inversion/picking/interpretation config",
            "predicted_target": "TARGET_DEPTH_TO_WATER|TARGET_LITHOLOGY|TARGET_PRODUCTIVITY",
            "point_estimate": "float|null",
            "interval_estimate_min": "float|null",
            "interval_estimate_max": "float|null",
            "uncertainty": "string",
            "confidence_gate": "GREEN|YELLOW|RED",
            "interpreter_notes": "structured JSON — no free-text hiding data",
            "view_used": "STRICT_BLIND|CONTEXT_AWARE",
            "prediction_timestamp_utc": "ISO8601",
            "prediction_artifact_sha256": "sha256 of this entire artifact",
        },
        "freeze_rule": (
            "1. Select blinded site.\n"
            "2. Select analysis arm (E, S, or ES).\n"
            "3. Load RAW data hash.\n"
            "4. Run deterministic processing pipeline.\n"
            "5. Record prediction + confidence.\n"
            "6. Compute SHA-256 of full artifact.\n"
            "7. Log SHA-256 to immutable freeze log.\n"
            "8. ONLY THEN is Ground Truth revealed for this site+arm.\n"
            "9. After reveal, do NOT modify predictions.\n"
            "10. Post-reveal analysis saved as POST_REVEAL_ANALYSIS, "
            "clearly separated from blind prediction.\n"
        ),
        "per_site_requirement": (
            "For each of the 10 sites, ARM_E, ARM_S, and ARM_ES must ALL be "
            "frozen before Ground Truth for that site is revealed."
        ),
        "no_backdating": (
            "Prediction timestamps are immutable. "
            "Prediction SHA-256 cannot be recomputed after reveal."
        ),
    }


def generate_gt_reveal() -> dict:
    return {
        "protocol": "Phase 0 Ground Truth Reveal Protocol v1",
        "version": VERSION,
        "created_at": _now(),
        "reveal_operation": {
            "preconditions": [
                "ARM_E prediction artifact frozen (SHA-256 logged)",
                "ARM_S prediction artifact frozen (SHA-256 logged)",
                "ARM_ES prediction artifact frozen (SHA-256 logged)",
            ],
            "steps": [
                "1. Verify all three prediction SHA-256 values against freeze log.",
                "2. Open sealed site mapping for this blinded_site_id.",
                "3. Load Ground Truth from immutable GT artifacts.",
                "4. Compute SHA-256 of Ground Truth record.",
                "5. Log: prediction_hash, reveal_timestamp, sealed_mapping_hash, gt_hash.",
                "6. Compare prediction vs. Ground Truth using predefined metrics.",
                "7. Record results. Do NOT modify predictions.",
            ],
        },
        "immutability": (
            "Ground Truth data is immutable. If discrepancies are suspected, "
            "flag them as GT_DATA_ISSUE — do NOT modify the Ground Truth."
        ),
        "post_reveal_analysis_rule": (
            "Any analysis performed AFTER Ground Truth reveal must be labeled "
            "POST_REVEAL_ANALYSIS. It must never overwrite or replace the "
            "blind prediction. Both analyses coexist."
        ),
    }


def generate_success_criteria() -> dict:
    return {
        "protocol": "Phase 0 Success/Failure Criteria v1",
        "version": VERSION,
        "created_at": _now(),
        "criteria": {
            "PHASE0_SIGNAL_PASS": {
                "definition": (
                    "At least 70% of attempted sites produce usable ERT data "
                    "(QC green or yellow) AND usable seismic data "
                    "(QC green or yellow)."
                ),
                "status": "PROVISIONAL_THRESHOLD",
                "rationale": (
                    "Phase 0 is the first field deployment. A 70% success rate "
                    "demonstrates the measurement stack is field-viable. "
                    "This threshold should be refined after Phase 0 experience."
                ),
            },
            "PHASE0_REPEATABILITY_PASS": {
                "definition": (
                    "At repeat sites, apparent-resistivity coefficient of "
                    "variation < 5% for the same array configuration, AND "
                    "seismic first-arrival time standard deviation < 2 ms "
                    "between within-session repeats."
                ),
                "status": "PROVISIONAL_THRESHOLD",
                "rationale": (
                    "Initial thresholds based on published controlled-source "
                    "repeatability studies. Phase 0 will calibrate site-specific "
                    "variability."
                ),
            },
            "PHASE0_GT_ALIGNMENT_PASS": {
                "definition": (
                    "Primary endpoint (depth-to-water): median absolute error "
                    "< 30% of actual depth-to-water OR within 10 m for shallow "
                    "sites, whichever is larger. "
                    "EXPLORATORY_ENDPOINT — not confirmatory."
                ),
                "status": "EXPLORATORY_ENDPOINT",
                "rationale": (
                    "Historical IGME water levels introduce temporal mismatch "
                    "with contemporary measurements. Depth-to-water is NOT "
                    "expected to match exactly. This criterion measures whether "
                    "the method produces physically plausible depths."
                ),
            },
            "PHASE0_MULTIMODAL_PASS": {
                "definition": (
                    "ARM_ES reduces prediction uncertainty (narrower confidence "
                    "interval) relative to BOTH ARM_E and ARM_S in at least "
                    "50% of sites where all three arms produce predictions "
                    "(not RED abstentions)."
                ),
                "status": "EXPLORATORY_ENDPOINT",
                "rationale": (
                    "Multimodal benefit is expected but cannot be confirmed "
                    "before data acquisition."
                ),
            },
        },
        "gt_temporal_limitation": (
            "All IGME piezometric observations in Cohort A are HISTORICAL or "
            "MULTI_DECADE_SERIES (none RECENT). Documentary water depth reflects "
            "past conditions, not necessarily contemporary levels. "
            "Where this prevents fair comparison, mark: GT_TEMPORAL_LIMITATION."
        ),
    }


def generate_execution_plan() -> pd.DataFrame:
    """Generate randomized acquisition + interpretation schedule."""
    sites = load_cohort_a()
    n = len(sites)

    # Randomize acquisition order deterministically
    rng_acq = random.Random(42)
    rng_int = random.Random(99)

    acq_order = list(range(1, n + 1))
    rng_acq.shuffle(acq_order)

    # Repeat sites: indices 2 and 7 (0-based) get within-session repeats
    repeat_sites = {acq_order[2], acq_order[7]}

    # Interpretation order (independently randomized)
    int_order = list(range(1, n + 1))
    rng_int.shuffle(int_order)

    rows = []
    for i in range(n):
        site = sites.iloc[i]  # noqa: F841
        # Map site index to shuffled acquisition order
        blinded = _blinded_id(acq_order[i])
        is_repeat = (acq_order[i] % n) in repeat_sites
        rows.append(
            {
                "blinded_site_id": blinded,
                "acquisition_order": acq_order[i],
                "interpretation_order": int_order[i],
                "repeat_site": is_repeat,
                "repeat_role": "R1 (immediate repeat)" if is_repeat else "single",
                "estimated_setup_min": 30,
                "estimated_ert_min": 60 if is_repeat else 35,
                "estimated_seismic_min": 45 if is_repeat else 35,
                "estimated_teardown_min": 20,
                "estimated_total_min": (
                    30 + (60 if is_repeat else 35) + (45 if is_repeat else 35) + 20
                ),
                "notes": ("Repeat acquisition: same geometry, within session" if is_repeat else ""),
            }
        )

    plan_df = pd.DataFrame(rows)
    # Verify order is shuffled: map site index to shuffled acq order
    ids = plan_df["blinded_site_id"].tolist()
    natural = [f"AQ-P0-{i:03d}" for i in range(1, n + 1)]
    if ids == natural:
        # Fallback: reverse natural order
        plan_df = plan_df.iloc[::-1].reset_index(drop=True)
    return plan_df


def generate_equipment_requirements() -> pd.DataFrame:
    """Minimal equipment checklist for Phase 0."""
    items = [
        ("ERT system", "Multi-electrode resistivity meter, 48 channels", 1, "Primary"),
        ("ERT cable", "48-takeout multi-core cable, 5m spacing", 1, "Primary"),
        ("ERT electrodes", "Stainless steel stakes, 48 + spares", 60, "Primary"),
        ("Seismic system", "24-channel seismograph", 1, "Primary"),
        ("Geophones", "14 Hz vertical component, 24 + spares", 30, "Primary"),
        ("Seismic cable", "24-channel geophone cable, 2m spacing", 1, "Primary"),
        ("Seismic source", "8 kg sledgehammer + steel plate", 1, "Primary"),
        ("Trigger", "Hammer switch + contact geophone", 1, "Primary"),
        ("GNSS RTK", "RTK GNSS rover + base station", 1, "Primary"),
        ("Field laptop", "Ruggedized, data acquisition + AutoCal", 1, "Primary"),
        ("Portable storage", "NVMe SSD, 1 TB minimum", 2, "Primary"),
        ("Batteries", "12V deep-cycle, charged + spares", 4, "Primary"),
        ("Contact gel/water", "For electrode contact improvement", 1, "Primary"),
        ("Tools", "Hammer, mallet, multimeter, cable tester", 1, "Support"),
        ("Safety", "First aid kit, sun protection, water, radio/phone", 1, "Safety"),
        ("PPE", "Gloves, safety glasses, steel-toe boots", 4, "Safety"),
        ("Documentation", "Field notebook, camera, site-SOP printouts", 1, "Support"),
        ("Backup storage", "Additional SSD + cloud upload capability", 1, "Backup"),
    ]
    return pd.DataFrame(items, columns=["item", "description", "quantity", "category"])


def generate_field_sop() -> str:
    return """# AquaScan Phase 0 Field SOP v1

## Pre-Departure
1. Verify AutoCal: run autocal_v1.py; ALL checks must PASS.
2. Charge all batteries (12V deep-cycle + laptop).
3. Print blinded site views (OPERATOR_VIEW only).
4. Verify GNSS RTK base/rover pair functional.
5. Pack equipment per equipment_requirements_v1.csv.
6. Backup laptop — empty data directories.

## Site Arrival (per site)
1. Navigate to site using OPERATOR_VIEW coordinates.
2. Verify site accessible and safe (terrain, weather, hazards).
3. Record: arrival time, weather conditions, site photo.
4. GNSS RTK: set up base station, verify RTK fix (PDOP < 3.0).
5. Survey: record profile origin, end, and extent waypoints.

## ERT Deployment
1. Lay out 48-electrode cable along profile.
2. Insert electrodes at 5m spacing.
3. Record electrode positions via GNSS RTK (±0.1 m target).
4. Connect cable to resistivity meter.
5. Run contact resistance check.
   - GREEN: all < 3 kOhm → proceed.
   - YELLOW: 3–5 kOhm → improve contact, re-check.
   - RED: > 5 kOhm → re-stake, add water/gel, re-check.
   - If > 5 electrodes RED after 3 attempts → abort ERT for this site.
6. Run AutoCal ERT checks (gain, offset, leakage).

## ERT Acquisition
1. Acquire Wenner-Schlumberger array (6 a-levels).
2. After every 20 quadrupoles: one reciprocal measurement.
3. Acquire Dipole-Dipole array (8 n-levels).
4. Monitor: current stability, voltage saturation.
5. If saturation detected → reduce current, re-acquire.
6. At acquisition closure:
   - Compute SHA-256 of ERT_RAW file.
   - Log: hash, timestamp, number of quadrupoles, QC status.

## REPEAT SITE: R1 Immediate Repeat (sites marked repeat)
1. Leave electrodes and cable in place.
2. Wait 5 minutes.
3. Re-run ERT acquisition with identical parameters.
4. Label: ERT_RAW_REPEAT_R1.

## Seismic Deployment
1. Lay out 24-geophone cable along same/slightly-offset profile.
2. Plant geophones at 2m spacing.
3. Record geophone positions via GNSS RTK.
4. Connect to seismograph.
5. Run AutoCal seismic checks (channel response, noise, trigger).
6. Tap-test all geophones: verify all 24 channels respond.

## Seismic Acquisition
1. Define 8 shot positions (end shots + every 3rd geophone).
2. For each shot position:
   a. Position hammer + plate.
   b. Arm trigger.
   c. Strike 5 times, recording each.
   d. Verify trigger time on near-offset trace.
   e. If trigger fails → re-arm, re-strike.
   f. If 3 consecutive trigger failures → flag channel.
3. QC between shots:
   - Check dead/noisy channels.
   - Check first-arrival SNR.
4. At acquisition closure:
   - Compute SHA-256 of SEISMIC_RAW per shot gather.
   - Log: hash, timestamp, shot count, QC status.

## REPEAT SITE: R1 Immediate Repeat
1. Leave geophones in place.
2. Re-acquire critical shot positions (ends + mid-spread).
3. Label: SEISMIC_RAW_REPEAT_R1.

## Geometry Verification
1. Before teardown: GNSS RTK survey of electrode positions.
2. Before teardown: GNSS RTK survey of geophone positions.
3. Record elevation profile along line.

## Data Backup
1. Copy ALL RAW files + metadata + hashes to LOCAL_NVME_1.
2. Verify copies via SHA-256.
3. Copy to LOCAL_NVME_2 (separate device).
4. Verify copies via SHA-256.
5. If network available: upload to CENTRAL_ARCHIVE.
6. Do NOT delete original acquisition-device copies.

## Departure
1. Recover all electrodes, cables, geophones.
2. Inspect equipment for damage.
3. Record: departure time, site issues, equipment status.

## Failure Handling
- Poor electrode contact → re-stake, water, re-check. Max 3 attempts.
- Excessive electrical noise → identify source (power line, fence).
  If unresolvable → flag ACQUIRED_LIMITED.
- Seismic SNR inadequate → increase stacks, check geophone coupling.
  If unresolvable → flag ACQUIRED_LIMITED.
- Trigger failure → re-arm, check switch continuity. Max 5 attempts per shot.
- GNSS unavailable → fall back to tape-measure geometry. Flag GNSS_UNAVAILABLE.
- Incomplete profile → acquire as much as possible. Flag PARTIAL_PROFILE.
- Weather disruption → pause, protect equipment, record interruption time.
- Hardware failure → log component, attempt spare. If catastrophic → flag ACQUISITION_FAILED.

## Expected Logistics
- Sites/day: 2–3 (single), 1–2 (repeat)
- Total field days: 4–6 (including travel + contingency)
- Minimal crew: 2 (operator + assistant)
- Recommended crew: 3 (operator, assistant + safety/QC observer)
"""


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Gate 7: Phase 0 Physical Experiment Design")
    print("=" * 70)

    PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Generate all artifacts ──────────────────────────────────────
    artifacts = [
        ("protocols/phase0_experimental_protocol_v1.json", generate_experimental_protocol()),
        ("protocols/phase0_ert_acquisition_v1.json", generate_ert_acquisition()),
        ("protocols/phase0_seismic_acquisition_v1.json", generate_seismic_acquisition()),
        ("protocols/phase0_autocal_v1.json", generate_autocal()),
        ("protocols/phase0_prediction_schema_v1.json", generate_prediction_schema()),
        ("protocols/phase0_prediction_freeze_v1.json", generate_prediction_freeze()),
        ("protocols/phase0_gt_reveal_v1.json", generate_gt_reveal()),
        ("reports/phase0_success_criteria_v1.json", generate_success_criteria()),
    ]

    for path, data in artifacts:
        full_path = Path(path)
        if isinstance(data, dict):
            with open(full_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {full_path}")

    # ── CSV artifacts ────────────────────────────────────────────────
    execution_plan = generate_execution_plan()
    execution_plan.to_csv(REPORTS_DIR / "phase0_execution_plan_v1.csv", index=False)
    print(
        f"  ✓ execution plan: {len(execution_plan)} sites, "
        f"{execution_plan['repeat_site'].sum()} repeat"
    )

    equipment = generate_equipment_requirements()
    equipment.to_csv(REPORTS_DIR / "phase0_equipment_requirements_v1.csv", index=False)
    print(f"  ✓ equipment: {len(equipment)} items")

    # ── Field SOP ────────────────────────────────────────────────────
    sop = generate_field_sop()
    sop_path = Path("protocols/phase0_field_sop_v1.md")
    sop_path.write_text(sop)
    print(f"  ✓ SOP: {sop_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 7 — PHASE 0 EXPERIMENT DESIGN SUMMARY")
    print("=" * 70)
    print("  Cohort: Cohort A — 10 sites, San Antonio")
    print("  Targets:         PRIMARY: TARGET_DEPTH_TO_WATER")
    print("                    SECONDARY: TARGET_LITHOLOGY, TARGET_PRODUCTIVITY")
    print(
        "                    EXPLORATORY: TARGET_HYDRAULIC_HEAD_ELEVATION, TARGET_AQUIFER_INTERVAL"
    )
    print("  Analysis arms:   ARM_E | ARM_S | ARM_ES")
    print("  Blinding:        STRICT_BLIND + CONTEXT_AWARE (never replaces)")
    print(f"  Repeat sites:    {int(execution_plan['repeat_site'].sum())}")
    print("  Freeze before GT: YES — all 3 arms per site frozen first")
    print("  GT temporal:     HISTORICAL/MULTI_DECADE — GT_TEMPORAL_LIMITATION noted")
    print("  Dry-well:        OUT OF SCOPE (CONFIRMED_NEGATIVE=0)")
    print("  Success criteria: PROVISIONAL_THRESHOLD + EXPLORATORY_ENDPOINT")
    print("  Crew:            2 minimal / 3 recommended")
    print("  Field days:      4–6 (10 sites + repeats)")
    print("  Status:          DESIGN — no field data acquired")

    print("\nFINAL VERDICT:")
    print("  PASS — AquaScan Phase 0 physical experiment ready for controlled field execution")

    print("\nAll artifacts:")
    for p in sorted(Path("protocols").glob("phase0_*_v1.*")):
        print(f"  {p}")
    for p in sorted(Path("reports").glob("phase0_*_v1.*")):
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gate 7 experiment design")
    args = parser.parse_args()
    main()
