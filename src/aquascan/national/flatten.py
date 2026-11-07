"""Flattening and lightweight QA helpers for enriched national records.

Semantics are the validated Gate 4/5 semantics (NormalizedGroundTruthRecord).
This module only shapes output for Parquet partitions and runs lightweight
incremental QA (Gate 29): invalid dates, coordinate anomalies, depth
outliers, negative flows, malformed payloads, semantic relationship checks.
QA never silently fixes anything — it records violations.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from aquascan.data.models import NormalizedGroundTruthRecord

# Endpoints that map to evidence families in the normalized record.
ENDPOINT_FAMILY = {
    "lithology": "geological_evidence",
    "piezometry": "water_level_evidence",
    "pumping_tests": "hydraulic_evidence",
    "chemistry": "water_quality_evidence",
    "hydrometry": "hydrometry_evidence",
}

NO_PREFIX = {
    "lithology": "no_lithology",
    "piezometry": "no_piezometry",
    "pumping_tests": "no_pumping_test",
    "chemistry": "no_chemistry",
    "hydrometry": "no_hydrometry",
    "technical": "no_technical",
    "documents": "no_documents",
}


def flatten_normalized_record(
    rec: NormalizedGroundTruthRecord,
    record_status: str,
    endpoint_statuses: dict[str, dict[str, Any]],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a NormalizedGroundTruthRecord into a Parquet-safe dict.

    Mirrors the validated Gate 5 flattening and adds:
    - record_status (COMPLETE / PARTIAL / FAILED / ...)
    - explicit NO_* markers for missing evidence (partial-record policy)
    - endpoint_status_json (per-endpoint status + response hash, provenance)
    - a curated set of source snapshot metadata columns.
    """
    d = rec.model_dump()
    d["geological_evidence_count"] = len(rec.geological_evidence)
    d["water_level_evidence_count"] = len(rec.water_level_evidence)
    d["hydraulic_evidence_count"] = len(rec.hydraulic_evidence)
    d["water_quality_evidence_count"] = len(rec.water_quality_evidence)
    d["hydrometry_evidence_count"] = len(rec.hydrometry_evidence)
    d["water_presence_status"] = rec.water_presence.status.value
    d["water_presence_basis"] = ";".join(rec.water_presence.evidence_basis)
    d["geological_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.geological_evidence], default=str
    )
    d["water_level_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.water_level_evidence], default=str
    )
    d["hydraulic_evidence_json"] = json.dumps(
        [e.model_dump() for e in rec.hydraulic_evidence], default=str
    )
    d["water_quality_evidence_json"] = json.dumps(rec.water_quality_evidence, default=str)
    d["hydrometry_evidence_json"] = json.dumps(rec.hydrometry_evidence, default=str)
    d["provenance_json"] = json.dumps(rec.provenance, default=str)
    d["temporal_summary_json"] = json.dumps(rec.temporal_summary.model_dump(), default=str)
    d["lithology_validation_json"] = json.dumps(rec.lithology_validation.model_dump(), default=str)
    for target, ass in rec.target_readiness.items():
        d[f"readiness_{target}"] = ass.readiness.value
        d[f"readiness_{target}_quality"] = ass.evidence_quality
        d[f"readiness_{target}_confidence"] = ass.documentary_confidence

    # Endpoint coverage / partial-record policy: a missing endpoint never
    # removes the record — missing information is represented explicitly as
    # NO_<ENDPOINT> markers (Gate 26).
    for ep, no_col in NO_PREFIX.items():
        s = endpoint_statuses.get(ep) or {}
        count = int(s.get("count", 0) or 0)
        d[no_col] = count == 0

    d["record_status"] = record_status
    d["endpoint_status_json"] = json.dumps(
        {
            ep: {
                "status": s.get("status"),
                "response_hash": s.get("response_hash", ""),
                "count": int(s.get("count", 0) or 0),
            }
            for ep, s in endpoint_statuses.items()
        },
        sort_keys=True,
        default=str,
    )

    # Source snapshot metadata (curated subset)
    if source:
        for col in (
            "latitude",
            "longitude",
            "elevation_m",
            "watershed",
            "hydrogeological_unit",
            "aquifer_system",
            "province",
            "municipality",
            "point_nature",
            "total_depth_m",
            "reference_flow_l_s",
            "drilling_method",
            "construction_date",
            "pump_type",
            "water_use",
            "organization",
            "qa_flag_count",
        ):
            d[f"source_{col}"] = source.get(col)

    # Remove nested structures that break parquet
    for key in (
        "geological_evidence",
        "water_level_evidence",
        "hydraulic_evidence",
        "water_quality_evidence",
        "hydrometry_evidence",
        "target_readiness",
        "temporal_summary",
        "lithology_validation",
        "water_presence",
        "provenance",
    ):
        d.pop(key, None)
    return d


def _is_valid_iso_date(value: str | None) -> bool:
    if value is None or value == "":
        return True
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def run_qa_checks(
    rec: NormalizedGroundTruthRecord,
    source: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Lightweight incremental QA (Gate 29).

    Returns a list of {"check_name", "severity", "message"}.
    WARNING severity → recorded, continue. CRITICAL → quarantine.
    QA never silently fixes anything.
    """
    violations: list[dict[str, str]] = []

    # 1. Invalid observation dates (piezometry / pumping tests / chemistry / hydrometry)
    for obs in rec.water_level_evidence:
        if not _is_valid_iso_date(obs.observation_date):
            violations.append(
                {
                    "check_name": "invalid_date",
                    "severity": "WARNING",
                    "message": f"unparseable water-level date: {obs.observation_date!r}",
                }
            )
    for test in rec.hydraulic_evidence:
        if not _is_valid_iso_date(test.test_date):
            violations.append(
                {
                    "check_name": "invalid_date",
                    "severity": "WARNING",
                    "message": f"unparseable pumping-test date: {test.test_date!r}",
                }
            )

    # 2. Coordinate anomaly (outside broad Spain bbox)
    if source is not None:
        lat = source.get("latitude")
        lon = source.get("longitude")
        if lat is not None and lon is not None:
            if lon < -18.5 or lon > 5.0 or lat < 27.5 or lat > 44.0:
                violations.append(
                    {
                        "check_name": "coordinate_anomaly",
                        "severity": "WARNING",
                        "message": f"coordinates outside Spain bbox: ({lat}, {lon})",
                    }
                )

    # 3. Depth outlier: lithology bottom beyond total borehole depth
    if source is not None:
        total_depth = source.get("total_depth_m")
        if total_depth is not None:
            try:
                td = float(total_depth)
                for interval in rec.geological_evidence:
                    if interval.bottom_depth_m is not None and interval.bottom_depth_m > td + 1.0:
                        violations.append(
                            {
                                "check_name": "depth_outlier",
                                "severity": "WARNING",
                                "message": (
                                    f"lithology bottom {interval.bottom_depth_m} m exceeds "
                                    f"total depth {td} m"
                                ),
                            }
                        )
            except (TypeError, ValueError):
                pass

    # 4. Negative flow where invalid (pumping tests)
    for test in rec.hydraulic_evidence:
        if test.flow_l_s is not None and test.flow_l_s < 0:
            violations.append(
                {
                    "check_name": "negative_flow",
                    "severity": "WARNING",
                    "message": f"negative pumping-test flow: {test.flow_l_s}",
                }
            )

    # 5. Semantic relationship: Acuifero flag alone is NOT a water entry.
    #    The validated normalize module already guarantees this; verify it.
    for interval in rec.geological_evidence:
        if interval.aquifer_flagged_lithology_interval:
            # Aquifer-flagged interval is allowed; it must never be the sole
            # basis of water_presence. Check the evidence basis.
            if rec.water_presence.evidence_basis == [] and rec.water_presence.status.value in (
                "PRESENT",
                "ABSENT",
            ):
                violations.append(
                    {
                        "check_name": "semantic_relationship_failure",
                        "severity": "CRITICAL",
                        "message": "water presence derived without any hydrogeological evidence",
                    }
                )
            break

    # 6. Semantic relationship: reference flow is never a pumping test.
    if source is not None and source.get("reference_flow_l_s") is not None:
        try:
            ref_flow = float(source["reference_flow_l_s"])
        except (TypeError, ValueError):
            ref_flow = None
        if ref_flow is not None and rec.water_presence.evidence_basis:
            # reference_flow_l_s must not appear in the evidence basis
            if any("reference_flow" in b for b in rec.water_presence.evidence_basis):
                violations.append(
                    {
                        "check_name": "semantic_relationship_failure",
                        "severity": "CRITICAL",
                        "message": "reference flow leaked into water-presence evidence",
                    }
                )

    # 7. Malformed payload: evidence rows with no usable fields
    for obs in rec.water_level_evidence:
        if obs.depth_to_water_m is None and obs.hydraulic_head_elevation_m is None:
            violations.append(
                {
                    "check_name": "malformed_payload",
                    "severity": "WARNING",
                    "message": "piezometry row with neither Nivel nor Valor",
                }
            )
            break

    return violations
