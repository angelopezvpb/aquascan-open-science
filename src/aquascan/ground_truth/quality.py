"""Ground Truth quality assessment.

Computes a Ground Truth Score and class (A/B/C/D) based on available
evidence. This is fundamentally different from the Candidate Priority
Score — it can only be computed AFTER real evidence (borehole logs,
pump tests, water quality data) is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aquascan.data.models import GroundTruthClass


@dataclass
class GroundTruthResult:
    """Result of Ground Truth quality assessment."""

    score: float  # 0-1 continuous score
    gt_class: GroundTruthClass
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    record_id: str = ""


def _has_field(record: dict[str, Any], field: str) -> bool:
    """Check if a field is present and non-null with valid data."""
    value = record.get(field)
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, int | float):
        import numpy as np

        if np.isnan(value) or np.isinf(value):
            return False
    return True


def score_ground_truth(
    record: dict[str, Any],
    record_id: str = "",
) -> GroundTruthResult:
    """Compute a Ground Truth quality score and class.

    Categories:
        A — Strong: coordinates, depth, lithology log, water entries,
            pump test, flow rate, identifiable documentation.
        B — Good: sufficient but incomplete (e.g., coordinates, depth,
            some water data, but missing pump test or lithology).
        C — Weak: partial or poorly verifiable (e.g., coordinates + depth
            only, or verbal report without documentation).
        D — Not suitable: too incomplete to use for primary training.

    Args:
        record: Dictionary with borehole/well data fields.
        record_id: Record identifier.

    Returns:
        GroundTruthResult with score, class, reasons, and components.
    """
    components: dict[str, float] = {}
    reasons: list[str] = []

    # --- Core evidence signals ---
    has_coords = _has_field(record, "latitude") and _has_field(record, "longitude")
    has_depth = _has_field(record, "total_depth")
    has_lithology = _has_field(record, "lithology_log")
    has_water_entries = _has_field(record, "water_entries")
    has_pump_test = _has_field(record, "pump_test_flow") and _has_field(
        record, "pump_test_duration"
    )
    has_flow_rate = _has_field(record, "pump_test_flow")
    has_static_wl = _has_field(record, "static_water_level")
    has_water_quality = _has_field(record, "water_electrical_conductivity") or _has_field(
        record, "water_tds"
    )
    has_identifiable_doc = _has_field(record, "documentation_ref")

    # --- Weighted scoring ---
    # Coordinates are foundational (0.15)
    components["coordinates"] = 0.15 if has_coords else 0.0
    if not has_coords:
        reasons.append("Missing reliable coordinates")

    # Depth (0.15)
    components["depth"] = 0.15 if has_depth else 0.0
    if has_depth:
        reasons.append("Total depth recorded")
    else:
        reasons.append("No depth data")

    # Lithology log (0.20) — strong signal
    components["lithology_log"] = 0.20 if has_lithology else 0.0
    if has_lithology:
        reasons.append("Lithology log available")

    # Water entries (0.15)
    components["water_entries"] = 0.15 if has_water_entries else 0.0
    if has_water_entries:
        reasons.append("Water entry depths recorded")

    # Pump test (0.15)
    components["pump_test"] = 0.15 if has_pump_test else 0.0
    if has_pump_test:
        reasons.append("Pump test data available (flow + duration)")
    elif has_flow_rate:
        components["pump_test"] = 0.05  # Partial credit for flow only
        reasons.append("Flow rate available (no duration)")

    # Static water level (0.10)
    components["static_water_level"] = 0.10 if has_static_wl else 0.0
    if has_static_wl:
        reasons.append("Static water level recorded")

    # Water quality (0.05)
    components["water_quality"] = 0.05 if has_water_quality else 0.0
    if has_water_quality:
        reasons.append("Water quality data available")

    # Identifiable documentation (0.05)
    components["documentation"] = 0.02 if has_identifiable_doc else 0.0
    # (Small bonus, as documentation alone doesn't prove data quality)

    total_score = sum(components.values())
    # Clamp to [0, 1]
    total_score = max(0.0, min(1.0, round(total_score, 4)))

    # --- Classification ---
    if total_score > 0.75:
        gt_class = GroundTruthClass.A
    elif total_score >= 0.50:
        gt_class = GroundTruthClass.B
    elif total_score >= 0.25:
        gt_class = GroundTruthClass.C
    else:
        gt_class = GroundTruthClass.D

    reasons.append(f"Ground Truth Class: {gt_class.value} (score={total_score:.2f})")

    return GroundTruthResult(
        score=total_score,
        gt_class=gt_class,
        reasons=reasons,
        components=components,
        record_id=record_id,
    )
