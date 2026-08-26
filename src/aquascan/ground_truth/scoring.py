"""Candidate Priority Score.

This score is used ONLY for prioritizing which public records are worth
investigating further. It is NOT the ground truth quality score.

It uses public metadata signals:
- Has depth?
- Has coordinates?
- Has flow rate?
- Has hydrogeological unit?
- Has detailed record?
- Record recency?
- Point type / nature?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class CandidatePriorityResult:
    """Result of computing a Candidate Priority Score."""

    score: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    record_id: str = ""


def compute_candidate_priority_score(
    record: dict[str, Any],
    record_id: str = "",
) -> CandidatePriorityResult:
    """Compute a Candidate Priority Score for a public record.

    This score helps prioritize which records to investigate. It is purely
    based on metadata completeness and public signals — it does NOT measure
    scientific ground truth quality.

    Args:
        record: A dictionary of public record fields.
        record_id: Record identifier for traceability.

    Returns:
        A CandidatePriorityResult with score, components, and reasons.
    """
    components: dict[str, float] = {}
    reasons: list[str] = []

    # --- Depth present (0.2) ---
    has_depth = (
        record.get("total_depth") is not None and float(record.get("total_depth", 0) or 0) > 0
    )
    components["has_depth"] = 0.2 if has_depth else 0.0
    if has_depth:
        reasons.append("Record includes depth information")
    else:
        reasons.append("No depth information")

    # --- Coordinates present (0.2) ---
    has_coords = record.get("latitude") is not None and record.get("longitude") is not None
    components["has_coordinates"] = 0.2 if has_coords else 0.0
    if has_coords:
        reasons.append("Record includes coordinates")
    else:
        reasons.append("No coordinates")

    # --- Flow rate present (0.15) ---
    has_flow = (
        record.get("pump_test_flow") is not None and float(record.get("pump_test_flow", 0) or 0) > 0
    )
    components["has_flow_rate"] = 0.15 if has_flow else 0.0
    if has_flow:
        reasons.append("Record includes flow rate data")

    # --- Hydrogeological unit present (0.15) ---
    has_hgu = (
        record.get("hydrogeological_unit") is not None
        and str(record.get("hydrogeological_unit", "")).strip() != ""
    )
    components["has_hydrogeological_unit"] = 0.15 if has_hgu else 0.0
    if has_hgu:
        reasons.append("Hydrogeological unit identified")

    # --- Detailed record (0.15) ---
    # A "detailed" record has extended metadata beyond the basics.
    # For IGME: method, pump type, usage, organism, or URL detail.
    has_metodo = (
        record.get("source_metadata", {}).get("metodo_perforacion") is not None
        if isinstance(record.get("source_metadata"), dict)
        else False
    )
    has_motobomba = (
        record.get("source_metadata", {}).get("motobomba") is not None
        if isinstance(record.get("source_metadata"), dict)
        else False
    )
    has_usos = (
        record.get("source_metadata", {}).get("usos_agua") is not None
        if isinstance(record.get("source_metadata"), dict)
        else False
    )
    components["has_detailed_record"] = 0.15 if (has_metodo or has_motobomba or has_usos) else 0.0
    if has_metodo or has_motobomba or has_usos:
        reasons.append("Record has detailed metadata (method/pump/usage)")

    # --- Record recency (0.10) ---
    # Prefer newer records (last 20 years)
    date_str = record.get("timestamp") or record.get("fecha")
    try:
        if date_str:
            record_date = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            age_years = (datetime.now(UTC) - record_date).days / 365.25
            if age_years < 5:
                components["recency"] = 0.10
                reasons.append("Recent record (< 5 years)")
            elif age_years < 20:
                components["recency"] = 0.05
                reasons.append("Moderate recency (< 20 years)")
            else:
                components["recency"] = 0.0
                reasons.append(f"Old record ({age_years:.0f} years)")
        else:
            components["recency"] = 0.0
            reasons.append("No date information")
    except (ValueError, TypeError):
        components["recency"] = 0.0
        reasons.append("Could not parse date")

    # --- Point type (0.05) ---
    point_type = str(
        record.get("source_metadata", {}).get("naturaleza", "")
        if isinstance(record.get("source_metadata"), dict)
        else ""
    ).lower()
    preferred_types = ["sondeo", "pozo", "piezometro", "piezómetro", "well", "borehole"]
    if any(pt in point_type for pt in preferred_types):
        components["point_type_bonus"] = 0.05
        reasons.append(f"Preferred point type: {point_type}")
    else:
        components["point_type_bonus"] = 0.0

    score = sum(components.values())
    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return CandidatePriorityResult(
        score=round(score, 4),
        components=components,
        reasons=reasons,
        record_id=record_id,
    )
