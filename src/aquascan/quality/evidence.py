"""Evidence Score computation.

The Evidence Score is separate from the Water Score.
It measures the quality of evidence backing a prediction.

A prediction can have:
    Water Score    = 0.90
    Evidence Score = 0.50
    → flagged as NO CONCLUYENTE

This module computes the evidence score from available data quality signals.
"""

from __future__ import annotations

from dataclasses import dataclass

from aquascan.data.models import EvidenceScore


@dataclass
class EvidenceScoreResult:
    """Result of evidence score computation."""

    score: float
    components: dict[str, float]
    reasons: list[str]
    is_conclusive: bool


def compute_evidence_score(
    ert_quality: float | None = None,
    seismic_quality: float | None = None,
    gnss_quality: float | None = None,
    has_hydrogeological_context: bool = False,
    has_geological_context: bool = False,
    distance_to_nearest_well_m: float | None = None,
    num_nearby_wells: int = 0,
    conclusive_threshold: float = 0.4,
) -> EvidenceScore:
    """Compute the evidence quality score.

    Args:
        ert_quality: ERT data quality score (0-1).
        seismic_quality: Seismic data quality score (0-1).
        gnss_quality: GNSS positioning quality (0-1).
        has_hydrogeological_context: Whether hydrogeological context is available.
        has_geological_context: Whether geological context is available.
        distance_to_nearest_well_m: Distance to nearest known well (meters).
        num_nearby_wells: Number of nearby known wells.
        conclusive_threshold: Threshold above which evidence is considered conclusive.

    Returns:
        An EvidenceScore instance.
    """
    components: dict[str, float] = {}
    reasons: list[str] = []

    # --- Geophysical data quality (dominant) ---
    geophys_scores = []
    if ert_quality is not None:
        components["ert_quality"] = ert_quality
        geophys_scores.append(ert_quality)
        if ert_quality < 0.5:
            reasons.append("ERT quality is low")
        else:
            reasons.append(f"ERT quality: {ert_quality:.2f}")
    else:
        components["ert_quality"] = 0.0
        reasons.append("No ERT data available")

    if seismic_quality is not None:
        components["seismic_quality"] = seismic_quality
        geophys_scores.append(seismic_quality)
        if seismic_quality < 0.5:
            reasons.append("Seismic quality is low")
        else:
            reasons.append(f"Seismic quality: {seismic_quality:.2f}")
    else:
        components["seismic_quality"] = 0.0

    if gnss_quality is not None:
        components["gnss_quality"] = gnss_quality
        if gnss_quality < 0.5:
            reasons.append("GNSS positioning quality is low")

    # Geophysical quality component (0-0.6 weight)
    if geophys_scores:
        geophys_mean = sum(geophys_scores) / len(geophys_scores)
        components["geophysical_overall"] = round(0.6 * geophys_mean, 4)
    else:
        components["geophysical_overall"] = 0.0
        reasons.append("No geophysical data at all")

    # --- Context components ---
    ctx_score = 0.0
    if has_hydrogeological_context:
        ctx_score += 0.15
        components["hydrogeological_context"] = 0.15
        reasons.append("Hydrogeological context available")
    else:
        components["hydrogeological_context"] = 0.0

    if has_geological_context:
        ctx_score += 0.15
        components["geological_context"] = 0.15
        reasons.append("Geological context available")
    else:
        components["geological_context"] = 0.0

    # Proximity to known wells (0-0.10)
    well_prox = 0.0
    if num_nearby_wells > 0 and distance_to_nearest_well_m is not None:
        if distance_to_nearest_well_m < 100:
            well_prox = 0.10
            reasons.append(f"Very close to known well ({distance_to_nearest_well_m:.0f}m)")
        elif distance_to_nearest_well_m < 500:
            well_prox = 0.07
            reasons.append(f"Near known well ({distance_to_nearest_well_m:.0f}m)")
        elif distance_to_nearest_well_m < 2000:
            well_prox = 0.03
            reasons.append(f"Moderately near known well ({distance_to_nearest_well_m:.0f}m)")
        else:
            reasons.append(f"Far from known wells ({distance_to_nearest_well_m:.0f}m)")
    elif num_nearby_wells == 0:
        reasons.append("No nearby wells for reference")
    components["well_proximity"] = well_prox

    total = sum(components.values())
    total = round(max(0.0, min(1.0, total)), 4)

    return EvidenceScore(
        score=total,
        components=components,
        reasons=reasons,
        # Note: is_conclusive flag is NOT stored on the EvidenceScore model
        # It's computed context for callers
    )


def is_evidence_conclusive(
    evidence: EvidenceScore,
    threshold: float = 0.4,
) -> bool:
    """Determine if evidence is sufficient for a conclusive prediction.

    Args:
        evidence: The evidence score.
        threshold: Minimum evidence score to be considered conclusive.

    Returns:
        True if evidence is above the threshold.
    """
    return evidence.score >= threshold
