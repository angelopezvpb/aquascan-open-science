"""ERT quality assessment (deterministic, public domain methods only).

Assesses ERT data quality based on measurable acquisition parameters:
- Contact resistance statistics
- Reciprocal error
- Stacking error
- Electrode geometry completeness

This module uses only deterministic, public-domain methods.
It does NOT implement proprietary AutoCal algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ERTQualityResult:
    """Result of ERT quality assessment."""

    quality_score: float  # 0-1
    contact_resistance_ok: bool
    reciprocity_error_pct: float | None = None
    stacking_error_pct: float | None = None
    electrode_count: int = 0
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def assess_ert_quality(
    contact_resistances: list[float] | None = None,
    reciprocity_errors: list[float] | None = None,
    stacking_errors: list[float] | None = None,
    electrode_count: int = 0,
    max_acceptable_contact_resistance: float = 2000.0,  # ohms
    max_acceptable_reciprocity_pct: float = 5.0,  # percent
    max_acceptable_stacking_pct: float = 3.0,  # percent
) -> ERTQualityResult:
    """Assess ERT data quality deterministically.

    Args:
        contact_resistances: List of contact resistance values (ohms).
        reciprocity_errors: List of reciprocal measurement errors (%).
        stacking_errors: List of stacking error values (%).
        electrode_count: Number of electrodes used.
        max_acceptable_contact_resistance: Max acceptable contact resistance (ohms).
        max_acceptable_reciprocity_pct: Max acceptable reciprocity error (%).
        max_acceptable_stacking_pct: Max acceptable stacking error (%).

    Returns:
        ERTQualityResult with quality score and details.
    """
    warnings: list[str] = []

    # --- Contact resistance ---
    contact_ok = True
    contact_score = 1.0
    if contact_resistances:
        high_resistance = sum(
            1 for r in contact_resistances if r > max_acceptable_contact_resistance
        )
        bad_pct = (high_resistance / len(contact_resistances)) * 100
        if bad_pct > 20:
            contact_ok = False
            warnings.append(
                f"{bad_pct:.0f}% of electrodes have high contact resistance "
                f"(>{max_acceptable_contact_resistance} Ω)"
            )
        contact_score = max(0.0, 1.0 - bad_pct / 100)
    else:
        warnings.append("No contact resistance data available")
        contact_score = 0.5  # Neutral when data missing

    # --- Reciprocity error ---
    recip_err: float | None = None
    recip_score = 1.0
    if reciprocity_errors:
        import numpy as np

        recip_err = float(np.mean(np.abs(reciprocity_errors)))
        if recip_err > max_acceptable_reciprocity_pct:
            warnings.append(
                f"Mean reciprocity error ({recip_err:.1f}%) exceeds "
                f"acceptable threshold ({max_acceptable_reciprocity_pct}%)"
            )
        recip_score = max(0.0, 1.0 - recip_err / (max_acceptable_reciprocity_pct * 4))
    else:
        warnings.append("No reciprocity data available")
        recip_score = 0.5

    # --- Stacking error ---
    stack_err: float | None = None
    stack_score = 1.0
    if stacking_errors:
        import numpy as np

        stack_err = float(np.mean(np.abs(stacking_errors)))
        if stack_err > max_acceptable_stacking_pct:
            warnings.append(
                f"Mean stacking error ({stack_err:.1f}%) exceeds "
                f"acceptable threshold ({max_acceptable_stacking_pct}%)"
            )
        stack_score = max(0.0, 1.0 - stack_err / (max_acceptable_stacking_pct * 4))
    else:
        stack_score = 0.5

    # --- Combined score (weighted) ---
    quality_score = 0.4 * contact_score + 0.3 * recip_score + 0.3 * stack_score
    quality_score = round(max(0.0, min(1.0, quality_score)), 4)

    return ERTQualityResult(
        quality_score=quality_score,
        contact_resistance_ok=contact_ok,
        reciprocity_error_pct=recip_err,
        stacking_error_pct=stack_err,
        electrode_count=electrode_count,
        warnings=warnings,
    )
