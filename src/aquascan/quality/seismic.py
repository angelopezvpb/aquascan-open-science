"""Seismic quality assessment (deterministic, public domain methods only).

Assesses seismic data quality based on measurable acquisition parameters:
- First-break signal-to-noise ratio
- Geophone coupling consistency
- Shot repeatability
- Trace completeness
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeismicQualityResult:
    """Result of seismic quality assessment."""

    quality_score: float  # 0-1
    snr_ok: bool
    geophone_coupling_ok: bool
    mean_snr: float | None = None
    trace_count: int = 0
    dead_traces: int = 0
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def assess_seismic_quality(
    snr_values: list[float] | None = None,
    dead_trace_indices: list[int] | None = None,
    total_traces: int = 0,
    min_acceptable_snr: float = 3.0,
    max_dead_trace_pct: float = 10.0,
) -> SeismicQualityResult:
    """Assess seismic data quality deterministically.

    Args:
        snr_values: Signal-to-noise ratio per trace.
        dead_trace_indices: Indices of dead/bad traces.
        total_traces: Total number of traces.
        min_acceptable_snr: Minimum acceptable SNR.
        max_dead_trace_pct: Maximum acceptable percentage of dead traces.

    Returns:
        SeismicQualityResult with quality score and details.
    """
    warnings: list[str] = []

    # --- SNR assessment ---
    snr_ok = True
    snr_score = 1.0
    mean_snr: float | None = None

    if snr_values:
        import numpy as np

        mean_snr = float(np.mean(snr_values))
        if mean_snr < min_acceptable_snr:
            snr_ok = False
            warnings.append(
                f"Mean SNR ({mean_snr:.1f}) below acceptable threshold ({min_acceptable_snr})"
            )
            snr_score = max(0.0, mean_snr / min_acceptable_snr)
        else:
            snr_score = 1.0
    else:
        warnings.append("No SNR data available")
        snr_score = 0.5

    # --- Dead traces ---
    coupling_ok = True
    dead_count = len(dead_trace_indices) if dead_trace_indices else 0
    trace_score = 1.0

    if total_traces > 0:
        dead_pct = (dead_count / total_traces) * 100
        if dead_pct > max_dead_trace_pct:
            coupling_ok = False
            warnings.append(
                f"{dead_count}/{total_traces} ({dead_pct:.1f}%) dead traces "
                f"exceeds threshold ({max_dead_trace_pct}%)"
            )
            trace_score = max(0.0, 1.0 - dead_pct / (max_dead_trace_pct * 3))
        elif dead_pct > 0:
            warnings.append(f"{dead_count}/{total_traces} ({dead_pct:.1f}%) dead traces")
        trace_score = max(0.0, 1.0 - dead_pct / (max_dead_trace_pct * 3))
    elif total_traces == 0:
        warnings.append("No trace count information")
        trace_score = 0.5

    # --- Combined score ---
    quality_score = 0.5 * snr_score + 0.5 * trace_score
    quality_score = round(max(0.0, min(1.0, quality_score)), 4)

    return SeismicQualityResult(
        quality_score=quality_score,
        snr_ok=snr_ok,
        geophone_coupling_ok=coupling_ok,
        mean_snr=mean_snr,
        trace_count=total_traces,
        dead_traces=dead_count,
        warnings=warnings,
    )
