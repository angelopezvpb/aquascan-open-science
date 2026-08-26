"""Seismic basic processing — deterministic, public-domain methods.

Provides:
- Simple first-break picking (threshold-based)
- P-wave velocity estimation from first arrivals
- Basic refraction intercept-time method
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pick_first_breaks_simple(
    amplitudes: np.ndarray,
    threshold_factor: float = 3.0,
    min_sample: int = 5,
) -> np.ndarray:
    """Simple threshold-based first-break picker.

    Picks the first sample where amplitude exceeds
    threshold_factor * background_noise_rms.

    Note: This is a basic method for quick-look analysis. Production
    picking should use more sophisticated algorithms (STA/LTA, AIC, etc.).

    Args:
        amplitudes: 2D array (traces x samples).
        threshold_factor: Multiplier on noise RMS for threshold.
        min_sample: Minimum sample index to consider.

    Returns:
        Array of first-break sample indices per trace.
    """
    n_traces, n_samples = amplitudes.shape
    picks = np.full(n_traces, -1, dtype=int)

    for i in range(n_traces):
        trace = amplitudes[i]
        # Estimate noise from pre-trigger samples (first 10% or 50 samples)
        noise_window = min(50, max(10, n_samples // 10))
        noise_rms = np.sqrt(np.mean(trace[:noise_window] ** 2))
        threshold = threshold_factor * noise_rms

        above = np.where(np.abs(trace) > threshold)[0]
        above = above[above >= min_sample]
        if len(above) > 0:
            picks[i] = above[0]

    return picks


def compute_vp_from_first_arrivals(
    df: pd.DataFrame,
    offset_col: str = "offset_m",
    time_col: str = "travel_time_s",
) -> dict[str, float]:
    """Estimate P-wave velocity from first-arrival travel times.

    Uses linear regression on offset vs. travel time (straight-line fit).
    For a basic two-layer case, also attempts an intercept-time estimate.

    Args:
        df: DataFrame with offset and travel time columns.
        offset_col: Column for source-receiver offset (m).
        time_col: Column for first-arrival travel time (s).

    Returns:
        Dict with 'vp_direct_m_s' (direct wave velocity),
        'r_squared' (fit quality), and optionally 'vp_refraction_m_s'.
    """
    x = df[offset_col].values
    t = df[time_col].values

    # Remove any rows with NaN
    mask = ~(np.isnan(x) | np.isnan(t))
    x = x[mask]
    t = t[mask]

    if len(x) < 3:
        return {
            "vp_direct_m_s": float("nan"),
            "r_squared": float("nan"),
            "error": "Insufficient data (< 3 points)",
        }

    # Direct wave velocity: v = dx/dt
    # Linear regression: t = (1/v) * x + t0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, t)

    if slope <= 0:
        return {
            "vp_direct_m_s": float("nan"),
            "r_squared": float(r_value**2),
            "error": "Negative or zero slope — check data ordering",
        }

    vp_direct = 1.0 / slope  # m/s
    r_squared = r_value**2

    result: dict[str, float] = {
        "vp_direct_m_s": float(vp_direct),
        "r_squared": float(r_squared),
        "intercept_s": float(intercept),
    }

    # --- Optional: two-layer refraction (intercept-time method) ---
    # Look for a break in slope indicating a faster refractor
    if len(x) >= 8:
        # Try splitting at midpoint and fitting two lines
        mid = len(x) // 2
        # Near offsets
        try:
            slope1, _, _, _, _ = stats.linregress(x[:mid], t[:mid])
            slope2, _, _, _, _ = stats.linregress(x[mid:], t[mid:])
            if slope2 < slope1 and slope2 > 0:  # Faster refractor
                vp_refraction = 1.0 / slope2
                result["vp_refraction_m_s"] = float(vp_refraction)
        except (ValueError, IndexError):
            pass

    return result


def estimate_depth_to_refractor(
    vp_layer1: float,
    vp_layer2: float,
    intercept_time: float,
) -> float | None:
    """Estimate depth to a horizontal refractor (two-layer case).

    Uses the intercept-time method:
        h = (ti / 2) / sqrt(1/v1^2 - 1/v2^2)

    Args:
        vp_layer1: P-wave velocity of the top layer (m/s).
        vp_layer2: P-wave velocity of the refractor (m/s).
        intercept_time: Intercept time of the refraction (seconds).

    Returns:
        Depth to refractor in meters, or None if invalid.
    """
    if vp_layer1 <= 0 or vp_layer2 <= vp_layer1 or intercept_time <= 0:
        return None

    try:
        term = np.sqrt(1.0 / vp_layer1**2 - 1.0 / vp_layer2**2)
        if term <= 0:
            return None
        return float((intercept_time / 2.0) / term)
    except (ValueError, ZeroDivisionError):
        return None
