"""ERT basic processing — deterministic, public-domain computations.

Provides geometric factors and apparent resistivity computation for
standard electrode arrays (Wenner, Schlumberger, dipole-dipole).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_geometric_factor_wenner(a: float) -> float:
    """Compute geometric factor K for Wenner array.

    Wenner: electrodes equally spaced at distance a.
    K = 2 * pi * a

    Args:
        a: Electrode spacing (meters).

    Returns:
        Geometric factor K.
    """
    return 2.0 * np.pi * a


def compute_geometric_factor_schlumberger(ab2: float, mn2: float) -> float:
    """Compute geometric factor K for Schlumberger array.

    Schlumberger: AB/2 and MN/2 configuration.
    K = pi * (AB/2)^2 / MN

    Args:
        ab2: Half the current electrode spacing (AB/2, meters).
        mn2: Half the potential electrode spacing (MN/2, meters).

    Returns:
        Geometric factor K.
    """
    mn = 2.0 * mn2
    if mn <= 0:
        raise ValueError(f"MN must be positive, got MN/2={mn2}")
    return np.pi * (ab2**2) / mn


def compute_geometric_factor_general(
    a_pos: float, b_pos: float, m_pos: float, n_pos: float
) -> float:
    """Compute geometric factor K for a general 4-electrode array.

    K = 2 * pi / (1/AM - 1/AN - 1/BM + 1/BN)
    where AM = |A-M|, AN = |A-N|, BM = |B-M|, BN = |B-N|

    Args:
        a_pos: Position of electrode A (meters).
        b_pos: Position of electrode B (meters).
        m_pos: Position of electrode M (meters).
        n_pos: Position of electrode N (meters).

    Returns:
        Geometric factor K.
    """
    am = abs(a_pos - m_pos)
    an = abs(a_pos - n_pos)
    bm = abs(b_pos - m_pos)
    bn = abs(b_pos - n_pos)

    # Avoid division by zero
    eps = 1e-12
    if am < eps or an < eps or bm < eps or bn < eps:
        raise ValueError("Electrode positions must be distinct")

    factor = 1.0 / am - 1.0 / an - 1.0 / bm + 1.0 / bn
    if abs(factor) < eps:
        raise ValueError("Geometric factor is singular for these positions")

    return (2.0 * np.pi) / factor


def compute_apparent_resistivity(
    df: pd.DataFrame,
    array_type: str = "general",
    resistance_col: str = "resistance_ohm",
    voltage_col: str | None = None,
    current_col: str | None = None,
) -> pd.DataFrame:
    """Compute apparent resistivity for ERT measurements.

    Apparent resistivity rho_a = K * R
    where K is the geometric factor and R = V/I.

    Args:
        df: DataFrame with electrode positions and resistance (or V, I).
        array_type: Type of array ('wenner', 'schlumberger', 'general').
        resistance_col: Column name for resistance (ohms).
        voltage_col: Column for voltage (mV) if resistance not provided.
        current_col: Column for current (mA) if resistance not provided.

    Returns:
        DataFrame with added 'apparent_resistivity_ohm_m' column.
    """
    out = df.copy()

    # Compute resistance if V and I are provided
    if resistance_col in out.columns:
        r = out[resistance_col].values
    elif voltage_col and current_col and voltage_col in out.columns and current_col in out.columns:
        v = out[voltage_col].values  # mV
        i = out[current_col].values  # mA
        r = (v / 1000.0) / (i / 1000.0)  # R = V/I in ohms
    else:
        raise ValueError(
            f"Need either '{resistance_col}' column or both "
            f"'{voltage_col}' and '{current_col}' columns"
        )

    # Compute geometric factor based on array type
    if array_type == "wenner":
        # Assumes equally spaced electrodes
        k = np.array(
            [
                compute_geometric_factor_wenner(abs(row["electrode_a"] - row["electrode_b"]) / 3)
                for _, row in out.iterrows()
            ]
        )
    elif array_type == "schlumberger":
        k = np.array(
            [
                compute_geometric_factor_schlumberger(
                    abs(row["electrode_a"] - row["electrode_b"]) / 2,
                    abs(row["electrode_m"] - row["electrode_n"]) / 2,
                )
                for _, row in out.iterrows()
            ]
        )
    else:  # general
        k = np.array(
            [
                compute_geometric_factor_general(
                    row["electrode_a"],
                    row["electrode_b"],
                    row["electrode_m"],
                    row["electrode_n"],
                )
                for _, row in out.iterrows()
            ]
        )

    out["geometric_factor"] = k
    out["apparent_resistivity_ohm_m"] = k * r

    return out


def estimate_mean_resistivity(
    df: pd.DataFrame,
    resistivity_col: str = "apparent_resistivity_ohm_m",
) -> dict[str, float]:
    """Compute basic statistics for a resistivity profile.

    Args:
        df: DataFrame with apparent resistivity data.
        resistivity_col: Column name for resistivity values.

    Returns:
        Dictionary with mean, std, median, min, max.
    """
    values = df[resistivity_col].dropna().values
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
