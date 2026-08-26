"""ERT data I/O — reading and writing common formats.

Supports basic CSV-based ERT data exchange. Full binary format support
(ABEM, Syscal, Res2DInv) is a future enhancement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_ert_csv(
    path: str | Path,
    a_col: str = "A",
    b_col: str = "B",
    m_col: str = "M",
    n_col: str = "N",
    resistance_col: str = "R",
    current_col: str | None = None,
    voltage_col: str | None = None,
) -> pd.DataFrame:
    """Read ERT measurement data from a CSV file.

    Expected columns: electrode positions (A, B, M, N),
    resistance (or voltage + current), optional metadata.

    Args:
        path: Path to CSV file.
        a_col: Column name for electrode A position.
        b_col: Column name for electrode B position.
        m_col: Column name for electrode M position.
        n_col: Column name for electrode N position.
        resistance_col: Column name for measured resistance (ohms).
        current_col: Optional column for injected current (mA).
        voltage_col: Optional column for measured voltage (mV).

    Returns:
        DataFrame with standardized column names.
    """
    df = pd.read_csv(path)

    # Standardize column names
    rename = {
        a_col: "electrode_a",
        b_col: "electrode_b",
        m_col: "electrode_m",
        n_col: "electrode_n",
        resistance_col: "resistance_ohm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if current_col and current_col in df.columns:
        df["current_ma"] = df[current_col]
    if voltage_col and voltage_col in df.columns:
        df["voltage_mv"] = df[voltage_col]

    # Add electrode spacing
    df["ab_spacing"] = np.abs(df["electrode_a"] - df["electrode_b"])
    df["mn_spacing"] = np.abs(df["electrode_m"] - df["electrode_n"])

    return df


def save_ert_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    format: str = "csv",
) -> None:
    """Save a processed ERT DataFrame.

    Args:
        df: DataFrame with ERT data.
        path: Output path.
        format: Output format ('csv' or 'parquet').
    """
    path = Path(path)
    if format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
