"""Seismic data I/O — basic CSV-based exchange format.

Full SEG-2/SEG-Y support is deferred to future versions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_seismic_csv(
    path: str | Path,
    source_col: str = "source_x",
    receiver_col: str = "receiver_x",
    time_col: str = "travel_time_ms",
    trace_col: str = "trace_id",
) -> pd.DataFrame:
    """Read first-arrival seismic data from a CSV file.

    Args:
        path: Path to CSV file.
        source_col: Column name for source position (meters).
        receiver_col: Column name for receiver position (meters).
        time_col: Column name for travel time (milliseconds).
        trace_col: Column name for trace identifier.

    Returns:
        DataFrame with standardized column names.
    """
    df = pd.read_csv(path)
    rename = {
        source_col: "source_x",
        receiver_col: "receiver_x",
        time_col: "travel_time_ms",
        trace_col: "trace_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Compute offset
    if "source_x" in df.columns and "receiver_x" in df.columns:
        df["offset_m"] = abs(df["source_x"] - df["receiver_x"])

    # Convert time to seconds
    df["travel_time_s"] = df["travel_time_ms"] / 1000.0

    return df


def save_seismic_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    format: str = "csv",
) -> None:
    """Save a processed seismic DataFrame.

    Args:
        df: DataFrame with seismic data.
        path: Output path.
        format: Output format ('csv' or 'parquet').
    """
    path = Path(path)
    if format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
