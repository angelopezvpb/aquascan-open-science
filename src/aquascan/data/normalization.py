"""Data normalization utilities.

Convert heterogeneous public data sources into the canonical AquaScan schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from aquascan.data.models import CampaignRecord
from aquascan.data.provenance import compute_sha256, create_provenance_record

# Sentinel value for "unknown" — never convert to zero
_UNKNOWN_SENTINEL = None


def _safe_float(value: Any) -> float | None:
    """Convert to float safely. Returns None for unparseable values.

    Critical: never converts None/NaN/empty to 0.0.
    """
    if value is None:
        return None
    if isinstance(value, int | float):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return float(value)
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert to int safely. Returns None for unparseable values.

    Critical: never converts None/NaN/empty to 0.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> str | None:
    """Convert to string safely. Returns None for empty/whitespace-only strings."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _safe_datetime(value: Any) -> datetime | None:
    """Parse datetime safely. Returns None for unparseable values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError):
        return None


def normalize_field_name(name: str) -> str:
    """Normalize a field name to snake_case.

    Handles common variations from IGME and other public sources.

    Args:
        name: Original field name.

    Returns:
        Snake_case normalized field name.
    """
    replacements = {
        # Spanish → English normalization
        "x_utm": "utm_x",
        "y_utm": "utm_y",
        "coordenada_x": "utm_x",
        "coordenada_y": "utm_y",
        "cota": "elevation",
        "altitud": "elevation",
        "profundidad": "total_depth",
        "caudal": "pump_test_flow",
        "conductividad": "water_electrical_conductivity",
        "naturaleza": "point_type",
        "unidad_hidrogeologica": "hydrogeological_unit",
        "municipio": "municipality",
        "provincia": "province",
        "fecha": "timestamp",
        "observaciones": "observations",
    }

    name = name.strip().lower()
    # Replace known Spanish field names
    name = replacements.get(name, name)
    # Snake_case: replace spaces and special chars with underscores
    name = "".join(c if c.isalnum() else "_" for c in name)
    name = name.strip("_").lower()
    # Collapse repeated underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name


def detect_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Detect potential duplicate records.

    Uses spatial + metadata coordinates to identify likely duplicates.

    Args:
        df: DataFrame with campaign records.
        subset: Columns to use for duplicate detection.

    Returns:
        DataFrame with a `_duplicate_of` column marking duplicates.
    """
    if subset is None:
        subset = ["latitude", "longitude", "total_depth"]

    available = [c for c in subset if c in df.columns]
    if len(available) < 2:
        df["_duplicate_of"] = None
        return df

    # Round coordinates to ~10m precision for fuzzy matching
    work = df.copy()
    if "latitude" in work.columns and "longitude" in work.columns:
        work["_lat_round"] = work["latitude"].round(4)
        work["_lon_round"] = work["longitude"].round(4)

    dup_cols = [c if c not in ("latitude", "longitude") else f"_{c[0:3]}_round" for c in available]
    available_dup = [c for c in dup_cols if c in work.columns]

    if len(available_dup) >= 2:
        work.duplicated(subset=available_dup, keep="first")
        # Mark duplicates with the index of the first occurrence
        groups = work.groupby(available_dup, dropna=False)
        first_idx = groups.transform("cumcount") == 0
        df["_duplicate_of"] = None
        for i in work[~first_idx].index:
            match = work.loc[
                (work[available_dup].eq(work.loc[i, available_dup]).all(axis=1)) & first_idx
            ]
            if not match.empty:
                df.at[i, "_duplicate_of"] = str(match.index[0])
    else:
        df["_duplicate_of"] = None

    return df


def normalize_campaign_record(
    raw: dict[str, Any],
    processing_step: str = "normalize",
    source_uri: str | None = None,
) -> CampaignRecord:
    """Normalize a raw record dict into a CampaignRecord.

    Args:
        raw: Raw record dictionary from a public source.
        processing_step: Name of the normalization step.
        source_uri: URI of the source.

    Returns:
        A CampaignRecord with normalized fields.
    """
    raw_bytes = str(sorted(raw.items())).encode("utf-8")
    source_hash = compute_sha256(raw_bytes)

    provenance = create_provenance_record(
        processing_step=processing_step,
        source_uri=source_uri,
        source_hash=source_hash,
    )

    return CampaignRecord(
        campaign_id=_safe_str(raw.get("campaign_id", raw.get("id"))) or provenance.source_record_id,
        site_id=_safe_str(raw.get("site_id", raw.get("codigo"))) or provenance.source_record_id,
        well_id=_safe_str(raw.get("well_id")),
        latitude=_safe_float(raw.get("latitude")),
        longitude=_safe_float(raw.get("longitude")),
        elevation=_safe_float(raw.get("elevation")),
        geology_domain=_safe_str(raw.get("geology_domain")),
        hydrogeological_unit=_safe_str(raw.get("hydrogeological_unit")),
        timestamp=_safe_datetime(raw.get("timestamp", raw.get("fecha"))),
        total_depth=_safe_float(raw.get("total_depth")),
        static_water_level=_safe_float(raw.get("static_water_level")),
        pump_test_flow=_safe_float(raw.get("pump_test_flow")),
        pump_test_duration=_safe_float(raw.get("pump_test_duration")),
        water_electrical_conductivity=_safe_float(raw.get("water_electrical_conductivity")),
        water_tds=_safe_float(raw.get("water_tds")),
        provenance=provenance,
    )


def dataframe_to_campaign_records(
    df: pd.DataFrame,
    source_uri: str | None = None,
) -> list[CampaignRecord]:
    """Convert a DataFrame of normalized records to CampaignRecord instances.

    Args:
        df: DataFrame with columns matching AquaScan field names.
        source_uri: Source URI for provenance.

    Returns:
        List of CampaignRecord instances.
    """
    records: list[CampaignRecord] = []
    for _, row in df.iterrows():
        record = normalize_campaign_record(
            row.to_dict(),
            processing_step="dataframe_to_campaign_records",
            source_uri=source_uri,
        )
        records.append(record)
    return records
