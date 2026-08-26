"""Scientific feature construction (L3 — ML Ready).

Builds feature arrays from CampaignRecords for machine learning.
"""

from __future__ import annotations

import pandas as pd

from aquascan.data.models import CampaignRecord


def build_features(
    records: list[CampaignRecord],
    include_ground_truth: bool = False,
) -> pd.DataFrame:
    """Construct scientific features from campaign records.

    Args:
        records: List of campaign records.
        include_ground_truth: If True, include ground truth columns.
            Set to False when building features for prediction.

    Returns:
        DataFrame with engineered features.
    """
    rows: list[dict[str, object]] = []

    for r in records:
        row: dict[str, float | str | None] = {
            "campaign_id": r.campaign_id,
            "site_id": r.site_id,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "elevation": r.elevation,
        }

        # Geophysical features
        row["ert_quality"] = r.ert_quality
        row["seismic_quality"] = r.seismic_quality
        row["ert_resistivity_mean"] = r.ert_resistivity_mean
        row["ert_resistivity_std"] = r.ert_resistivity_std
        row["seismic_vp_mean"] = r.seismic_vp_mean
        row["seismic_vp_std"] = r.seismic_vp_std

        # Derived features
        if r.ert_resistivity_mean is not None and r.seismic_vp_mean is not None:
            if r.seismic_vp_mean > 0:
                row["resistivity_vp_ratio"] = r.ert_resistivity_mean / r.seismic_vp_mean
            else:
                row["resistivity_vp_ratio"] = None

        # Depth features
        row["water_depth_min"] = r.water_depth_min
        row["water_depth_max"] = r.water_depth_max
        row["total_depth"] = r.total_depth

        if (
            r.water_depth_min is not None
            and r.water_depth_max is not None
            and r.total_depth is not None
            and r.total_depth > 0
        ):
            row["water_depth_ratio"] = (r.water_depth_min + r.water_depth_max) / 2 / r.total_depth
        else:
            row["water_depth_ratio"] = None

        # Quality interaction features
        if r.ert_quality is not None and r.seismic_quality is not None:
            row["quality_product"] = r.ert_quality * r.seismic_quality
        else:
            row["quality_product"] = None

        # Categorical context (one-hot encoding done downstream)
        row["geology_domain"] = r.geology_domain
        row["hydrogeological_unit"] = r.hydrogeological_unit

        if include_ground_truth:
            row["ground_truth_score"] = r.ground_truth_score
            row["ground_truth_class"] = r.ground_truth_class.value if r.ground_truth_class else None
            row["static_water_level"] = r.static_water_level
            row["pump_test_flow"] = r.pump_test_flow
            row["water_electrical_conductivity"] = r.water_electrical_conductivity

        rows.append(row)

    df = pd.DataFrame(rows)

    # One-hot encode geological context
    if "geology_domain" in df.columns:
        dummies = pd.get_dummies(df["geology_domain"], prefix="geo")
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["geology_domain"])

    if "hydrogeological_unit" in df.columns:
        dummies = pd.get_dummies(df["hydrogeological_unit"], prefix="hgu")
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["hydrogeological_unit"])

    return df


def to_ml_dataframe(
    records: list[CampaignRecord],
    target_col: str = "ground_truth_score",
    drop_ids: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convert campaign records to ML-ready X, y.

    Args:
        records: List of campaign records.
        target_col: Name of the column to use as target.
        drop_ids: If True, drop ID columns from features.

    Returns:
        Tuple of (feature DataFrame, target Series or None).
    """
    df = build_features(records, include_ground_truth=True)

    # Separate IDs
    id_cols = [c for c in ["campaign_id", "site_id"] if c in df.columns]
    if not drop_ids:
        _ids = df[id_cols].copy()  # noqa: F841

    # Extract target
    y = df[target_col] if target_col in df.columns else None

    # Drop non-feature columns
    cols_to_drop = id_cols + [
        c
        for c in [
            "latitude",
            "longitude",
            "ground_truth_score",
            "ground_truth_class",
            "static_water_level",
            "pump_test_flow",
            "water_electrical_conductivity",
        ]
        if c in df.columns and c != target_col
    ]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return X, y
