"""Geographic and domain-aware data splitting.

CRITICAL: Never use random row-level splits for geo/hydro data.
Same well/site/region must never appear in both train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_by_well_id(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create train/test splits grouped by well_id.

    All records from the same well stay together — never split a well
    across train and test.

    Args:
        df: DataFrame with well_id column.
        well_id_col: Name of the well ID column.
        n_splits: Number of folds.
        random_state: Random seed.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(len(df))

    if well_id_col not in df.columns:
        # Fall back to index-based groups if no well_id
        groups = np.arange(len(df))
    else:
        groups = df[well_id_col].values

    splits = list(gkf.split(X, groups=groups))
    return splits


def split_by_geology_domain(
    df: pd.DataFrame,
    domain_col: str = "geology_domain",
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create splits where train and test domains are disjoint.

    Uses GroupKFold on geological domain — ensures a domain never
    appears in both train and test.

    Args:
        df: DataFrame with geological domain column.
        domain_col: Name of the geology domain column.
        n_splits: Number of folds.
        random_state: Random seed.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(len(df))
    groups = df[domain_col].values if domain_col in df.columns else np.arange(len(df))
    return list(gkf.split(X, groups=groups))


class GeographicHoldoutSplit:
    """Holdout split based on geographic distance.

    Selects a random center point, then assigns the N nearest records
    to the test set. This simulates "we know about this region, can we
    predict a new area?"

    This is the most realistic evaluation for prospecting: we train on
    known areas and predict on unexplored ones.
    """

    def __init__(
        self,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ):
        self.test_fraction = test_fraction
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split by geographic distance from a random centroid.

        Args:
            df: DataFrame with latitude and longitude columns.
            lat_col: Latitude column name.
            lon_col: Longitude column name.

        Returns:
            (train_indices, test_indices) arrays.
        """
        rng = np.random.RandomState(self.random_state)

        n_test = max(1, int(len(df) * self.test_fraction))

        # Pick a random point as the test centroid
        center_idx = rng.randint(0, len(df))
        center_lat = df[lat_col].iloc[center_idx]
        center_lon = df[lon_col].iloc[center_idx]

        # Compute distances (approximate, flat-earth OK for small regions)
        lat = df[lat_col].values
        lon = df[lon_col].values

        # Simple Euclidean approximation in degree-space
        # (Good enough for splitting; not for real geodesy)
        dlat = lat - center_lat
        dlon = lon - center_lon
        # Scale longitude by cos(latitude) for rough equal-area
        cos_lat = np.cos(np.radians(np.mean(lat)))
        distances_sq = dlat**2 + (dlon * cos_lat) ** 2

        # Nearest to center go to test
        nearest = np.argsort(distances_sq)
        test_idx = nearest[:n_test]
        train_idx = nearest[n_test:]

        return train_idx, test_idx


class SpatialDistanceSplit:
    """Split ensuring minimum spatial distance between train and test.

    Iteratively assigns points to train/test while ensuring no test point
    is closer than min_distance_km to any train point.

    This is strict — some points may not be assigned to either set.
    """

    def __init__(
        self,
        min_distance_km: float = 1.0,
        test_fraction: float = 0.2,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.min_distance_km = min_distance_km
        self.test_fraction = test_fraction
        self.max_iter = max_iter
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split with minimum distance constraint.

        Args:
            df: DataFrame with lat/lon columns.
            lat_col: Latitude column name.
            lon_col: Longitude column name.

        Returns:
            (train_indices, test_indices, excluded_indices).
            Excluded are points too close to both train and test.
        """
        rng = np.random.RandomState(self.random_state)
        n_total = len(df)
        n_test_target = max(1, int(n_total * self.test_fraction))

        lat = df[lat_col].values
        lon = df[lon_col].values

        # Approximate degree-to-km conversion
        km_per_deg_lat = 111.32
        cos_lat = np.cos(np.radians(np.mean(lat)))
        km_per_deg_lon = km_per_deg_lat * cos_lat

        # Start with random test seed
        all_idx = set(range(n_total))
        test_set: set[int] = set()
        train_set: set[int] = set()

        # Pick test center randomly
        candidates = list(all_idx)
        rng.shuffle(candidates)

        for idx in candidates:
            if len(test_set) >= n_test_target:
                break

            # Check distance to existing test points
            too_close = False
            for t_idx in test_set:
                dlat = lat[idx] - lat[t_idx]
                dlon = (lon[idx] - lon[t_idx]) * cos_lat / km_per_deg_lat * km_per_deg_lat
                # Simple check — actual geodesic would be more precise
                d2 = (dlat * km_per_deg_lat) ** 2 + (dlon) ** 2
                if np.sqrt(d2) < self.min_distance_km:
                    too_close = True
                    break

            if not too_close:
                test_set.add(idx)

        # Train set: points far from ALL test points
        for idx in all_idx - test_set:
            far_from_all = True
            for t_idx in test_set:
                dlat = (lat[idx] - lat[t_idx]) * km_per_deg_lat
                dlon = (lon[idx] - lon[t_idx]) * km_per_deg_lon
                if np.sqrt(dlat**2 + dlon**2) < self.min_distance_km:
                    far_from_all = False
                    break
            if far_from_all:
                train_set.add(idx)

        excluded = all_idx - train_set - test_set

        return (
            np.array(sorted(train_set)),
            np.array(sorted(test_set)),
            np.array(sorted(excluded)),
        )


def check_geographic_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    well_id_col: str | None = "well_id",
    site_id_col: str | None = "site_id",
    max_distance_m: float = 10.0,
) -> dict[str, bool | int | list[str]]:
    """Check for data leakage between train and test sets.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        lat_col: Latitude column.
        lon_col: Longitude column.
        well_id_col: Well ID column (checked for overlap).
        site_id_col: Site ID column (checked for overlap).
        max_distance_m: Maximum allowed distance (m) for "same point" check.

    Returns:
        Dict with leak detection results.
    """
    issues: list[str] = []

    # Check well_id overlap
    well_leak = False
    if well_id_col and well_id_col in train_df.columns and well_id_col in test_df.columns:
        train_wells = set(train_df[well_id_col].dropna())
        test_wells = set(test_df[well_id_col].dropna())
        overlap = train_wells & test_wells
        if overlap:
            well_leak = True
            issues.append(f"Well ID overlap: {len(overlap)} wells in both train and test")

    # Check site_id overlap
    site_leak = False
    if site_id_col and site_id_col in train_df.columns and site_id_col in test_df.columns:
        train_sites = set(train_df[site_id_col].dropna())
        test_sites = set(test_df[site_id_col].dropna())
        overlap = train_sites & test_sites
        if overlap:
            site_leak = True
            issues.append(f"Site ID overlap: {len(overlap)} sites in both train and test")

    # Check geographic proximity
    geo_leak = False
    n_nearby = 0
    if lat_col in train_df.columns and lon_col in train_df.columns:
        # Approximate check: for each test point, find nearest train point
        # For efficiency, sample if test is large
        test_sample = test_df
        if len(test_df) > 100:
            test_sample = test_df.sample(100, random_state=42)

        for _, t_row in test_sample.iterrows():
            t_lat, t_lon = t_row[lat_col], t_row[lon_col]
            if pd.isna(t_lat) or pd.isna(t_lon):
                continue

            # Vectorized distance to all train points (approximate)
            dlat = (train_df[lat_col] - t_lat) * 111320.0  # meters
            dlon = (train_df[lon_col] - t_lon) * (111320.0 * np.cos(np.radians(t_lat)))
            min_dist = np.sqrt((dlat**2 + dlon**2).min())

            if min_dist < max_distance_m:
                n_nearby += 1

        if n_nearby > 0:
            geo_leak = True
            issues.append(
                f"Geographic leakage: {n_nearby}/{len(test_sample)} test points "
                f"within {max_distance_m}m of training data"
            )

    return {
        "has_leakage": well_leak or site_leak or geo_leak,
        "well_id_overlap": well_leak,
        "site_id_overlap": site_leak,
        "geographic_overlap": geo_leak,
        "n_nearby_points": n_nearby,
        "issues": issues,
    }
