"""Tests for geographic split and data leakage prevention.

CRITICAL: These tests verify that data leakage is prevented.
Geo/hydro data must never use naive random splits.
"""

import numpy as np
import pandas as pd
import pytest

from aquascan.evaluation.geographic_split import (
    GeographicHoldoutSplit,
    check_geographic_leakage,
    split_by_geology_domain,
    split_by_well_id,
)


@pytest.fixture
def sample_geo_df() -> pd.DataFrame:
    """Create a sample DataFrame with spatial and well ID data."""
    rng = np.random.RandomState(42)
    n = 50

    return pd.DataFrame(
        {
            "campaign_id": [f"CAMP-{i:03d}" for i in range(n)],
            "well_id": [f"WELL-{i // 5:03d}" for i in range(n)],  # 5 records per well
            "site_id": [f"SITE-{i // 10:03d}" for i in range(n)],
            "latitude": 40.0 + rng.uniform(-0.5, 0.5, n),
            "longitude": -3.0 + rng.uniform(-0.5, 0.5, n),
            "geology_domain": rng.choice(["Granite", "Limestone", "Alluvial"], n),
            "target": rng.randint(0, 2, n),
        }
    )


class TestWellIdSplit:
    """Test that splits by well_id keep wells together."""

    def test_no_well_in_both_sets(self, sample_geo_df: pd.DataFrame) -> None:
        splits = split_by_well_id(sample_geo_df, n_splits=3)
        for train_idx, test_idx in splits:
            train_wells = set(sample_geo_df.iloc[train_idx]["well_id"])
            test_wells = set(sample_geo_df.iloc[test_idx]["well_id"])
            overlap = train_wells & test_wells
            assert len(overlap) == 0, f"Well ID leakage: {overlap} appears in both train and test"

    def test_splits_non_empty(self, sample_geo_df: pd.DataFrame) -> None:
        splits = split_by_well_id(sample_geo_df, n_splits=3)
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestGeologyDomainSplit:
    """Test splits by geological domain."""

    def test_no_domain_in_both_sets(self, sample_geo_df: pd.DataFrame) -> None:
        splits = split_by_geology_domain(sample_geo_df, n_splits=3)
        for train_idx, test_idx in splits:
            train_domains = set(sample_geo_df.iloc[train_idx]["geology_domain"])
            test_domains = set(sample_geo_df.iloc[test_idx]["geology_domain"])
            overlap = train_domains & test_domains
            assert len(overlap) == 0, f"Domain leakage: {overlap} in both train and test"


class TestGeographicHoldout:
    """Test geographic holdout split."""

    def test_train_test_disjoint(self, sample_geo_df: pd.DataFrame) -> None:
        splitter = GeographicHoldoutSplit(test_fraction=0.3, random_state=42)
        train_idx, test_idx = splitter.split(sample_geo_df)

        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, "Train and test indices overlap"

        assert len(test_idx) > 0
        assert len(train_idx) > 0

    def test_test_size_approximate(self, sample_geo_df: pd.DataFrame) -> None:
        splitter = GeographicHoldoutSplit(test_fraction=0.3, random_state=42)
        train_idx, test_idx = splitter.split(sample_geo_df)
        expected_test = max(1, int(len(sample_geo_df) * 0.3))
        assert len(test_idx) == expected_test


class TestLeakageDetection:
    """Test that leakage detection correctly identifies problems."""

    def test_no_leakage_clean_split(self, sample_geo_df: pd.DataFrame) -> None:
        """Clean split should show no leakage."""
        # Split by well_id guarantees no well overlap
        splits = split_by_well_id(sample_geo_df, n_splits=2)
        train_idx, test_idx = splits[0]

        train = sample_geo_df.iloc[train_idx]
        test = sample_geo_df.iloc[test_idx]

        result = check_geographic_leakage(train, test)
        # Site IDs may overlap if different wells are on the same site.
        # Well IDs must not overlap.
        assert result["well_id_overlap"] is False, f"Issues: {result['issues']}"

    def test_detects_well_id_overlap(self, sample_geo_df: pd.DataFrame) -> None:
        """If same well in both sets, leakage should be detected."""
        # Artificially create leakage
        train = sample_geo_df.iloc[:30]
        test = sample_geo_df.iloc[20:]  # Overlap!

        result = check_geographic_leakage(train, test)
        assert result["has_leakage"] is True
        assert result["well_id_overlap"] is True

    def test_detects_site_id_overlap(self, sample_geo_df: pd.DataFrame) -> None:
        """Same site in both sets is also leakage."""
        train = sample_geo_df.iloc[:30]
        test = sample_geo_df.iloc[20:]

        result = check_geographic_leakage(train, test)
        assert result["site_id_overlap"] is True

    def test_detects_geographic_proximity(self) -> None:
        """Very close points in train and test should be flagged."""
        train = pd.DataFrame(
            {
                "latitude": [40.0, 41.0],
                "longitude": [-3.0, -3.0],
            }
        )
        test = pd.DataFrame(
            {
                "latitude": [40.00001],  # Very close to 40.0
                "longitude": [-3.00001],
            }
        )

        result = check_geographic_leakage(train, test, max_distance_m=50.0)
        assert (
            result["geographic_overlap"] is True
        ), f"Should detect nearby points: {result['issues']}"

    def test_distant_points_no_geo_leak(self) -> None:
        """Far apart points should not trigger leakage."""
        train = pd.DataFrame(
            {
                "latitude": [40.0],
                "longitude": [-3.0],
            }
        )
        test = pd.DataFrame(
            {
                "latitude": [43.0],  # ~300 km away
                "longitude": [-8.0],
            }
        )

        result = check_geographic_leakage(train, test, max_distance_m=100.0)
        assert result["geographic_overlap"] is False


class TestRandomSplitWarning:
    """Test that naive random splits are explicitly warned against.

    This test doesn't implement a random split — it verifies that our
    infrastructure makes geographic splits the default and easy to use.
    """

    def test_geographic_holdout_is_easy(self, sample_geo_df: pd.DataFrame) -> None:
        """Geographic holdout should be the default/easy path."""
        splitter = GeographicHoldoutSplit()
        train_idx, test_idx = splitter.split(sample_geo_df)
        assert len(train_idx) + len(test_idx) == len(sample_geo_df)
