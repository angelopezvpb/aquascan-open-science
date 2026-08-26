"""Tests for data normalization."""

from aquascan.data.normalization import (
    _safe_float,
    _safe_str,
    detect_duplicates,
    normalize_campaign_record,
    normalize_field_name,
)


class TestSafeFloat:
    """Test the _safe_float helper — null never becomes zero."""

    def test_none_returns_none(self) -> None:
        assert _safe_float(None) is None

    def test_nan_returns_none(self) -> None:
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self) -> None:
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_string_number(self) -> None:
        assert _safe_float("3.14") == 3.14

    def test_invalid_string(self) -> None:
        assert _safe_float("not a number") is None

    def test_zero_is_zero(self) -> None:
        """Zero is valid — this is different from unknown."""
        assert _safe_float(0) == 0.0
        assert _safe_float(0.0) == 0.0

    def test_empty_string(self) -> None:
        assert _safe_float("") is None

    def test_none_is_not_zero(self) -> None:
        """CRITICAL: None must never become 0."""
        result = _safe_float(None)
        assert result is None
        assert result != 0


class TestSafeStr:
    """Test the _safe_str helper."""

    def test_none_returns_none(self) -> None:
        assert _safe_str(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _safe_str("") is None

    def test_whitespace_returns_none(self) -> None:
        assert _safe_str("   ") is None

    def test_valid_string(self) -> None:
        assert _safe_str("hello") == "hello"


class TestNormalizeFieldName:
    """Test field name normalization."""

    def test_snake_case(self) -> None:
        assert normalize_field_name("Hello World") == "hello_world"

    def test_spanish_to_english(self) -> None:
        assert normalize_field_name("cota") == "elevation"
        assert normalize_field_name("caudal") == "pump_test_flow"
        assert normalize_field_name("profundidad") == "total_depth"

    def test_already_normalized(self) -> None:
        assert normalize_field_name("latitude") == "latitude"


class TestNormalizeCampaignRecord:
    """Test record normalization."""

    def test_basic_normalization(self) -> None:
        raw = {
            "campaign_id": "CAMP-001",
            "latitude": 40.0,
            "longitude": -3.0,
            "total_depth": 50.0,
        }
        record = normalize_campaign_record(raw)
        assert record.campaign_id == "CAMP-001"
        assert record.latitude == 40.0
        assert record.total_depth == 50.0

    def test_normalization_preserves_provenance(self) -> None:
        raw = {"campaign_id": "CAMP-002"}
        record = normalize_campaign_record(raw, source_uri="test://source")
        assert record.provenance.source_uri == "test://source"
        assert record.provenance.processing_step == "normalize"

    def test_null_fields_remain_null(self) -> None:
        raw = {
            "campaign_id": "CAMP-003",
            "pump_test_flow": None,
        }
        record = normalize_campaign_record(raw)
        assert record.pump_test_flow is None

    def test_generates_id_for_missing(self) -> None:
        raw = {}
        record = normalize_campaign_record(raw)
        assert record.campaign_id is not None
        assert len(record.campaign_id) > 0


class TestDetectDuplicates:
    """Test duplicate detection."""

    def test_no_duplicates(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "latitude": [40.0, 41.0, 42.0],
                "longitude": [-3.0, -3.0, -3.0],
                "total_depth": [50.0, 60.0, 70.0],
            }
        )
        result = detect_duplicates(df)
        assert all(r is None for r in result["_duplicate_of"])

    def test_with_duplicates(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "latitude": [40.0, 40.0, 41.0],
                "longitude": [-3.0, -3.0, -3.0],
                "total_depth": [50.0, 50.0, 60.0],
            }
        )
        result = detect_duplicates(df)
        # First duplicate should be marked
        assert result["_duplicate_of"].iloc[1] is not None
        assert result["_duplicate_of"].iloc[0] is None
        assert result["_duplicate_of"].iloc[2] is None
