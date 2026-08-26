"""Tests for provenance tracking system."""

from aquascan.data.provenance import (
    ProvenanceTracker,
    compute_sha256,
    create_provenance_record,
)


class TestSHA256:
    """Test SHA-256 computation."""

    def test_consistent_hash(self) -> None:
        h1 = compute_sha256(b"hello world")
        h2 = compute_sha256(b"hello world")
        assert h1 == h2
        assert len(h1) == 64  # Hex encoding of 32 bytes

    def test_different_hash(self) -> None:
        h1 = compute_sha256(b"hello world")
        h2 = compute_sha256(b"hello world!")
        assert h1 != h2

    def test_string_input(self) -> None:
        h_bytes = compute_sha256(b"test")
        h_str = compute_sha256("test")
        assert h_bytes == h_str


class TestProvenanceTracker:
    """Test provenance tracking across multiple transformations."""

    def test_empty_tracker(self) -> None:
        tracker = ProvenanceTracker()
        assert len(tracker.steps) == 0
        assert tracker.latest() is None

    def test_record_step(self) -> None:
        tracker = ProvenanceTracker()
        step = tracker.record(
            processing_step="igme_download",
            source_uri="https://services.arcgis.com/...",
        )
        assert step.processing_step == "igme_download"
        assert step.source_uri == "https://services.arcgis.com/..."
        assert len(tracker.steps) == 1

    def test_multiple_steps(self) -> None:
        tracker = ProvenanceTracker()
        tracker.record("download", source_hash="abc123")
        tracker.record("normalize", parent_artifact="abc123")
        tracker.record("validate")
        assert len(tracker.steps) == 3
        assert tracker.latest().processing_step == "validate"

    def test_json_export(self) -> None:
        tracker = ProvenanceTracker()
        tracker.record("download", source_hash="abc123")
        json_str = tracker.to_json()
        assert "download" in json_str
        assert "abc123" in json_str


class TestCreateProvenanceRecord:
    """Test provenance record factory."""

    def test_all_fields(self) -> None:
        record = create_provenance_record(
            processing_step="normalize",
            source_uri="file://data/raw/test.json",
            source_hash="abc123",
            parent_artifact="parent-001",
            instrument_version="v2.1",
        )
        assert record.processing_step == "normalize"
        assert record.source_uri == "file://data/raw/test.json"
        assert record.source_hash == "abc123"
        assert record.parent_artifact == "parent-001"
        assert record.instrument_version == "v2.1"

    def test_timestamp_auto_generated(self) -> None:
        record = create_provenance_record("test")
        assert record.timestamp is not None
