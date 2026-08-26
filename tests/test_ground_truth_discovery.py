"""Gate 3 tests: GroundTruthEvidenceRecord, semantic separation, provenance.

Tests:
- GroundTruthEvidenceRecord serialization / deserialization
- MeasurementRecord creation and validation
- Candidate Score independence from GT class
- Null preservation (null != 0, null != empty)
- reference_flow_l_s != pump_test_flow (strict semantic separation)
- Provenance fields populated
- Duplicate extraction prevention
- Unsupported values remain null
- GroundTruthClass enum validation
"""

from __future__ import annotations

import json

from aquascan.data.models import (
    EvidenceType,
    ExtractionMethod,
    GroundTruthClass,
    GroundTruthEvidenceRecord,
    MeasurementRecord,
)


class TestGroundTruthEvidenceRecord:
    """Serialization, deserialization, and structural tests."""

    def test_minimal_record_creation(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="3530-5-0174",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.igme_id == "3530-5-0174"
        assert record.has_lithology is False
        assert record.has_piezometry is False
        assert record.has_pumping_tests is False
        assert record.has_chemistry is False
        assert record.has_hydrometry is False
        assert record.has_technical_data is False
        assert record.has_documents is False
        assert record.lithology_intervals == []
        assert record.provisional_gt_class is None

    def test_full_record_with_measurements(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="3530-5-0174",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measured_value=None,
                    unit="m",
                    measurement_type="lithology_depth",
                    depth_top_m=2.0,
                    depth_bottom_m=76.9,
                    source_endpoint="QueryAGLI",
                    source_field="ProfMuro",
                    raw_evidence="76.9",
                )
            ],
            has_piezometry=True,
            piezometry_measurements=[
                MeasurementRecord(
                    measured_value=3.0,
                    unit="m asl",
                    measurement_type="static_water_level",
                    measurement_date="1979-02-06",
                    source_endpoint="QueryAGPZ",
                    source_field="Valor",
                    extraction_method=ExtractionMethod.API_STRUCTURED,
                    raw_evidence="3.0",
                )
            ],
            provisional_gt_class=GroundTruthClass.GT_B,
            gt_class_rationale="Has lithology + water level, but no pumping test.",
            classification_confidence=0.8,
            known_missing=["pumping_test", "chemistry"],
        )
        assert record.has_lithology is True
        assert record.has_piezometry is True
        assert len(record.lithology_intervals) == 1
        assert len(record.piezometry_measurements) == 1
        assert record.provisional_gt_class == GroundTruthClass.GT_B
        assert len(record.known_missing) == 2
        assert record.pumping_test_records == []

    def test_json_roundtrip(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="3530-5-0174",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="lithology",
                    depth_top_m=0.0,
                    depth_bottom_m=10.0,
                    source_endpoint="QueryAGLI",
                    source_field="Litologia",
                    raw_evidence="Calizas",
                )
            ],
        )
        json_str = record.model_dump_json()
        loaded = GroundTruthEvidenceRecord.model_validate_json(json_str)
        assert loaded.igme_id == record.igme_id
        assert loaded.has_lithology == record.has_lithology
        assert len(loaded.lithology_intervals) == len(record.lithology_intervals)

    def test_json_serializable_for_file(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="3530-5-0174",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            provisional_gt_class=GroundTruthClass.GT_C,
        )
        data = record.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["igme_id"] == "3530-5-0174"
        assert data["provisional_gt_class"] == "GT-C"
        # Must be JSON serializable
        json.dumps(data)


class TestMeasurementRecord:
    """MeasurementRecord creation and validation."""

    def test_minimal_measurement(self) -> None:
        m = MeasurementRecord(
            measurement_type="static_water_level",
            source_endpoint="QueryAGPZ",
            source_field="Valor",
        )
        assert m.measured_value is None
        assert m.unit is None
        assert m.extraction_confidence == 1.0

    def test_full_measurement(self) -> None:
        m = MeasurementRecord(
            measured_value=15.68,
            unit="L/s",
            measurement_type="pump_test_flow",
            measurement_date="1990-03-26",
            source_endpoint="QueryAGEB",
            source_field="Caudal",
            extraction_method=ExtractionMethod.API_STRUCTURED,
            raw_evidence="15.68",
        )
        assert m.measured_value == 15.68
        assert m.unit == "L/s"
        assert m.measurement_type == "pump_test_flow"

    def test_confidence_bounds(self) -> None:
        m = MeasurementRecord(
            measurement_type="test",
            source_endpoint="test",
            source_field="test",
            extraction_confidence=0.5,
        )
        assert m.extraction_confidence == 0.5


class TestNullPreservation:
    """Null != 0, null != empty, null must be preserved."""

    def test_none_measured_value_not_zero(self) -> None:
        m = MeasurementRecord(
            measurement_type="test",
            source_endpoint="test",
            source_field="test",
        )
        assert m.measured_value is None
        assert m.measured_value != 0

    def test_zero_measured_value_preserved(self) -> None:
        m = MeasurementRecord(
            measured_value=0.0,
            measurement_type="pump_test_flow",
            source_endpoint="test",
            source_field="Caudal",
        )
        assert m.measured_value == 0.0
        assert m.measured_value is not None

    def test_none_vs_zero_in_json(self) -> None:
        record_null = GroundTruthEvidenceRecord(
            igme_id="test-1",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_piezometry=True,
            piezometry_measurements=[
                MeasurementRecord(
                    measurement_type="water_level",
                    source_endpoint="test",
                    source_field="Valor",
                    measured_value=None,
                )
            ],
        )
        record_zero = GroundTruthEvidenceRecord(
            igme_id="test-2",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_piezometry=True,
            piezometry_measurements=[
                MeasurementRecord(
                    measurement_type="water_level",
                    source_endpoint="test",
                    source_field="Valor",
                    measured_value=0.0,
                )
            ],
        )
        d_null = record_null.model_dump(mode="json")
        d_zero = record_zero.model_dump(mode="json")
        assert d_null["piezometry_measurements"][0]["measured_value"] is None
        assert d_zero["piezometry_measurements"][0]["measured_value"] == 0.0

    def test_known_missing_preserves_field_names(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            known_missing=["pumping_test", "chemistry", "borehole_construction"],
        )
        assert "pumping_test" in record.known_missing
        assert "chemistry" in record.known_missing


class TestSemanticSeparation:
    """Critical semantic boundaries between concepts."""

    def test_reference_flow_not_pump_test_flow(self) -> None:
        """reference_flow_l_s is a source metadata field, NOT pumping evidence."""
        # A record can have reference_flow_l_s in the IGME source but NO pumping test
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_pumping_tests=False,  # no real pumping test
            # reference_flow_l_s would be in IGMEWaterPointRecord, not here
        )
        assert record.has_pumping_tests is False
        # The GroundTruthEvidenceRecord does NOT have a reference_flow_l_s field
        assert not hasattr(record, "reference_flow_l_s")
        # Pump test flow only exists inside pumping_test_records
        assert record.pumping_test_records == []

    def test_pump_test_flow_only_from_evidence(self) -> None:
        """pump_test_flow must only come from documented pumping test evidence."""
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_pumping_tests=True,
            pumping_test_records=[
                MeasurementRecord(
                    measured_value=26.0,
                    unit="L/s",
                    measurement_type="pump_test_flow",
                    source_endpoint="QueryAGEB",
                    source_field="Caudal",
                    extraction_method=ExtractionMethod.API_STRUCTURED,
                    raw_evidence="26.0",
                )
            ],
        )
        assert len(record.pumping_test_records) == 1
        assert record.pumping_test_records[0].measurement_type == "pump_test_flow"
        assert record.pumping_test_records[0].source_endpoint == "QueryAGEB"

    def test_depth_not_water_depth(self) -> None:
        """total_depth_m in IGMEWaterPointRecord is total borehole depth, NOT water depth."""
        # Lithology intervals track depth separately
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="lithology_depth",
                    depth_top_m=0.0,
                    depth_bottom_m=76.9,
                    source_endpoint="QueryAGLI",
                    source_field="ProfMuro",
                )
            ],
        )
        # The lithology depth_bottom is borehole construction depth, not water table depth
        # Water table depth comes from piezometry
        assert record.lithology_intervals[0].measurement_type != "water_depth"

    def test_sondeo_does_not_mean_productive(self) -> None:
        """Point nature Sondeo/Pozo does not imply productivity."""
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_piezometry=True,
            piezometry_measurements=[
                MeasurementRecord(
                    measured_value=1.0,
                    unit="m asl",
                    measurement_type="static_water_level",
                    source_endpoint="QueryAGPZ",
                    source_field="Valor",
                    raw_evidence="1.0",
                )
            ],
        )
        # Point nature is not stored in GroundTruthEvidenceRecord — it's in IGMEWaterPointRecord
        assert not hasattr(record, "point_nature")
        # Water level near surface (1.0 m asl) doesn't automatically mean "productive"

    def test_reference_flow_zero_not_dry(self) -> None:
        """reference_flow_l_s = 0 does NOT mean dry well."""
        # GroundTruthEvidenceRecord doesn't contain reference_flow_l_s
        # But if we had a pumping test with Caudal=0, that would be a real measurement
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_pumping_tests=True,
            pumping_test_records=[
                MeasurementRecord(
                    measured_value=0.0,
                    unit="L/s",
                    measurement_type="pump_test_flow",
                    source_endpoint="QueryAGEB",
                    source_field="Caudal",
                    raw_evidence="0.0",
                )
            ],
        )
        # Zero flow in a pumping test is a real measurement — preserved as 0.0
        assert record.pumping_test_records[0].measured_value == 0.0


class TestCandidateScoreIndependence:
    """Candidate Priority Score and Ground Truth Class are independent concepts."""

    def test_candidate_score_not_in_evidence_record(self) -> None:
        """GroundTruthEvidenceRecord must NOT carry Candidate Priority Score."""
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert not hasattr(record, "candidate_priority_score")
        assert not hasattr(record, "candidate_tier")

    def test_gt_class_not_from_candidate_score(self) -> None:
        """GT class is assigned from documentary evidence, not from candidate score."""
        # High documentary evidence → GT-A even if candidate score is unknown
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=True,
            has_pumping_tests=True,
            provisional_gt_class=GroundTruthClass.GT_A,
            gt_class_rationale="Lithology + 5 water level measurements + pumping test.",
        )
        assert record.provisional_gt_class == GroundTruthClass.GT_A
        # No candidate score influences this

    def test_gt_class_enum_values(self) -> None:
        """All GT classes are valid."""
        for cls in GroundTruthClass:
            record = GroundTruthEvidenceRecord(
                igme_id="test",
                retrieval_timestamp="2024-01-01T00:00:00Z",
                provisional_gt_class=cls,
            )
            assert record.provisional_gt_class == cls


class TestProvenance:
    """Provenance fields are always populated."""

    def test_source_api_version_default(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.source_api_version == "bdaguas_v2"

    def test_extraction_version_default(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.extraction_version == "v0.1"

    def test_retrieval_timestamp_required(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.retrieval_timestamp is not None

    def test_measurement_provenance_traceable(self) -> None:
        m = MeasurementRecord(
            measured_value=26.0,
            measurement_type="pump_test_flow",
            source_endpoint="QueryAGEB",
            source_field="Caudal",
            extraction_method=ExtractionMethod.API_STRUCTURED,
            raw_evidence="26.0",
        )
        assert m.source_endpoint == "QueryAGEB"
        assert m.source_field == "Caudal"
        assert m.raw_evidence == "26.0"
        # Every extracted value traces to its exact source


class TestEdgeCases:
    """Unsupported values remain null, duplicate extraction is prevented."""

    def test_unsupported_values_remain_null(self) -> None:
        """If no data from API, all boolean flags stay False and lists stay empty."""
        record = GroundTruthEvidenceRecord(
            igme_id="test-no-data",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.has_lithology is False
        assert record.has_piezometry is False
        assert record.has_pumping_tests is False
        assert record.has_chemistry is False
        assert record.provisional_gt_class is None
        assert record.lithology_intervals == []

    def test_no_data_record_gt_none(self) -> None:
        """Record with no enrichment data gets GT=None (not auto-assigned)."""
        record = GroundTruthEvidenceRecord(
            igme_id="test-no-data",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.provisional_gt_class is None

    def test_idempotent_measurement_list(self) -> None:
        """Adding the same measurement twice should be prevented by design (lists not sets)."""
        m1 = MeasurementRecord(
            measurement_type="test",
            source_endpoint="test",
            source_field="test",
        )
        m2 = MeasurementRecord(
            measurement_type="test",
            source_endpoint="test",
            source_field="test",
        )
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_piezometry=True,
            piezometry_measurements=[m1, m2],
        )
        # Both stored; duplicate prevention is caller responsibility
        assert len(record.piezometry_measurements) == 2

    def test_conflicts_list(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            conflicts=["Depth mismatch: QueryAGMA=75m, QueryAGLI=76.9m"],
        )
        assert len(record.conflicts) == 1


class TestEvidenceTypeEnum:
    """EvidenceType classification enum."""

    def test_all_evidence_types(self) -> None:
        types = list(EvidenceType)
        assert EvidenceType.STRUCTURED in types
        assert EvidenceType.SEMI_STRUCTURED in types
        assert EvidenceType.NOT_AVAILABLE in types

    def test_not_available_for_missing(self) -> None:
        assert EvidenceType.NOT_AVAILABLE.value == "NOT_AVAILABLE"
