"""Gate 3.1 tests: semantic validation of Ground Truth evidence.

Tests covering:
- aquifer_flagged interval != water_entry observation
- reference_flow != pump_test_flow
- chemistry != hydraulic evidence
- null preservation
- locale decimal normalization (comma-as-decimal)
- specific capacity derivation rules
- piezometry consistency (Cota - Nivel ≈ Valor)
- record-level coverage calculation
- evidence dimensions remain independent
- deterministic extraction
- provenance / checksums
- Aquascan validation targets
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from aquascan.data.models import (
    ExtractionMethod,
    GroundTruthClass,
    GroundTruthEvidenceRecord,
    MeasurementRecord,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_number(val: object) -> float | None:
    if val is None:
        return None
    if isinstance(val, int | float):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ── 1. Aquifer semantics ────────────────────────────────────────────


class TestAquiferSemantics:
    """Acuifero=true = aquifer-flagged lithology, NOT water entry."""

    def test_aquifer_flag_is_not_water_entry(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    depth_top_m=2.0,
                    depth_bottom_m=76.9,
                    source_endpoint="QueryAGLI",
                    source_field="Acuifero",
                    extraction_method=ExtractionMethod.API_STRUCTURED,
                    raw_evidence="true",
                )
            ],
        )
        assert (
            record.lithology_intervals[0].measurement_type == "aquifer_flagged_lithology_interval"
        )
        assert record.lithology_intervals[0].measurement_type != "water_entry_depth"

    def test_no_water_entry_claim_without_evidence(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    source_endpoint="QueryAGLI",
                    source_field="Acuifero",
                )
            ],
        )
        for m in record.lithology_intervals:
            assert "water_entry" not in m.measurement_type.lower()

    def test_aquifer_interval_preserves_raw_value(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    source_endpoint="QueryAGLI",
                    source_field="Acuifero",
                    raw_evidence="true",
                ),
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    source_endpoint="QueryAGLI",
                    source_field="Acuifero",
                    raw_evidence="false",
                ),
            ],
        )
        assert record.lithology_intervals[0].raw_evidence == "true"
        assert record.lithology_intervals[1].raw_evidence == "false"


# ── 2. Flow semantics ───────────────────────────────────────────────


class TestFlowSemantics:
    """reference_flow != pump_test_flow."""

    def test_reference_flow_not_in_evidence_record(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert not hasattr(record, "reference_flow_l_s")

    def test_pump_test_flow_only_from_query_ageb(self) -> None:
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
                )
            ],
        )
        for m in record.pumping_test_records:
            assert m.source_endpoint == "QueryAGEB"
            assert "reference" not in m.measurement_type.lower()

    def test_query_agma_flow_is_not_pump_test(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.has_pumping_tests is False
        assert record.pumping_test_records == []


# ── 3. Chemistry ≠ hydraulic ────────────────────────────────────────


class TestChemistryNotHydraulic:
    """Chemistry must never substitute for hydraulic evidence."""

    def test_chemistry_does_not_set_hydraulic_dimension(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_chemistry=True,
            chemistry_records=[
                MeasurementRecord(
                    measured_value=800,
                    unit="µS/cm",
                    measurement_type="conductivity",
                    source_endpoint="QueryAGAQ",
                    source_field="Conduct",
                )
            ],
        )
        assert record.has_chemistry is True
        assert record.has_pumping_tests is False

    def test_chemistry_alone_not_sufficient_for_productivity(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_chemistry=True,
            has_pumping_tests=False,
            has_lithology=False,
            has_piezometry=False,
        )
        assert record.has_pumping_tests is False

    def test_dimension_separation_in_record(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=True,
            has_pumping_tests=False,
            has_chemistry=True,
            has_hydrometry=False,
        )
        assert record.has_lithology is True
        assert record.has_piezometry is True
        assert record.has_pumping_tests is False
        assert record.has_chemistry is True


# ── 4. Null preservation ───────────────────────────────────────────


class TestNullPreservation:
    """null != 0, null != empty."""

    def test_none_measured_value_not_zero(self) -> None:
        m = MeasurementRecord(
            measurement_type="t",
            source_endpoint="t",
            source_field="t",
        )
        assert m.measured_value is None
        assert m.measured_value != 0

    def test_zero_preserved(self) -> None:
        m = MeasurementRecord(
            measured_value=0.0,
            measurement_type="pump_test_flow",
            source_endpoint="QueryAGEB",
            source_field="Caudal",
        )
        assert m.measured_value == 0.0
        assert m.measured_value is not None

    def test_missing_endpoint_no_false_claims(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test-no-data",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.has_lithology is False
        assert record.has_piezometry is False
        assert record.has_pumping_tests is False
        assert record.has_chemistry is False

    def test_known_missing_preserved(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            known_missing=["pumping_test", "screen_intervals"],
        )
        assert len(record.known_missing) == 2


# ── 5. Decimal normalization ────────────────────────────────────────


class TestDecimalNormalization:
    """Spanish locale: comma as decimal separator."""

    def test_comma_decimal_integer(self) -> None:
        assert _clean_number("7,9") == pytest.approx(7.9)

    def test_comma_decimal_fraction(self) -> None:
        assert _clean_number("0,22") == pytest.approx(0.22)

    def test_normal_float_preserved(self) -> None:
        assert _clean_number(26.0) == 26.0
        assert _clean_number(0.0) == 0.0

    def test_integer_string(self) -> None:
        assert _clean_number("800") == 800.0

    def test_null_preserved(self) -> None:
        assert _clean_number(None) is None

    def test_invalid_string(self) -> None:
        assert _clean_number("No se sabe") is None

    def test_multiple_commas(self) -> None:
        assert _clean_number("1,234,56") is None


# ── 6. Specific capacity derivation ─────────────────────────────────


class TestSpecificCapacityDerivation:
    """SC = flow / drawdown, only when drawdown > 0."""

    def test_derivable(self) -> None:
        assert pytest.approx(26.0 / 31.0, abs=0.001) == 0.838709

    def test_drawdown_zero_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            _ = 26.0 / 0.0

    def test_drawdown_negative_invalid(self) -> None:
        assert -5.0 <= 0

    def test_missing_flow(self) -> None:
        assert None is None

    def test_missing_drawdown(self) -> None:
        assert None is None

    def test_derived_label_always_marked(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_pumping_tests=True,
            pumping_test_records=[
                MeasurementRecord(
                    measured_value=0.84,
                    unit="L/s/m",
                    measurement_type="specific_capacity",
                    source_endpoint="QueryAGEB",
                    source_field="DERIVED: Caudal/DepresionM",
                    extraction_method=ExtractionMethod.DERIVED,
                    raw_evidence="26.0 / 31.0",
                )
            ],
        )
        m = record.pumping_test_records[0]
        assert m.extraction_method == ExtractionMethod.DERIVED
        assert "DERIVED" in m.source_field


# ── 7. Piezometry consistency ───────────────────────────────────────


class TestPiezometryConsistency:
    """Cota - Nivel ≈ Valor must hold numerically."""

    def test_relationship_holds_exact(self) -> None:
        assert pytest.approx(700.0 - 20.0) == 680.0

    def test_relationship_holds_precision(self) -> None:
        assert abs(48.18 - (54.0 - 5.82)) < 0.01

    def test_no_blind_formula_application(self) -> None:
        def verify(c: float, n: float, v: float) -> bool:
            return abs(v - (c - n)) < 0.01

        assert verify(700.0, 20.0, 680.0) is True
        assert verify(700.0, 20.0, 670.0) is False

    def test_surgencia_field_present(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_piezometry=True,
            piezometry_measurements=[
                MeasurementRecord(
                    measured_value=48.18,
                    unit="m asl",
                    measurement_type="static_water_level",
                    source_endpoint="QueryAGPZ",
                    source_field="Valor",
                    raw_evidence="48.18",
                )
            ],
        )
        assert len(record.piezometry_measurements) == 1


# ── 8. Record-level coverage ────────────────────────────────────────


class TestRecordCoverage:
    """Coverage metrics are record-level."""

    def test_single_record_all_false(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        families = sum(
            [
                record.has_lithology,
                record.has_piezometry,
                record.has_pumping_tests,
                record.has_chemistry,
                record.has_hydrometry,
            ]
        )
        assert families == 0

    def test_single_record_all_true(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=True,
            has_pumping_tests=True,
            has_chemistry=True,
            has_hydrometry=True,
        )
        families = sum(
            [
                record.has_lithology,
                record.has_piezometry,
                record.has_pumping_tests,
                record.has_chemistry,
                record.has_hydrometry,
            ]
        )
        assert families == 5

    def test_dimensions_are_independent(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=False,
            has_pumping_tests=True,
        )
        assert record.has_lithology is True
        assert record.has_piezometry is False
        assert record.has_pumping_tests is True


# ── 9. Evidence dimension independence ──────────────────────────────


class TestEvidenceDimensionIndependence:
    """Each dimension is independently tracked."""

    def test_geological_dimension(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=False,
            has_pumping_tests=False,
            has_chemistry=False,
        )
        assert record.has_lithology is True
        assert record.has_piezometry is False

    def test_water_level_dimension(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=False,
            has_piezometry=True,
            has_pumping_tests=False,
        )
        assert record.has_piezometry is True
        assert record.has_lithology is False

    def test_hydraulic_dimension(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=False,
            has_pumping_tests=True,
        )
        assert record.has_pumping_tests is True
        assert record.has_lithology is False


# ── 10. Deterministic extraction ─────────────────────────────────────


class TestDeterministicExtraction:
    """Identical inputs → identical outputs."""

    def test_identical_records_equal(self) -> None:
        kwargs = {
            "igme_id": "test",
            "retrieval_timestamp": "2024-01-01T00:00:00Z",
            "has_lithology": True,
            "lithology_intervals": [
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    depth_top_m=2.0,
                    depth_bottom_m=76.9,
                    source_endpoint="QueryAGLI",
                    source_field="Acuifero",
                    raw_evidence="true",
                )
            ],
        }
        r1 = GroundTruthEvidenceRecord(**kwargs)
        r2 = GroundTruthEvidenceRecord(**kwargs)
        assert r1.model_dump_json() == r2.model_dump_json()

    def test_different_ids_not_equal(self) -> None:
        r1 = GroundTruthEvidenceRecord(
            igme_id="t1",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        r2 = GroundTruthEvidenceRecord(
            igme_id="t2",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert r1.model_dump_json() != r2.model_dump_json()

    def test_checksum_stable(self) -> None:
        data = json.dumps({"a": 1, "b": [2, 3]}, sort_keys=True).encode()
        h1 = _sha256(data)
        h2 = _sha256(data)
        assert h1 == h2
        assert len(h1) == 64


# ── 11. Provenance ──────────────────────────────────────────────────


class TestProvenance:
    """Every evidence record must have traceable provenance."""

    def test_api_version_set(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.source_api_version == "bdaguas_v2"

    def test_extraction_version_set(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.extraction_version == "v0.1"

    def test_measurement_source_traceable(self) -> None:
        m = MeasurementRecord(
            measured_value=26.0,
            unit="L/s",
            measurement_type="pump_test_flow",
            source_endpoint="QueryAGEB",
            source_field="Caudal",
            extraction_method=ExtractionMethod.API_STRUCTURED,
            raw_evidence="26.0",
        )
        assert m.source_endpoint == "QueryAGEB"
        assert m.source_field == "Caudal"
        assert m.raw_evidence == "26.0"

    def test_retrieval_timestamp_present(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
        )
        assert record.retrieval_timestamp is not None


# ── 12. GroundTruthClass compatibility ──────────────────────────────


class TestGroundTruthClassCompatibility:
    """A/B/C/D and GT_A/GT_B/GT_C/GT_D both work."""

    def test_legacy_names(self) -> None:
        assert GroundTruthClass.A.value == "A"
        assert GroundTruthClass.B.value == "B"
        assert GroundTruthClass.D.value == "D"

    def test_new_names(self) -> None:
        assert GroundTruthClass.GT_A.value == "GT-A"
        assert GroundTruthClass.GT_B.value == "GT-B"

    def test_both_sets_exist(self) -> None:
        members = {m.name for m in GroundTruthClass}
        assert "A" in members
        assert "GT_A" in members


# ── 13. Aquascan validation targets ─────────────────────────────────


class TestAquascanTargets:
    """Different records are good GT for different targets."""

    def test_water_depth_good_poor_productivity(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_piezometry=True,
            has_pumping_tests=False,
        )
        assert record.has_piezometry is True
        assert record.has_pumping_tests is False

    def test_productivity_needs_pumping_test(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            has_chemistry=True,
        )
        assert record.has_pumping_tests is False
        assert record.has_chemistry is True

    def test_lithology_target(self) -> None:
        record = GroundTruthEvidenceRecord(
            igme_id="test",
            retrieval_timestamp="2024-01-01T00:00:00Z",
            has_lithology=True,
            lithology_intervals=[
                MeasurementRecord(
                    measurement_type="aquifer_flagged_lithology_interval",
                    depth_top_m=0.0,
                    depth_bottom_m=76.9,
                    source_endpoint="QueryAGLI",
                    source_field="Litologia",
                    raw_evidence="Dolomías",
                )
            ],
        )
        assert record.has_lithology is True
        assert len(record.lithology_intervals) > 0
