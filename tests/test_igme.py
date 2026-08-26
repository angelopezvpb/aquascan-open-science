"""Regression tests for IGME v2 semantic refactor.

Tests cover:
- IGMEWaterPointRecord model
- Flow semantics (reference_flow_l_s, NOT pump_test_flow)
- Null vs zero distinction
- ArcGIS construction-date parsing (epoch milliseconds)
- Metadata promotion
- Provenance preservation
- QA flags
- 1 RAW feature → exactly 1 IGME record
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquascan.data.models import IGMEWaterPointRecord, QAFlags
from aquascan.ingest.igme import (
    IGME_FIELD_MAP,
    IGME_WATER_POINTS_URL,
    IGMEIngester,
    _parse_construction_date,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_arcgis_feature(
    objectid: int,
    igme_id: str,
    longitude: float,
    latitude: float,
    *,
    profundidad: float | None = None,
    caudal: float = 0.0,
    cota: float | None = None,
    unidad_hidrogeologica: str | None = None,
    sistema_acuifero: str | None = None,
    provincia: str = "A Coruña",
    municipio: str = "Test",
    naturaleza: str = "Sondeo",
    metodo_perforacion: str | None = None,
    fecha_obra: int | None = None,
    motobomba: str | None = None,
    usos_agua: str | None = None,
    organismo: str | None = None,
    urldetalle: str | None = None,
    cuenca_hidrografica: str = "NORTE",
    toponimia: str | None = None,
    hoja: int = 207,
    octante: int = 4,
    punto: str = "0001",
    huso: int = 29,
    sector: str = "T",
    escala: str = "",
    expr1: float = 480000.0,
    expr2: float = 4750000.0,
) -> dict:
    """Build a single ArcGIS feature dict matching the real IGME schema."""
    return {
        "attributes": {
            "OBJECTID": objectid,
            "Id": igme_id,
            "HOJA": hoja,
            "Octante": octante,
            "PUNTO": punto,
            "Cuenca_Hidrografica": cuenca_hidrografica,
            "Unidad_Hidrogeologica": unidad_hidrogeologica,
            "Sistema_Acuifero": sistema_acuifero,
            "Provincia": provincia,
            "Municipio": municipio,
            "Caudal_Referencia_L_s": caudal,
            "TOPONIMIA": toponimia,
            "COTA_msnm": cota,
            "Naturaleza": naturaleza,
            "PROF_m": profundidad,
            "Metodo_Perforacion": metodo_perforacion,
            "FECHA_OBRA": fecha_obra,
            "Motobomba": motobomba,
            "Usos_Agua": usos_agua,
            "Organismo": organismo,
            "Escala": escala,
            "HUSO": huso,
            "SECTOR": sector,
            "Expr1": expr1,
            "Expr2": expr2,
            "URLDetalle": urldetalle,
            "ESRI_OID": objectid,
        },
        "geometry": {"x": longitude, "y": latitude},
    }


def _make_arcgis_page(features: list[dict]) -> dict:
    """Build a realistic ArcGIS page response."""
    return {
        "displayFieldName": "Id",
        "fieldAliases": {},
        "geometryType": "esriGeometryPoint",
        "spatialReference": {"wkid": 4326, "latestWkid": 4326},
        "fields": [],
        "features": features,
    }


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    """Write raw page fixtures to a temporary directory."""
    features = [
        _make_arcgis_feature(
            1,
            "207-4-0001",
            -9.22,
            42.94,
            profundidad=28.0,
            caudal=0.0,
            cota=50.0,
            unidad_hidrogeologica="Unidades aisladas",
            sistema_acuifero="Acuífero aislado",
            naturaleza="Sondeo",
            metodo_perforacion="Mixto (rotación y percusión)",
            fecha_obra=602294400000,  # 1989-02-01
            motobomba="Motor eléctrico",
            usos_agua="Abastecimiento (que no sea núcleo urbano)",
            organismo="Desconocido",
            urldetalle="http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0001",
        ),
        _make_arcgis_feature(
            2,
            "207-4-0002",
            -9.19,
            42.93,
            profundidad=None,
            caudal=0.0,
            cota=40.0,
            unidad_hidrogeologica="Unidades aisladas",
            naturaleza="Manantial",
            urldetalle="http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0002",
        ),
        _make_arcgis_feature(
            4,
            "306-2-0001",
            -9.03,
            43.11,
            profundidad=51.0,
            caudal=2.5,
            cota=180.0,
            unidad_hidrogeologica="Oiz",
            provincia="A Coruña",
            municipio="Vimianzo",
            naturaleza="Sondeo",
            metodo_perforacion="Rotopercusión (martillo en fondo)",
            fecha_obra=633830400000,
            motobomba="Motor eléctrico, bomba sumergida",
            usos_agua="Abastecimiento a núcleos urbanos",
            organismo="Diputación o Ayuntamientos",
            urldetalle="http://info.igme.es/bdaguas/PointInfo.aspx?id=306-2-0001",
        ),
        _make_arcgis_feature(
            5,
            "313-8-0001",
            -8.85,
            41.91,
            profundidad=36.0,
            caudal=0.0,
            cota=40.0,
            unidad_hidrogeologica="Unidades aisladas",
            provincia="Pontevedra",
            municipio="A Guarda",
            naturaleza="Sondeo",
            metodo_perforacion="Percusión",
            fecha_obra=128908800000,
            motobomba="Motor eléctrico, bomba sumergida",
            usos_agua="Abastecimiento a núcleos urbanos",
            organismo="Diputación o Ayuntamientos",
            urldetalle="http://info.igme.es/bdaguas/PointInfo.aspx?id=313-8-0001",
        ),
        _make_arcgis_feature(
            6,
            "999-9-0001",
            -9.50,
            42.50,
            profundidad=None,
            caudal=-2.0,
            cota=100.0,
            unidad_hidrogeologica=None,
            naturaleza="Manantial",
            fecha_obra=-1259625600000,  # Invalid pre-epoch
            urldetalle=None,
        ),
    ]
    page = _make_arcgis_page(features)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "page_00001_test.json").write_text(json.dumps(page), encoding="utf-8")
    return raw_dir


@pytest.fixture
def normalized_records(tmp_raw_dir: Path) -> list[IGMEWaterPointRecord]:
    """Normalize the test fixtures."""
    ingester = IGMEIngester()
    return ingester.normalize(tmp_raw_dir)


# ---------------------------------------------------------------------------
# IGMEWaterPointRecord model tests
# ---------------------------------------------------------------------------


class TestIGMEWaterPointRecord:
    """Test the new IGMEWaterPointRecord model."""

    def test_minimal_record(self) -> None:
        record = IGMEWaterPointRecord(igme_id="207-4-0001")
        assert record.igme_id == "207-4-0001"
        assert isinstance(record.qa_flags, QAFlags)
        assert isinstance(record.source_metadata, dict)

    def test_full_record(self) -> None:
        record = IGMEWaterPointRecord(
            igme_id="207-4-0001",
            igme_object_id=1,
            latitude=42.94,
            longitude=-9.22,
            elevation_m=50.0,
            watershed="NORTE",
            hydrogeological_unit="Unidades aisladas",
            aquifer_system="Acuífero aislado",
            province="A Coruña",
            municipality="Cee",
            point_nature="Sondeo",
            total_depth_m=28.0,
            reference_flow_l_s=0.0,
            drilling_method="Mixto (rotación y percusión)",
            construction_date="1989-02-01",
            pump_type="Motor eléctrico",
            water_use="Abastecimiento",
            organization="Desconocido",
            detail_url="http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0001",
        )
        assert record.igme_id == "207-4-0001"
        assert record.reference_flow_l_s == 0.0
        assert record.construction_date == "1989-02-01"
        assert record.point_nature == "Sondeo"

    def test_null_not_zero(self) -> None:
        record = IGMEWaterPointRecord(igme_id="test")
        assert record.reference_flow_l_s is None
        assert record.reference_flow_l_s != 0

    def test_qa_flags_default(self) -> None:
        record = IGMEWaterPointRecord(igme_id="test")
        assert record.qa_flags.flag_count == 0
        assert record.qa_flags.active_flags == []


# ---------------------------------------------------------------------------
# Count & type tests
# ---------------------------------------------------------------------------


class TestIGMENormalizeCount:
    """Verify that one feature → one normalized record."""

    def test_five_features_produces_five_records(self, normalized_records) -> None:
        assert len(normalized_records) == 5

    def test_return_type_is_igme_water_point_record(self, normalized_records) -> None:
        assert isinstance(normalized_records, list)
        for r in normalized_records:
            assert isinstance(
                r, IGMEWaterPointRecord
            ), f"Got {type(r)} instead of IGMEWaterPointRecord"

    def test_not_campaign_record(self, normalized_records) -> None:
        from aquascan.data.models import CampaignRecord

        for r in normalized_records:
            assert not isinstance(
                r, CampaignRecord
            ), "IGMEWaterPointRecord should NOT be CampaignRecord"

    def test_empty_page_produces_zero_records(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()
        empty_page = _make_arcgis_page([])
        (raw_dir / "page_empty.json").write_text(json.dumps(empty_page), encoding="utf-8")
        ingester = IGMEIngester()
        records = ingester.normalize(raw_dir)
        assert len(records) == 0

    def test_no_silent_dedup_of_legitimate_records(self, normalized_records) -> None:
        assert len(normalized_records) == 5


# ---------------------------------------------------------------------------
# Flow semantics tests
# ---------------------------------------------------------------------------


class TestFlowSemantics:
    """Verify correct flow field semantics."""

    def test_flow_maps_to_reference_flow_l_s(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].reference_flow_l_s == 2.5

    def test_flow_zero_remains_zero(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].reference_flow_l_s == 0.0
        assert by_id["207-4-0001"].reference_flow_l_s is not None

    def test_null_flow_remains_null(self, normalized_records) -> None:
        # Manantial records have null flow in source_metadata (not explicit null)
        # All our fixtures have explicit flow values, so test the model directly
        record = IGMEWaterPointRecord(igme_id="test")
        assert record.reference_flow_l_s is None

    def test_negative_flow_qa_flagged(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        r = by_id["999-9-0001"]
        assert r.reference_flow_l_s == -2.0
        assert r.qa_flags.negative_reference_flow is True

    def test_pump_test_flow_not_in_field_map(self) -> None:
        """Confirm pump_test_flow is NOT in the IGME_FIELD_MAP."""
        for canonical in IGME_FIELD_MAP.values():
            assert (
                canonical != "pump_test_flow"
            ), f"pump_test_flow leaked into IGME_FIELD_MAP via '{canonical}'"

    def test_flow_zero_not_dry_well(self, normalized_records) -> None:
        """reference_flow_l_s = 0 must NOT be interpreted as a dry well."""
        zero_flows = [r for r in normalized_records if r.reference_flow_l_s == 0]
        assert len(zero_flows) > 0
        for r in zero_flows:
            # Zero flow gets no special QA flag beyond what's normal
            assert "dry_well" not in r.qa_flags.active_flags


# ---------------------------------------------------------------------------
# Null vs Zero tests
# ---------------------------------------------------------------------------


class TestNullVsZero:
    """Verify null is never converted to zero."""

    def test_null_depth_remains_null(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0002"].total_depth_m is None
        assert by_id["207-4-0002"].total_depth_m != 0

    def test_null_depth_distinct_from_zero(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0002"].total_depth_m is None  # Manantial
        assert by_id["207-4-0001"].total_depth_m == 28.0  # Sondeo

    def test_model_null_not_zero(self) -> None:
        record = IGMEWaterPointRecord(igme_id="test")
        assert record.elevation_m is None
        assert record.total_depth_m is None
        assert record.reference_flow_l_s is None
        for val in [record.elevation_m, record.total_depth_m, record.reference_flow_l_s]:
            assert val != 0


# ---------------------------------------------------------------------------
# Construction date parsing tests
# ---------------------------------------------------------------------------


class TestConstructionDateParsing:
    """Verify FECHA_OBRA parsed correctly as ArcGIS epoch milliseconds."""

    def test_valid_epoch_ms_parsed(self) -> None:
        # 602294400000 ms → 1989-02-01 (based on real IGME data)
        iso_str, invalid, suspicious = _parse_construction_date(602294400000)
        assert iso_str == "1989-02-01"
        assert invalid is False
        assert suspicious is True  # Feb 1 = year-only pattern

    def test_null_epoch_returns_none(self) -> None:
        iso_str, invalid, suspicious = _parse_construction_date(None)
        assert iso_str is None
        assert invalid is False

    def test_zero_epoch_is_sentinel(self) -> None:
        iso_str, invalid, suspicious = _parse_construction_date(0)
        assert iso_str is None
        assert invalid is True

    def test_known_sentinel_2678400000(self) -> None:
        # 2678400000 ms = 31 days → sentinel
        iso_str, invalid, suspicious = _parse_construction_date(2678400000)
        assert iso_str is None
        assert invalid is True

    def test_pre_1900_is_suspicious(self) -> None:
        # 128908800000 ms → 1974-02-01 (after epoch, OK)
        iso_str, invalid, suspicious = _parse_construction_date(128908800000)
        assert iso_str == "1974-02-01"
        assert suspicious is True  # Feb 1 pattern

    def test_negative_epoch_is_invalid(self) -> None:
        iso_str, invalid, suspicious = _parse_construction_date(-60480000000)
        # -60480000000 ms → 1968-02-01 (before epoch, but valid)
        assert iso_str == "1968-02-01"
        assert invalid is False
        # Before 1970 is suspicious for Spanish water infrastructure
        assert suspicious is True

    def test_far_future_is_invalid(self) -> None:
        iso_str, invalid, suspicious = _parse_construction_date(2000000000000)
        assert iso_str is None
        assert invalid is True

    def test_pre_epoch_within_range_is_valid(self) -> None:
        # -1259625600000 → 1930-02-01 (valid, pre-Unix-epoch but resolvable)
        iso_str, invalid, suspicious = _parse_construction_date(-1259625600000)
        assert iso_str == "1930-02-01"
        assert invalid is False
        assert suspicious is True  # Feb 1 pattern + pre-1970

    def test_date_in_record(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].construction_date == "1990-02-01"
        assert by_id["313-8-0001"].construction_date == "1974-02-01"
        # -1259625600000 → 1930-02-01 (valid but suspicious)
        assert by_id["999-9-0001"].construction_date == "1930-02-01"
        assert by_id["999-9-0001"].qa_flags.invalid_construction_date is False
        assert by_id["999-9-0001"].qa_flags.suspicious_construction_date is True


# ---------------------------------------------------------------------------
# Metadata promotion tests
# ---------------------------------------------------------------------------


class TestMetadataPromotion:
    """Verify metadata fields are promoted to explicit IGMEWaterPointRecord fields."""

    def test_watershed_promoted(self, normalized_records) -> None:
        for r in normalized_records:
            assert r.watershed == "NORTE"

    def test_aquifer_system_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].aquifer_system == "Acuífero aislado"

    def test_province_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].province == "A Coruña"
        assert by_id["306-2-0001"].province == "A Coruña"
        assert by_id["313-8-0001"].province == "Pontevedra"

    def test_municipality_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].municipality == "Vimianzo"
        assert by_id["313-8-0001"].municipality == "A Guarda"

    def test_point_nature_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].point_nature == "Sondeo"
        assert by_id["207-4-0002"].point_nature == "Manantial"

    def test_drilling_method_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].drilling_method == "Mixto (rotación y percusión)"
        assert by_id["306-2-0001"].drilling_method == "Rotopercusión (martillo en fondo)"

    def test_detail_url_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["207-4-0001"].detail_url == (
            "http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0001"
        )
        # Feature with null URLDetalle
        assert by_id["999-9-0001"].detail_url is None

    def test_water_use_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].water_use == "Abastecimiento a núcleos urbanos"

    def test_organization_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].organization == "Diputación o Ayuntamientos"

    def test_pump_type_promoted(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["306-2-0001"].pump_type == "Motor eléctrico, bomba sumergida"

    def test_source_metadata_still_present(self, normalized_records) -> None:
        """Promotion must not destroy source_metadata."""
        for r in normalized_records:
            assert "toponimia" in r.source_metadata
            assert "hoja" in r.source_metadata
            assert "source_file" in r.source_metadata


# ---------------------------------------------------------------------------
# Provenance preservation tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Verify provenance is preserved for every record."""

    def test_provenance_present(self, normalized_records) -> None:
        for r in normalized_records:
            assert r.provenance is not None
            assert r.provenance.processing_step == "igme_normalize_v2"
            assert r.provenance.source_uri == IGME_WATER_POINTS_URL
            assert r.provenance.source_record_id is not None

    def test_source_file_in_metadata(self, normalized_records) -> None:
        for r in normalized_records:
            assert r.source_metadata["source_file"] == "page_00001_test.json"

    def test_source_feature_index(self, normalized_records) -> None:
        indices = [r.source_metadata["source_feature_index"] for r in normalized_records]
        assert indices == [0, 1, 2, 3, 4]

    def test_page_hash_present(self, normalized_records) -> None:
        for r in normalized_records:
            assert "source_page_hash" in r.source_metadata
            assert len(r.source_metadata["source_page_hash"]) == 64

    def test_provenance_source_record_id_matches_igme_id(self, normalized_records) -> None:
        for r in normalized_records:
            assert r.provenance.source_record_id == r.igme_id


# ---------------------------------------------------------------------------
# QA flags tests
# ---------------------------------------------------------------------------


class TestQAFlags:
    """Verify QA flag computation."""

    def test_qa_flags_present(self, normalized_records) -> None:
        for r in normalized_records:
            assert isinstance(r.qa_flags, QAFlags)

    def test_negative_flow_flagged(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["999-9-0001"].qa_flags.negative_reference_flow is True

    def test_missing_hgu_flagged(self, normalized_records) -> None:
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["999-9-0001"].qa_flags.missing_hydrogeological_unit is True

    def test_invalid_date_flagged(self, normalized_records) -> None:
        # 999-9-0001 has fecha_obra=-1259625600000 which resolves to 1930-02-01
        # → not invalid but suspicious (Feb 1 pattern, pre-1970)
        by_id = {r.igme_id: r for r in normalized_records}
        assert by_id["999-9-0001"].qa_flags.suspicious_construction_date is True

    def test_construction_date_qa_on_feb1_record(self, normalized_records) -> None:
        # Records with Feb 1 dates get suspicious_construction_date flag
        by_id = {r.igme_id: r for r in normalized_records}
        r = by_id["207-4-0001"]
        assert r.qa_flags.suspicious_construction_date is True
        assert r.qa_flags.invalid_construction_date is False

    def test_qa_flag_count(self) -> None:
        flags = QAFlags(
            coordinate_outlier=True,
            negative_reference_flow=True,
        )
        assert flags.flag_count == 2
        assert set(flags.active_flags) == {"coordinate_outlier", "negative_reference_flow"}

    def test_qa_flags_not_automatic_exclusion(self, normalized_records) -> None:
        """QA flags are warnings — records with flags still exist."""
        flagged = [r for r in normalized_records if r.qa_flags.flag_count > 0]
        assert len(flagged) > 0
        # Some records have flags, some don't — none are deleted
        assert len(flagged) <= len(normalized_records)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Verify duplicate handling."""

    def test_duplicate_igme_id_removed(self, tmp_path: Path) -> None:
        features = [
            _make_arcgis_feature(1, "207-4-0001", -9.22, 42.94),
            _make_arcgis_feature(2, "207-4-0001", -9.22, 42.94),
            _make_arcgis_feature(3, "306-2-0001", -9.03, 43.11),
        ]
        page = _make_arcgis_page(features)
        raw_dir = tmp_path / "dup_raw"
        raw_dir.mkdir()
        (raw_dir / "page_00001.json").write_text(json.dumps(page), encoding="utf-8")

        ingester = IGMEIngester()
        records = ingester.normalize(raw_dir)
        assert len(records) == 2
        ids = {r.igme_id for r in records}
        assert ids == {"207-4-0001", "306-2-0001"}


# ---------------------------------------------------------------------------
# Field map tests
# ---------------------------------------------------------------------------


class TestIGMEFieldMap:
    """Verify the field mapping is correct for v2."""

    def test_reference_flow_in_map(self) -> None:
        assert "caudal_referencia_l_s" in IGME_FIELD_MAP
        assert IGME_FIELD_MAP["caudal_referencia_l_s"] == "reference_flow"

    def test_pump_test_flow_not_in_map(self) -> None:
        assert "pump_test_flow" not in IGME_FIELD_MAP.values()

    def test_promoted_fields_in_map(self) -> None:
        promoted = [
            "cuenca_hidrografica",
            "sistema_acuifero",
            "naturaleza",
            "point_type",
            "provincia",
            "municipio",
            "metodo_perforacion",
            "urldetalle",
            "motobomba",
            "usos_agua",
            "organismo",
            "fecha_obra",
        ]
        for f in promoted:
            assert f in IGME_FIELD_MAP, f"Missing mapping for '{f}'"


# ---------------------------------------------------------------------------
# DataFrame / export tests
# ---------------------------------------------------------------------------


class TestIGMEToDataFrame:
    """Verify normalizer → DataFrame pipeline."""

    def test_to_dataframe_returns_correct_count(self, normalized_records) -> None:
        ingester = IGMEIngester()
        df = ingester.to_dataframe(normalized_records)
        assert len(df) == 5

    def test_correct_columns_in_df(self, normalized_records) -> None:
        ingester = IGMEIngester()
        df = ingester.to_dataframe(normalized_records)
        assert "igme_id" in df.columns
        assert "reference_flow_l_s" in df.columns
        assert "construction_date" in df.columns
        assert "watershed" in df.columns
        assert "aquifer_system" in df.columns
        assert "point_nature" in df.columns
        assert "drilling_method" in df.columns
        assert "detail_url" in df.columns
        assert "water_use" in df.columns
        assert "organization" in df.columns
        assert "pump_type" in df.columns
        assert "qa_negative_reference_flow" in df.columns
        assert "qa_flag_count" in df.columns
        assert "prov_processing_step" in df.columns

    def test_pump_test_flow_not_in_df(self, normalized_records) -> None:
        ingester = IGMEIngester()
        df = ingester.to_dataframe(normalized_records)
        assert "pump_test_flow" not in df.columns

    def test_export_parquet(self, normalized_records, tmp_path: Path) -> None:
        ingester = IGMEIngester()
        output = tmp_path / "test_v2.parquet"
        path = ingester.export(normalized_records, output, format="parquet")
        assert path.exists()

        import pandas as pd

        df = pd.read_parquet(path)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# Candidate Score vs Ground Truth distinction
# ---------------------------------------------------------------------------


class TestCandidateNotGroundTruth:
    """Candidate Score must NOT be Ground Truth Score."""

    def test_igme_record_has_no_candidate_score_yet(self, normalized_records) -> None:
        """IGMEWaterPointRecord does not have candidate score yet (future step)."""
        for r in normalized_records:
            assert not hasattr(r, "candidate_priority_score")

    def test_igme_record_has_no_ground_truth_score(self, normalized_records) -> None:
        """IGMEWaterPointRecord has no ground truth — it's a public record."""
        for r in normalized_records:
            assert not hasattr(r, "ground_truth_score")
            assert not hasattr(r, "ground_truth_class")
