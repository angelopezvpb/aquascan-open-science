"""Integration tests for IGME normalization pipeline (v2).

These tests use realistic ArcGIS fixtures matching the real IGME schema
to verify the full normalization pipeline with IGMEWaterPointRecord.
"""

import json
from pathlib import Path

import pytest

from aquascan.data.models import IGMEWaterPointRecord
from aquascan.ingest.igme import IGME_FIELD_MAP, IGME_WATER_POINTS_URL, IGMEIngester

# ---------------------------------------------------------------------------
# Realistic ArcGIS fixture matching the actual IGME layer 0 schema
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def five_feature_page() -> dict:
    """A raw page with 5 IGME features, matching the real schema."""
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
            3,
            "207-8-0001",
            -9.26,
            42.89,
            profundidad=None,
            caudal=0.0,
            cota=100.0,
            unidad_hidrogeologica="Unidades aisladas",
            naturaleza="Manantial",
            urldetalle="http://info.igme.es/bdaguas/PointInfo.aspx?id=207-8-0001",
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
    ]
    return _make_arcgis_page(features)


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, five_feature_page: dict) -> Path:
    """Write a raw page fixture to a temporary directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    page_file = raw_dir / "page_00001_14a472b4132b.json"
    page_file.write_text(json.dumps(five_feature_page, indent=2), encoding="utf-8")
    return raw_dir


# ---------------------------------------------------------------------------
# Core normalization tests
# ---------------------------------------------------------------------------


class TestIGMENormalizeCount:
    """Verify that one feature → one normalized record."""

    def test_five_features_produces_five_records(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        assert len(records) == 5, f"Expected 5 records, got {len(records)}"

    def test_empty_page_produces_zero_records(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()
        empty_page = _make_arcgis_page([])
        (raw_dir / "page_00001_empty.json").write_text(json.dumps(empty_page), encoding="utf-8")
        ingester = IGMEIngester()
        records = ingester.normalize(raw_dir)
        assert len(records) == 0

    def test_return_type_is_igme_water_point_record(self, tmp_raw_dir: Path) -> None:
        """normalize() must return list[IGMEWaterPointRecord]."""
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        assert isinstance(records, list)
        for r in records:
            assert isinstance(
                r, IGMEWaterPointRecord
            ), f"Got {type(r)} instead of IGMEWaterPointRecord"


class TestIGMEFieldPreservation:
    """Verify that real IGME fields survive normalization with v2 semantics."""

    def test_igme_id_survives(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        ids = [r.igme_id for r in records]
        assert "207-4-0001" in ids
        assert "306-2-0001" in ids

    def test_depth_survives(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].total_depth_m == 28.0

    def test_depth_null_remains_null(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0002"].total_depth_m is None

    def test_flow_survives(self, tmp_raw_dir: Path) -> None:
        """Caudal_Referencia_L_s → reference_flow_l_s (NOT pump_test_flow)."""
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["306-2-0001"].reference_flow_l_s == 2.5

    def test_flow_zero_remains_zero(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].reference_flow_l_s == 0.0
        assert by_id["207-4-0001"].reference_flow_l_s is not None

    def test_longitude_latitude_survive(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        r1 = by_id["207-4-0001"]
        assert r1.longitude == pytest.approx(-9.22, abs=0.01)
        assert r1.latitude == pytest.approx(42.94, abs=0.01)

    def test_elevation_survives(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].elevation_m == 50.0

    def test_hydrogeological_unit_survives(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].hydrogeological_unit == "Unidades aisladas"
        assert by_id["306-2-0001"].hydrogeological_unit == "Oiz"

    def test_detail_url_promoted_and_survives(self, tmp_raw_dir: Path) -> None:
        """URLDetalle now promoted to detail_url field (was in source_metadata)."""
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].detail_url == (
            "http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0001"
        )


class TestIGMEProvenance:
    """Verify provenance is preserved (v2)."""

    def test_provenance_present(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        for r in records:
            assert r.provenance is not None
            assert r.provenance.processing_step == "igme_normalize_v2"
            assert r.provenance.source_uri == IGME_WATER_POINTS_URL
            assert r.provenance.source_record_id is not None

    def test_source_file_in_metadata(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        for r in records:
            assert "source_file" in r.source_metadata
            assert r.source_metadata["source_file"] == "page_00001_14a472b4132b.json"

    def test_source_feature_index(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        indices = [r.source_metadata["source_feature_index"] for r in records]
        assert indices == [0, 1, 2, 3, 4]

    def test_page_hash_present(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        for r in records:
            assert "source_page_hash" in r.source_metadata
            assert len(r.source_metadata["source_page_hash"]) == 64


class TestIGMEDeduplication:
    """Verify duplicate handling."""

    def test_no_silent_dedup_of_legitimate_records(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        assert len(records) == 5

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


class TestIGMEToDataFrame:
    """Verify normalizer → DataFrame pipeline with v2 model."""

    def test_to_dataframe_from_igme_records(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        df = ingester.to_dataframe(records)
        assert len(df) == 5

    def test_model_dump_produces_flat_dicts(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        df = ingester.to_dataframe(records)
        assert len(df) == 5
        assert "igme_id" in df.columns
        assert "total_depth_m" in df.columns
        assert "reference_flow_l_s" in df.columns
        assert "pump_test_flow" not in df.columns


class TestIGMEFieldMap:
    """Verify the field mapping covers real IGME fields for v2."""

    def test_real_field_names_mapped(self) -> None:
        real_fields = [
            "id",
            "prof_m",
            "caudal_referencia_l_s",
            "cota_msnm",
            "unidad_hidrogeologica",
            "sistema_acuifero",
            "cuenca_hidrografica",
            "naturaleza",
            "provincia",
            "municipio",
            "fecha_obra",
            "urldetalle",
            "objectid",
            "esri_oid",
            "hoja",
            "octante",
            "punto",
            "metodo_perforacion",
            "motobomba",
            "usos_agua",
            "organismo",
            "huso",
            "sector",
            "escala",
            "expr1",
            "expr2",
            "toponimia",
        ]
        for f in real_fields:
            assert f in IGME_FIELD_MAP, f"Missing mapping for '{f}'"

    def test_pump_test_flow_not_in_map(self) -> None:
        assert "pump_test_flow" not in IGME_FIELD_MAP.values()


class TestIGMESourceMetadata:
    """Verify source metadata preservation (v2 — promoted fields also in metadata)."""

    def test_aquifer_system_promoted(self, tmp_raw_dir: Path) -> None:
        """aquifer_system is now promoted to explicit field."""
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        assert by_id["207-4-0001"].aquifer_system == "Acuífero aislado"

    def test_watershed_promoted(self, tmp_raw_dir: Path) -> None:
        """watershed (cuenca_hidrografica) is now promoted to explicit field."""
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        for r in records:
            assert r.watershed == "NORTE"

    def test_null_fields_preserved(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        # toponimia stays in source_metadata
        assert by_id["207-4-0002"].source_metadata["toponimia"] is None

    def test_null_distinct_from_zero(self, tmp_raw_dir: Path) -> None:
        ingester = IGMEIngester()
        records = ingester.normalize(tmp_raw_dir)
        by_id = {r.igme_id: r for r in records}
        # Manantial → null depth
        assert by_id["207-4-0002"].total_depth_m is None
        # Sondeo with 28m depth → non-null
        assert by_id["207-4-0001"].total_depth_m == 28.0
