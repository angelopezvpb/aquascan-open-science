"""IGME — Inventario de Puntos de Agua ingestion.

Integrates the IGME public ArcGIS REST service as the first public
data source for AquaScan.

Features:
- Paginated ArcGIS REST queries with stable ordering (OBJECTID ASC)
- Total record count query (returnCountOnly=true)
- RAW response preservation with SHA-256 checksums
- Corruption detection (repeated pages, stalled offset, duplicate IDs,
  premature empty pages, incomplete downloads)
- Field name normalization (Spanish → English/canonical)
- Dedicated IGMEWaterPointRecord model (semantically separate from CampaignRecord)
- Corrected flow semantics: Caudal_Referencia_L_s → reference_flow_l_s (NOT pump_test_flow)
- Corrected date semantics: FECHA_OBRA parsed as ArcGIS epoch milliseconds
- Metadata promotion: watershed, aquifer_system, province, municipality, point_nature,
  drilling_method, detail_url, water_use, organization, pump_type
- QA flag computation
- Duplicate detection
- Export to Parquet, CSV, JSONL

Respects HTTP limits, implements retries, and logs all operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from aquascan.data.models import IGMEWaterPointRecord, QAFlags
from aquascan.data.normalization import (
    _safe_float,
    _safe_int,
    _safe_str,
    normalize_field_name,
)
from aquascan.data.provenance import compute_sha256, create_provenance_record


class DownloadCorruptionError(RuntimeError):
    """Raised when the download detects data corruption or inconsistency."""

    pass


logger = logging.getLogger(__name__)


# IGME ArcGIS REST service endpoint for water points inventory
IGME_WATER_POINTS_URL = (
    "https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_PuntosAgua/MapServer/0/query"
)

# Field name mapping (normalized field name → IGMEWaterPointRecord canonical).
# Keys are the result of normalize_field_name() applied to the raw IGME field.
# Fields without an explicit mapping are preserved with their normalized name
# and stored in source_metadata.
IGME_FIELD_MAP: dict[str, str] = {
    # Primary identifier — the real IGME point ID
    "id": "igme_code",
    # Location
    "cota_msnm": "elevation",
    # Depth
    "prof_m": "total_depth",
    # Flow — CORRECTED: reference flow, NOT pump test flow
    "caudal_referencia_l_s": "reference_flow",
    # Hydrogeology — promoted to explicit fields
    "unidad_hidrogeologica": "hydrogeological_unit",
    "sistema_acuifero": "aquifer_system",
    "cuenca_hidrografica": "watershed",
    # Point classification — promoted to explicit field
    "naturaleza": "point_nature",
    "point_type": "point_nature",  # normalize_field_name converts "naturaleza" → "point_type"
    # Administration — promoted to explicit fields
    "provincia": "province",
    "municipio": "municipality",
    # Construction date — parsed as epoch milliseconds
    "fecha_obra": "fecha_obra_raw",
    # URL — promoted to explicit field
    "urldetalle": "detail_url",
    # ESRI technical fields
    "objectid": "igme_object_id",
    "esri_oid": "esri_oid",
    # Map sheet / quadrant
    "hoja": "hoja",
    "octante": "octante",
    "punto": "punto",
    # Drilling / equipment — promoted to explicit fields
    "metodo_perforacion": "drilling_method",
    "motobomba": "pump_type",
    "usos_agua": "water_use",
    "organismo": "organization",
    # Toponymy
    "toponimia": "toponimia",
    # UTM grid
    "huso": "huso",
    "sector": "sector",
    "escala": "escala",
    # Projected coords (UTM)
    "expr1": "expr1",
    "expr2": "expr2",
}

# Sentinel date values that indicate partial/corrupt dates
_SENTINEL_EPOCH_MS: set[int] = {
    2678400000,  # exactly 31 days (1970-01-31) — likely partial date
    0,  # epoch zero
}

# Bounding box: broad Spain (including Canary Islands, Balearics)
SPAIN_BBOX = {
    "lon_min": -18.5,  # El Hierro
    "lon_max": 5.0,  # Mediterranean islands
    "lat_min": 27.5,  # Canary Islands
    "lat_max": 44.0,  # Northern Spain
}

# Plausible elevation range for Spain (m)
ELEVATION_MIN_M = -100
ELEVATION_MAX_M = 4000  # Mulhacén is ~3479m; allow some margin

# Plausible depth range (m)
DEPTH_MAX_M = 3000

# Duplicate coordinate detection precision (decimal degrees)
DUPLICATE_COORD_DECIMALS = 4


def _parse_construction_date(epoch_ms: int | None) -> tuple[str | None, bool, bool]:
    """Parse FECHA_OBRA as ArcGIS epoch milliseconds.

    ArcGIS REST returns esriFieldTypeDate as milliseconds since Unix epoch.
    The previous implementation incorrectly treated these as nanoseconds.

    Returns:
        Tuple of (iso_date_string | None, is_invalid, is_suspicious)

        - iso_date_string: ISO 8601 date (YYYY-MM-DD) or None if unparseable
        - is_invalid: True if the value cannot be parsed as a valid date
        - is_suspicious: True if the date looks like a partial/sentinel value
    """
    if epoch_ms is None:
        return None, False, False

    # Zero epoch is sentinel
    if epoch_ms == 0:
        return None, True, False

    # Known sentinel values
    if epoch_ms in _SENTINEL_EPOCH_MS:
        return None, True, False

    try:
        # ArcGIS returns epoch milliseconds
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
    except (OSError, ValueError, OverflowError):
        return None, True, False

    iso_str = dt.strftime("%Y-%m-%d")

    # Check for partial-date patterns:
    # Jan 1 or Feb 1 often indicate "year only" dates
    # in IGME (the system defaults to these when only year is known)
    is_suspicious = dt.month == 1 and dt.day == 1 or dt.month == 2 and dt.day == 1

    # Pre-1900 dates are suspicious for Spanish water infrastructure
    if dt.year < 1900:
        is_suspicious = True

    # Post-2030 dates are clearly wrong
    if dt.year > 2030:
        return None, True, False

    return iso_str, False, is_suspicious


def _compute_qa_flags(
    latitude: float | None,
    longitude: float | None,
    elevation: float | None,
    total_depth: float | None,
    reference_flow: float | None,
    hydrogeological_unit: str | None,
    construction_date_valid: bool,
    construction_date_suspicious: bool,
    is_duplicate_coord: bool,
) -> QAFlags:
    """Compute QA flags for a single IGME record.

    Returns:
        QAFlags instance with appropriate flags set.
    """
    flags = QAFlags()

    # Coordinate outlier: outside broad Spain bounding box
    if latitude is not None and longitude is not None:
        if (
            longitude < SPAIN_BBOX["lon_min"]
            or longitude > SPAIN_BBOX["lon_max"]
            or latitude < SPAIN_BBOX["lat_min"]
            or latitude > SPAIN_BBOX["lat_max"]
        ):
            flags.coordinate_outlier = True

    # Elevation outlier
    if elevation is not None and (elevation < ELEVATION_MIN_M or elevation > ELEVATION_MAX_M):
        flags.elevation_outlier = True

    # Depth outlier
    if total_depth is not None and total_depth > DEPTH_MAX_M:
        flags.depth_outlier = True

    # Negative reference flow
    if reference_flow is not None and reference_flow < 0:
        flags.negative_reference_flow = True

    # Duplicate coordinate
    if is_duplicate_coord:
        flags.duplicate_coordinate = True

    # Missing hydrogeological unit
    if not hydrogeological_unit:
        flags.missing_hydrogeological_unit = True

    # Invalid/suspicious construction date
    if construction_date_valid:
        flags.invalid_construction_date = True
    if construction_date_suspicious:
        flags.suspicious_construction_date = True

    return flags


class IGMEIngester:
    """Client for ingesting IGME water points inventory.

    Implements paginated ArcGIS REST queries with RAW data preservation,
    normalization, and export.
    """

    def __init__(
        self,
        output_dir: Path = Path("data/raw/igme"),
        page_size: int = 1000,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        requests_per_second: float = 2.0,
    ):
        """
        Args:
            output_dir: Directory to store raw JSON responses.
            page_size: Records per page (max ArcGIS default is 1000).
            max_retries: Maximum retry attempts per request.
            retry_delay_seconds: Backoff delay between retries.
            requests_per_second: Rate limiting for requests.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.page_size = min(page_size, 1000)
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._total_records = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def _query_count(self, client: httpx.Client) -> int:
        """Query the total record count from the IGME service.

        Returns:
            Total number of records available.
        """
        params: dict[str, str] = {
            "f": "json",
            "where": "1=1",
            "returnCountOnly": "true",
        }
        self._rate_limit()
        response = client.get(IGME_WATER_POINTS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        count: int = data.get("count", 0)
        return count

    def _build_query_params(
        self,
        offset: int,
        out_fields: str = "*",
        geometry: str | None = None,
    ) -> dict[str, str]:
        """Build ArcGIS REST query parameters.

        Args:
            offset: Result offset for pagination.
            out_fields: Fields to return ('*' for all).
            geometry: Optional geometry filter.

        Returns:
            Query parameter dictionary.
        """
        params: dict[str, str] = {
            "f": "json",
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": "true",
            "resultOffset": str(offset),
            "resultRecordCount": str(self.page_size),
            "orderByFields": "OBJECTID ASC",
        }
        if geometry:
            params["geometry"] = geometry
        return params

    def _fetch_page(self, client: httpx.Client, params: dict[str, str]) -> dict[str, Any]:
        """Fetch a single page with retry logic.

        Returns:
            Parsed JSON response.
        """
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                response = client.get(IGME_WATER_POINTS_URL, params=params)
                response.raise_for_status()
                data: dict[str, Any] = response.json()

                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise RuntimeError(f"ArcGIS API error: {error_msg}")

                return data
            except (httpx.HTTPError, RuntimeError) as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(self.retry_delay * (2**attempt))

        # Unreachable but satisfies type checker
        raise RuntimeError("Unexpected: retry loop exhausted")

    def download(
        self,
        max_pages: int | None = None,
        max_records: int | None = None,
    ) -> dict[str, Any]:
        """Download IGME water points inventory.

        Pagination strategy:
        1. Query returnCountOnly=true to get expected_total.
        2. Fetch pages with orderByFields=OBJECTID ASC for stable ordering.
        3. Increment offset by actual features received (not page_size).
        4. Continue until downloaded >= expected_total OR features empty.
        5. Respect max_pages as optional testing limit.
        6. Detect corruption: repeated pages, stalled offset, duplicate
           OBJECTIDs, premature empty pages, incomplete downloads.

        Args:
            max_pages: Maximum pages to download (None = unlimited).
            max_records: Maximum records to download (None = unlimited).

        Returns:
            Summary dict with 'records', 'pages', 'size_mb', 'checksum',
            'expected_total', 'complete'.
        """
        total_records = 0
        page_count = 0
        total_bytes = 0
        all_hashes: list[str] = []
        seen_object_ids: set[int] = set()
        last_page_last_oid: int | None = None
        corruption_errors: list[str] = []

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            # Step 1: Query total record count
            expected_total = self._query_count(client)
            logger.info(f"Total records available: {expected_total}")

            metadata: dict[str, Any] = {
                "source": "IGME_Inventario_Puntos_Agua",
                "source_url": IGME_WATER_POINTS_URL,
                "download_start": datetime.now(UTC).isoformat(),
                "page_size": self.page_size,
                "expected_total_records": expected_total,
                "max_pages": max_pages,
            }

            offset = 0

            while True:
                # --- Stop conditions ---
                if max_pages is not None and page_count >= max_pages:
                    logger.info(f"Reached max_pages limit ({max_pages})")
                    break
                if max_records is not None and total_records >= max_records:
                    logger.info(f"Reached max_records limit ({max_records})")
                    break
                if total_records >= expected_total:
                    logger.info(f"Downloaded all {expected_total} records")
                    break

                page_count += 1
                params = self._build_query_params(offset)
                data = self._fetch_page(client, params)

                features = data.get("features", [])
                if not features:
                    # Premature empty page — report as corruption
                    msg = (
                        f"Empty page {page_count} at offset {offset} "
                        f"before expected total {expected_total} "
                        f"(downloaded {total_records})"
                    )
                    logger.warning(msg)
                    corruption_errors.append(msg)
                    break

                # --- Corruption: check first/last OBJECTID ---
                features_sorted = sorted(
                    features,
                    key=lambda f: f["attributes"].get("OBJECTID", 0),
                )
                first_oid = features_sorted[0]["attributes"].get("OBJECTID")
                last_oid = features_sorted[-1]["attributes"].get("OBJECTID")

                # Repeated page detection: same first OBJECTID as a previous page
                if first_oid in seen_object_ids:
                    msg = (
                        f"Corruption: first OBJECTID {first_oid} on page {page_count} "
                        f"already seen on a previous page. Repeated page detected."
                    )
                    logger.error(msg)
                    raise DownloadCorruptionError(msg)

                # Same last OBJECTID as previous page (offset not advancing)
                if last_page_last_oid is not None and last_oid == last_page_last_oid:
                    msg = (
                        f"Corruption: last OBJECTID {last_oid} on page {page_count} "
                        f"is the same as previous page. Offset may not be advancing."
                    )
                    logger.error(msg)
                    raise DownloadCorruptionError(msg)

                last_page_last_oid = last_oid

                # --- Corruption: duplicate OBJECTIDs within this page ---
                page_oids = [f["attributes"].get("OBJECTID") for f in features]
                oid_counts: dict[int, int] = {}
                for oid in page_oids:
                    oid_counts[oid] = oid_counts.get(oid, 0) + 1
                dupes_in_page = {oid: c for oid, c in oid_counts.items() if c > 1}
                if dupes_in_page:
                    msg = (
                        f"Corruption: duplicate OBJECTIDs within page {page_count}: {dupes_in_page}"
                    )
                    logger.error(msg)
                    raise DownloadCorruptionError(msg)

                # Track seen OBJECTIDs for cross-page dedup detection
                for oid in page_oids:
                    seen_object_ids.add(oid)

                # --- Save raw page ---
                raw_bytes = json.dumps(data, indent=2).encode("utf-8")
                page_hash = hashlib.sha256(raw_bytes).hexdigest()
                page_file = self.output_dir / f"page_{page_count:05d}_{page_hash[:12]}.json"
                page_file.write_bytes(raw_bytes)

                total_bytes += len(raw_bytes)
                all_hashes.append(page_hash)
                total_records += len(features)

                logger.info(
                    f"Page {page_count}: {len(features)} records "
                    f"(total: {total_records}/{expected_total}), "
                    f"OBJECTID: {first_oid}→{last_oid}, "
                    f"hash: {page_hash[:12]}"
                )

                # --- Check if more pages exist ---
                # exceededTransferLimit is a TOP-LEVEL key in ArcGIS REST responses
                has_more = data.get("exceededTransferLimit", False)
                if not has_more:
                    break

                # Advance offset by actual features received
                offset += len(features)

        # --- Determine download completeness ---
        complete = total_records >= expected_total and not corruption_errors and max_pages is None
        if max_pages is not None:
            complete = False  # Explicit test limit → not complete
        if corruption_errors:
            complete = False

        # --- Write metadata ---
        metadata.update(
            {
                "download_end": datetime.now(UTC).isoformat(),
                "downloaded_records": total_records,
                "pages_downloaded": page_count,
                "total_bytes": total_bytes,
                "page_hashes": all_hashes,
                "complete": complete,
                "corruption_errors": corruption_errors if corruption_errors else None,
                "first_object_id": min(seen_object_ids) if seen_object_ids else None,
                "last_object_id": max(seen_object_ids) if seen_object_ids else None,
            }
        )
        metafile = self.output_dir / "metadata.json"
        metafile.write_text(json.dumps(metadata, indent=2))

        if corruption_errors:
            logger.warning(f"Download completed with {len(corruption_errors)} corruption warnings")

        return {
            "records": total_records,
            "pages": page_count,
            "size_mb": round(total_bytes / (1024 * 1024), 2),
            "checksum": hashlib.sha256("".join(all_hashes).encode()).hexdigest(),
            "expected_total": expected_total,
            "complete": complete,
        }

    def normalize(self, raw_dir: Path) -> list[IGMEWaterPointRecord]:
        """Normalize raw IGME JSON pages into IGMEWaterPointRecord instances.

        One RAW feature → one IGMEWaterPointRecord.

        Key semantic changes from the previous implementation:
        - Caudal_Referencia_L_s → reference_flow_l_s (NOT pump_test_flow)
        - FECHA_OBRA parsed as ArcGIS epoch milliseconds (not nanoseconds)
        - Metadata fields promoted to explicit record fields
        - QA flags computed for every record
        - Returns IGMEWaterPointRecord, not CampaignRecord

        Args:
            raw_dir: Directory containing raw IGME JSON page files.

        Returns:
            List of IGMEWaterPointRecord instances with provenance, source metadata, and QA flags.
        """
        records: list[IGMEWaterPointRecord] = []
        seen_ids: set[str] = set()
        raw_feature_count = 0
        duplicate_count = 0

        # Track coordinate pairs for duplicate detection
        coord_counts: dict[tuple[int, int], int] = {}

        json_files = sorted(raw_dir.glob("page_*.json"))
        logger.info(f"Found {len(json_files)} raw page files in {raw_dir}")

        # First pass: count coordinate occurrences
        for jf in json_files:
            data = json.loads(jf.read_bytes().decode("utf-8"))
            for feature in data.get("features", []):
                geometry = feature.get("geometry", {})
                lat = geometry.get("y")
                lon = geometry.get("x")
                if lat is not None and lon is not None:
                    key = (
                        round(float(lat), DUPLICATE_COORD_DECIMALS),
                        round(float(lon), DUPLICATE_COORD_DECIMALS),
                    )
                    coord_counts[key] = coord_counts.get(key, 0) + 1

        # Second pass: normalize each feature
        for jf in json_files:
            raw_bytes = jf.read_bytes()
            page_hash = compute_sha256(raw_bytes)
            data = json.loads(raw_bytes.decode("utf-8"))
            features = data.get("features", [])

            for feat_idx, feature in enumerate(features):
                raw_feature_count += 1
                attrs = feature.get("attributes", {})
                geometry = feature.get("geometry", {})

                # --- Normalize field names ---
                normalized: dict[str, Any] = {}
                for key, value in attrs.items():
                    nk = normalize_field_name(key)
                    mk = IGME_FIELD_MAP.get(nk, nk)
                    normalized[mk] = value

                # --- Extract coordinates from ArcGIS geometry {x, y} ---
                longitude = _safe_float(geometry.get("x"))
                latitude = _safe_float(geometry.get("y"))

                # --- Build IGME identity ---
                igme_id = _safe_str(normalized.get("igme_code"))
                if not igme_id:
                    # Fallback: use OBJECTID
                    obj_id = normalized.get("igme_object_id")
                    igme_id = str(obj_id) if obj_id is not None else f"unknown-{feat_idx}"

                # --- Deduplicate by IGME Id ---
                if igme_id in seen_ids:
                    duplicate_count += 1
                    logger.debug(f"Duplicate IGME Id: {igme_id} in {jf.name}")
                    continue
                seen_ids.add(igme_id)

                # --- Promoted metadata fields ---
                watershed = _safe_str(normalized.get("watershed"))
                hydrogeological_unit = _safe_str(normalized.get("hydrogeological_unit"))
                aquifer_system = _safe_str(normalized.get("aquifer_system"))
                province = _safe_str(normalized.get("province"))
                municipality = _safe_str(normalized.get("municipality"))
                point_nature = _safe_str(normalized.get("point_nature"))
                drilling_method = _safe_str(normalized.get("drilling_method"))
                pump_type = _safe_str(normalized.get("pump_type"))
                water_use = _safe_str(normalized.get("water_use"))
                organization = _safe_str(normalized.get("organization"))
                detail_url = _safe_str(normalized.get("detail_url"))

                # --- Numeric fields ---
                elevation = _safe_float(normalized.get("elevation"))
                total_depth = _safe_float(normalized.get("total_depth"))
                # Caudal_Referencia_L_s → reference_flow_l_s (semantically correct)
                reference_flow = _safe_float(normalized.get("reference_flow"))
                igme_object_id = _safe_int(normalized.get("igme_object_id"))

                # --- Construction date: FECHA_OBRA as epoch milliseconds ---
                fecha_obra_raw = normalized.get("fecha_obra_raw")
                fecha_obra_int: int | None = None
                if fecha_obra_raw is not None:
                    try:
                        fecha_obra_int = int(fecha_obra_raw)
                    except (ValueError, TypeError):
                        fecha_obra_int = None

                construction_date, date_invalid, date_suspicious = _parse_construction_date(
                    fecha_obra_int
                )

                # --- Duplicate coordinate detection ---
                is_dup_coord = False
                if latitude is not None and longitude is not None:
                    key = (
                        round(latitude, DUPLICATE_COORD_DECIMALS),
                        round(longitude, DUPLICATE_COORD_DECIMALS),
                    )
                    is_dup_coord = coord_counts.get(key, 0) > 1

                # --- QA flags ---
                qa_flags = _compute_qa_flags(
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    total_depth=total_depth,
                    reference_flow=reference_flow,
                    hydrogeological_unit=hydrogeological_unit,
                    construction_date_valid=date_invalid,
                    construction_date_suspicious=date_suspicious,
                    is_duplicate_coord=is_dup_coord,
                )

                # --- Build source metadata for all remaining IGME fields ---
                source_metadata: dict[str, Any] = {
                    "hoja": normalized.get("hoja"),
                    "octante": normalized.get("octante"),
                    "punto": _safe_str(normalized.get("punto")),
                    "toponimia": _safe_str(normalized.get("toponimia")),
                    "escala": _safe_str(normalized.get("escala")),
                    "huso": normalized.get("huso"),
                    "sector": _safe_str(normalized.get("sector")),
                    "expr1": normalized.get("expr1"),
                    "expr2": normalized.get("expr2"),
                    "esri_oid": normalized.get("esri_oid"),
                    "fecha_obra_raw": fecha_obra_raw,
                    "source_file": jf.name,
                    "source_page_hash": page_hash,
                    "source_feature_index": feat_idx,
                    "source_uri": IGME_WATER_POINTS_URL,
                }

                # --- Build provenance ---
                provenance = create_provenance_record(
                    processing_step="igme_normalize_v2",
                    source_uri=IGME_WATER_POINTS_URL,
                    source_hash=page_hash,
                    parent_artifact=jf.name,
                    processing_version="0.2.0",
                )
                # Set source_record_id to track the IGME Id
                provenance.source_record_id = igme_id

                # --- Build the canonical IGMEWaterPointRecord ---
                record = IGMEWaterPointRecord(
                    igme_id=igme_id,
                    igme_object_id=igme_object_id,
                    latitude=latitude,
                    longitude=longitude,
                    elevation_m=elevation,
                    watershed=watershed,
                    hydrogeological_unit=hydrogeological_unit,
                    aquifer_system=aquifer_system,
                    province=province,
                    municipality=municipality,
                    point_nature=point_nature,
                    total_depth_m=total_depth,
                    reference_flow_l_s=reference_flow,
                    drilling_method=drilling_method,
                    construction_date=construction_date,
                    pump_type=pump_type,
                    water_use=water_use,
                    organization=organization,
                    detail_url=detail_url,
                    source_metadata=source_metadata,
                    provenance=provenance,
                    qa_flags=qa_flags,
                )

                records.append(record)

        logger.info(
            f"Normalized {len(records)} unique records "
            f"from {raw_feature_count} raw features "
            f"({duplicate_count} duplicates removed)"
        )
        return records

    def to_dataframe(self, records: list[IGMEWaterPointRecord]) -> pd.DataFrame:
        """Convert IGMEWaterPointRecord instances to a flat DataFrame.

        Args:
            records: List of IGMEWaterPointRecord instances.

        Returns:
            DataFrame with IGMEWaterPointRecord fields flattened.
        """
        if not records:
            return pd.DataFrame()

        # Flatten each record: model_dump + flatten qa_flags
        rows: list[dict[str, Any]] = []
        for r in records:
            row = r.model_dump()
            # Flatten QA flags
            qa = row.pop("qa_flags", {})
            for flag_name, flag_value in qa.items():
                row[f"qa_{flag_name}"] = flag_value
            row["qa_flag_count"] = r.qa_flags.flag_count
            # Flatten provenance for readability
            prov = row.pop("provenance", {})
            row["prov_source_hash"] = prov.get("source_hash")
            row["prov_processing_step"] = prov.get("processing_step")
            row["prov_source_uri"] = prov.get("source_uri")
            row["prov_parent_artifact"] = prov.get("parent_artifact")
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"DataFrame: {len(df)} rows, {len(df.columns)} columns")
        return df

    def export(
        self,
        records: list[IGMEWaterPointRecord],
        output_path: Path,
        format: str = "parquet",
    ) -> Path:
        """Export records to file.

        Args:
            records: List of IGMEWaterPointRecord instances.
            output_path: Output file path.
            format: Output format ('parquet', 'csv', 'jsonl').

        Returns:
            The output path.
        """
        df = self.to_dataframe(records)

        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "jsonl":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path
