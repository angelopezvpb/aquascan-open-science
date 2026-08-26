"""AquaScan canonical data models.

All models are defined with Pydantic v2 for runtime validation and JSON Schema export.

Key principles:
- Every field has an explicit type, documented units, and description.
- Fields are nullable when the value may genuinely be unknown.
- null != 0 — never convert unknown values to zero.
- Provenance is tracked for every record.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DatasetLevel(StrEnum):
    """Data processing level."""

    L0 = "L0"  # RAW — immutable original data
    L1 = "L1"  # CALIBRATED / QC — normalized, quality controlled
    L2 = "L2"  # GEOPHYSICS — derived geophysical products
    L3 = "L3"  # ML READY — scientific features
    L4 = "L4"  # GROUND TRUTH — independent borehole verification


# GroundTruthClass is now defined below with GroundTruthEvidenceRecord.
# Uses GT_A, GT_B, GT_C, GT_D for explicit naming.


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Provenance record linking every data product to its RAW origin.

    Answers the question: "From which RAW data did this value originate?"
    """

    source_record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the source record.",
    )
    source_uri: str | None = Field(
        default=None,
        description="URI of the source (URL, file path, database reference).",
    )
    source_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the RAW source artifact.",
    )
    processing_step: str = Field(
        default="unknown",
        description="Name of the processing step that produced this data.",
    )
    processing_version: str = Field(
        default="0.1.0",
        description="Version of the processing code.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of when the record was created or processed.",
    )
    parent_artifact: str | None = Field(
        default=None,
        description="Identifier of the parent artifact this was derived from.",
    )
    instrument_version: str | None = Field(
        default=None,
        description="Version or model of the instrument that collected the data.",
    )


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class QualityMetrics(BaseModel):
    """Quality control metrics for a campaign's geophysical data."""

    ert_quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ERT data quality score (0-1). Based on contact resistance, "
        "reciprocity, stacking errors, etc.",
    )
    seismic_quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Seismic data quality score (0-1). Based on first-break clarity, "
        "SNR, geophone coupling, etc.",
    )
    gnss_quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="GNSS positioning quality (0-1). Based on fix type, PDOP, "
        "baseline length, etc.",
    )


# ---------------------------------------------------------------------------
# Core Campaign Record
# ---------------------------------------------------------------------------


class CampaignRecord(BaseModel):
    """Canonical AquaScan campaign record.

    This is the central data structure that flows through all processing
    levels from L0 (raw) to L4 (ground truth). Not all fields are populated
    at every level — fields are nullable when data hasn't been collected yet.
    """

    # --- Identity ---
    campaign_id: str = Field(
        description="Unique identifier for the geophysical campaign.",
    )
    site_id: str = Field(
        description="Site identifier. Multiple campaigns may share a site.",
    )
    well_id: str | None = Field(
        default=None,
        description="Well/borehole identifier, if a well is associated.",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Version of the AquaScan schema this record conforms to.",
    )

    # --- Temporal ---
    timestamp: datetime | None = Field(
        default=None,
        description="Date and time of the geophysical survey (UTC).",
    )

    # --- Spatial ---
    latitude: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (WGS84).",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (WGS84).",
    )
    elevation: float | None = Field(
        default=None,
        description="Surface elevation in meters above sea level.",
    )

    # --- Geological context ---
    geology_domain: str | None = Field(
        default=None,
        description="Geological domain or formation name.",
    )
    hydrogeological_unit: str | None = Field(
        default=None,
        description="Hydrogeological unit name or aquifer code.",
    )

    # --- Geophysical data references ---
    ert_raw_uri: str | None = Field(
        default=None,
        description="URI to the raw ERT data file (L0).",
    )
    seismic_raw_uri: str | None = Field(
        default=None,
        description="URI to the raw seismic data file (L0).",
    )

    # --- Quality ---
    ert_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ERT quality score (0-1).",
    )
    seismic_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Seismic quality score (0-1).",
    )

    # --- Geophysical derived products (L2) ---
    ert_resistivity_mean: float | None = Field(
        default=None,
        description="Mean apparent resistivity from ERT inversion (ohm·m).",
    )
    ert_resistivity_std: float | None = Field(
        default=None,
        description="Standard deviation of apparent resistivity (ohm·m).",
    )
    seismic_vp_mean: float | None = Field(
        default=None,
        description="Mean P-wave velocity (m/s).",
    )
    seismic_vp_std: float | None = Field(
        default=None,
        description="Standard deviation of P-wave velocity (m/s).",
    )

    # --- Water depth estimates from geophysics ---
    water_depth_min: float | None = Field(
        default=None,
        description="Minimum estimated depth to water table from geophysics (m).",
    )
    water_depth_max: float | None = Field(
        default=None,
        description="Maximum estimated depth to water table from geophysics (m).",
    )

    # --- Well data / Ground Truth (L4) ---
    static_water_level: float | None = Field(
        default=None,
        description="Measured static water level depth (m below surface).",
    )
    total_depth: float | None = Field(
        default=None,
        description="Total drilled depth (m).",
    )
    pump_test_flow: float | None = Field(
        default=None,
        description="Flow rate from pump test (L/s).",
    )
    pump_test_duration: float | None = Field(
        default=None,
        description="Duration of pump test (hours).",
    )
    water_electrical_conductivity: float | None = Field(
        default=None,
        description="Water electrical conductivity (µS/cm).",
    )
    water_tds: float | None = Field(
        default=None,
        description="Total Dissolved Solids (mg/L).",
    )

    # --- Ground Truth scoring ---
    ground_truth_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ground Truth quality score (0-1, continuous).",
    )
    ground_truth_class: GroundTruthClass | None = Field(
        default=None,
        description="Ground Truth quality class (A/B/C/D).",
    )
    ground_truth_source: str | None = Field(
        default=None,
        description="Source of the ground truth data (e.g., 'borehole_log', "
        "'pump_test', 'igme_record').",
    )
    ground_truth_reasons: list[str] | None = Field(
        default=None,
        description="List of reasons explaining the ground truth score.",
    )

    # --- Candidate Priority Score (NOT the same as Ground Truth) ---
    candidate_priority_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score used to prioritize public records for investigation. "
        "This is NOT the scientific ground truth score. It uses data completeness "
        "and basic metadata to rank candidate records for follow-up.",
    )

    # --- Source metadata ---
    source_metadata: dict | None = Field(
        default=None,
        description="Source-specific metadata fields preserved from the original data."
        " For IGME, this includes fields like 'cuenca_hidrografica',"
        " 'sistema_acuifero', 'naturaleza', 'metodo_perforacion',"
        " 'url_detalle', etc.",
    )

    # --- Provenance ---
    provenance: Provenance = Field(
        default_factory=Provenance,
        description="Provenance tracking for this record.",
    )

    # --- AutoCal version (placeholder for proprietary) ---
    autocal_version: str | None = Field(
        default=None,
        description="Version of AutoCal used (placeholder for proprietary system).",
    )

    @field_validator("geology_domain", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: str | None) -> str | None:
        """Convert empty strings to None — unknown is null, not empty."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


# ---------------------------------------------------------------------------
# Evidence & Water Scores
# ---------------------------------------------------------------------------


class EvidenceScore(BaseModel):
    """Evidence Score — quality of the evidence backing a prediction.

    Separated from the Water Score intentionally. A prediction can have:

        Water Score    = 0.90  (model thinks there's water)
        Evidence Score = 0.50  (but the evidence quality is moderate)

    and still be flagged as NO CONCLUYENTE.
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence quality score (0-1).",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown by evidence component.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the evidence score.",
    )


class WaterPrediction(BaseModel):
    """Result of a water presence prediction.

    Keeps Water Score and Evidence Score separate. The `abstain` flag
    signals that the prediction should be treated as NO CONCLUYENTE.
    """

    campaign_id: str = Field(
        description="Campaign this prediction applies to.",
    )
    water_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability/score that a favorable hydrogeological target exists (0-1).",
    )
    evidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of the evidence backing the prediction (0-1).",
    )
    abstain: bool = Field(
        default=False,
        description="If True, the prediction is NO CONCLUYENTE and should "
        "not be treated as a positive or negative classification.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction (0-1).",
    )
    model_version: str = Field(
        default="0.1.0",
        description="Version of the model that produced this prediction.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this prediction was generated.",
    )
    features_used: list[str] = Field(
        default_factory=list,
        description="List of feature names used for this prediction.",
    )


# ---------------------------------------------------------------------------
# IGME Water Point Record
# ---------------------------------------------------------------------------


class QAFlags(BaseModel):
    """Quality assurance flags for IGME water point records.

    QA flags are warnings, not automatic exclusions. They identify
    records that deserve human review before use in training or analysis.
    """

    coordinate_outlier: bool = Field(
        default=False,
        description="Coordinates fall outside the expected Spain bounding box.",
    )
    elevation_outlier: bool = Field(
        default=False,
        description="Elevation is outside the plausible range for Spain (-100 to 4000 m).",
    )
    depth_outlier: bool = Field(
        default=False,
        description="Total depth exceeds plausible range (>3000 m for Spain).",
    )
    negative_reference_flow: bool = Field(
        default=False,
        description="Reference flow value is negative.",
    )
    duplicate_coordinate: bool = Field(
        default=False,
        description="Coordinates are shared with other records (possible duplicate).",
    )
    missing_hydrogeological_unit: bool = Field(
        default=False,
        description="Hydrogeological unit is missing.",
    )
    invalid_construction_date: bool = Field(
        default=False,
        description="Construction date is unparseable or contains a sentinel value.",
    )
    suspicious_construction_date: bool = Field(
        default=False,
        description="Construction date appears to be a partial date "
        "(year-only, Jan 1/Feb 1 pattern).",
    )

    @property
    def flag_count(self) -> int:
        """Count of active QA flags."""
        return sum(
            [
                self.coordinate_outlier,
                self.elevation_outlier,
                self.depth_outlier,
                self.negative_reference_flow,
                self.duplicate_coordinate,
                self.missing_hydrogeological_unit,
                self.invalid_construction_date,
                self.suspicious_construction_date,
            ]
        )

    @property
    def active_flags(self) -> list[str]:
        """List of active flag names."""
        flags: list[str] = []
        for field_name in [
            "coordinate_outlier",
            "elevation_outlier",
            "depth_outlier",
            "negative_reference_flow",
            "duplicate_coordinate",
            "missing_hydrogeological_unit",
            "invalid_construction_date",
            "suspicious_construction_date",
        ]:
            if getattr(self, field_name):
                flags.append(field_name)
        return flags


class IGMEWaterPointRecord(BaseModel):
    """Canonical IGME water point record — a historical/public source record.

    This is semantically distinct from CampaignRecord. An IGME water point
    is data from the Spanish Geological Survey's public inventory, not an
    AquaScan geophysical measurement campaign.
    """

    # --- Primary identity ---
    igme_id: str = Field(description="IGME water point identifier (e.g., '207-4-0001').")
    igme_object_id: int | None = Field(
        default=None,
        description="ArcGIS OBJECTID from the IGME feature service.",
    )

    # --- Spatial ---
    latitude: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (WGS84).",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (WGS84).",
    )
    elevation_m: float | None = Field(
        default=None,
        description="Surface elevation in meters above sea level (COTA_msnm).",
    )

    # --- Hydrogeological context ---
    watershed: str | None = Field(
        default=None,
        description="Watershed / river basin (Cuenca_Hidrografica).",
    )
    hydrogeological_unit: str | None = Field(
        default=None,
        description="Hydrogeological unit name (Unidad_Hidrogeologica).",
    )
    aquifer_system: str | None = Field(
        default=None,
        description="Aquifer system name (Sistema_Acuifero).",
    )

    # --- Administrative ---
    province: str | None = Field(
        default=None,
        description="Province (Provincia).",
    )
    municipality: str | None = Field(
        default=None,
        description="Municipality (Municipio).",
    )

    # --- Point characteristics ---
    point_nature: str | None = Field(
        default=None,
        description="Type of water point (Naturaleza): Sondeo, Manantial, Pozo, etc.",
    )
    total_depth_m: float | None = Field(
        default=None,
        description="Total drilled depth in meters (PROF_m).",
    )
    reference_flow_l_s: float | None = Field(
        default=None,
        description="Reference flow rate in L/s (Caudal_Referencia_L_s). "
        "NOT a pump test flow — this is a reference/historical value.",
    )

    # --- Construction ---
    drilling_method: str | None = Field(
        default=None,
        description="Drilling/perforation method (Metodo_Perforacion).",
    )
    construction_date: str | None = Field(
        default=None,
        description="Construction date (FECHA_OBRA) as ISO 8601 date string (YYYY-MM-DD). "
        "Null if unparseable. NOT an AquaScan campaign timestamp.",
    )

    # --- Equipment ---
    pump_type: str | None = Field(
        default=None,
        description="Pump type / extraction equipment (Motobomba).",
    )
    water_use: str | None = Field(
        default=None,
        description="Water use classification (Usos_Agua).",
    )

    # --- Organization ---
    organization: str | None = Field(
        default=None,
        description="Responsible organization (Organismo).",
    )

    # --- Reference URL ---
    detail_url: str | None = Field(
        default=None,
        description="URL to the IGME detail page (URLDetalle).",
    )

    # --- Source metadata (all additional fields preserved) ---
    source_metadata: dict = Field(
        default_factory=dict,
        description="Complete source metadata preserved from the original IGME record.",
    )

    # --- Provenance ---
    provenance: Provenance = Field(
        default_factory=Provenance,
        description="Provenance tracking for this record.",
    )

    # --- QA flags ---
    qa_flags: QAFlags = Field(
        default_factory=QAFlags,
        description="Quality assurance flags for this record.",
    )


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


class ExperimentRecord(BaseModel):
    """Metadata for a reproducible experiment."""

    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique experiment identifier.",
    )
    git_commit: str | None = Field(
        default=None,
        description="Git commit hash at experiment time.",
    )
    dataset_version: str = Field(
        description="Version identifier for the dataset used.",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="AquaScan schema version.",
    )
    configuration: dict = Field(
        default_factory=dict,
        description="Full experiment configuration (model, hyperparameters, etc.).",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    model: str = Field(
        description="Model name or identifier.",
    )
    hyperparameters: dict = Field(
        default_factory=dict,
        description="Model hyperparameters.",
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Experiment results / metrics.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this experiment was recorded.",
    )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


class SyntheticLayer(BaseModel):
    """A conceptual layer for synthetic subsurface models."""

    depth_top: float = Field(description="Depth to top of layer (m).")
    depth_bottom: float = Field(description="Depth to bottom of layer (m).")
    resistivity: float = Field(description="Electrical resistivity (ohm·m).")
    vp: float = Field(description="P-wave velocity (m/s).")
    saturated: bool = Field(
        default=False,
        description="Whether the layer is water-saturated.",
    )


class SyntheticProfile(BaseModel):
    """A synthetic 1D subsurface profile for simulation and testing."""

    profile_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this synthetic profile.",
    )
    layers: list[SyntheticLayer] = Field(
        description="Ordered list of subsurface layers (top to bottom).",
    )
    noise_level: float = Field(
        default=0.0,
        ge=0.0,
        description="Relative noise level added to simulated measurements (0-1).",
    )
    water_table_depth: float | None = Field(
        default=None,
        description="True water table depth (m), known in synthetic data.",
    )
    label: str = Field(
        default="SYNTHETIC",
        description="Marker indicating this is synthetic data.",
    )
    warnings: list[str] = Field(
        default_factory=lambda: [
            "SYNTHETIC DATA — NOT FIELD VALIDATION. "
            "Synthetic data tests code correctness but cannot validate "
            "real-world performance."
        ],
        description="Warnings about synthetic data limitations.",
    )


# ---------------------------------------------------------------------------
# Ground Truth Evidence Models
# ---------------------------------------------------------------------------


class GroundTruthClass(StrEnum):
    """Documentary Ground Truth quality classes.

    These measure the quality and completeness of documentary evidence
    available for a record, NOT the probability of finding water.

    Legacy names (A, B, C, D) are kept for backward compatibility with
    existing ground_truth/quality.py and related tests.
    New names (GT_A, GT_B, GT_C, GT_D) are recommended for clarity.
    """

    # Legacy names (backward compatible)
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    # Recommended explicit names
    GT_A = "GT-A"
    GT_B = "GT-B"
    GT_C = "GT-C"
    GT_D = "GT-D"


# ---------------------------------------------------------------------------
# Enrichment enums
# ---------------------------------------------------------------------------


class TemporalRelevanceStatus(StrEnum):
    """Classification of how recent/relevant water-level observations are.

    Thresholds (documented):
        RECENT:            last measurement ≤ 5 years ago
        HISTORICAL:        last measurement > 5 years ago, but ≥ 1 measurement
        MULTI_DECADE_SERIES: span ≥ 20 years with ≥ 5 measurements
        DATE_UNKNOWN:      no date information available
    """

    RECENT = "RECENT"
    HISTORICAL = "HISTORICAL"
    MULTI_DECADE_SERIES = "MULTI_DECADE_SERIES"
    DATE_UNKNOWN = "DATE_UNKNOWN"


class LithologyAnomaly(StrEnum):
    """Anomalies detected in lithological intervals."""

    ORDERING_VIOLATION = "ORDERING_VIOLATION"
    OVERLAPPING_INTERVALS = "OVERLAPPING_INTERVALS"
    GAP = "GAP"
    NEGATIVE_DEPTH = "NEGATIVE_DEPTH"
    BOTTOM_LT_TOP = "BOTTOM_LT_TOP"
    EXCEEDS_TOTAL_DEPTH = "EXCEEDS_TOTAL_DEPTH"


class TargetReadiness(StrEnum):
    """Per-target Ground Truth readiness classification.

    READY:     all critical fields present, evidence quality sufficient
    PARTIAL:   some evidence available but key gaps remain
    NOT_READY: insufficient evidence for this target
    """

    READY = "READY"
    PARTIAL = "PARTIAL"
    NOT_READY = "NOT_READY"


class WaterPresenceStatus(StrEnum):
    """Evidence-based water presence determination.

    PRESENT:  confirmed by actual hydrogeological observation
    ABSENT:   confirmed by negative observation (e.g., dry pumping test)
    UNKNOWN:  insufficient evidence to determine
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class EnrichmentEndpoint(StrEnum):
    """IGME bdaguas enrichment API endpoints."""

    PIEZOMETRY = "QueryAGPZ"
    LITHOLOGY = "QueryAGLI"
    PUMPING_TESTS = "QueryAGEB"
    HYDROMETRY = "QueryAGHD"
    CHEMISTRY = "QueryAGAQ"
    TECHNICAL = "QueryAGTE"
    DOCUMENTS = "QueryDocsByID"


class EvidenceType(StrEnum):
    """Classification of evidence availability."""

    STRUCTURED = "STRUCTURED"
    SEMI_STRUCTURED = "SEMI_STRUCTURED"
    DOCUMENT = "DOCUMENT"
    IMAGE_SCAN = "IMAGE_SCAN"
    FREE_TEXT = "FREE_TEXT"
    LINKED_EXTERNAL_SOURCE = "LINKED_EXTERNAL_SOURCE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ExtractionMethod(StrEnum):
    """How a Ground Truth value was extracted."""

    API_STRUCTURED = "API_STRUCTURED"
    API_NORMALIZED = "API_NORMALIZED"
    DOCUMENT_PARSED = "DOCUMENT_PARSED"
    OCR_EXTRACTED = "OCR_EXTRACTED"
    MANUAL_ENTRY = "MANUAL_ENTRY"
    DERIVED = "DERIVED"


class MeasurementRecord(BaseModel):
    """A single measurement point within a Ground Truth evidence record."""

    measured_value: float | None = Field(
        default=None,
        description="The measured value (may be null if measurement exists but value unknown).",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement (e.g., 'm', 'L/s', 'µS/cm', 'mg/L').",
    )
    measurement_type: str = Field(
        description="Type of measurement (e.g., 'static_water_level', 'pump_test_flow').",
    )
    depth_top_m: float | None = Field(
        default=None,
        description="Top of depth interval for this measurement (m).",
    )
    depth_bottom_m: float | None = Field(
        default=None,
        description="Bottom of depth interval for this measurement (m).",
    )
    measurement_date: str | None = Field(
        default=None,
        description="Date of measurement (ISO 8601).",
    )
    source_endpoint: str = Field(
        description="IGME API endpoint this value was extracted from.",
    )
    source_field: str = Field(
        description="Specific JSON field the value came from.",
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.API_STRUCTURED,
        description="Method used to extract this value.",
    )
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction correctness (0-1).",
    )
    raw_evidence: str | None = Field(
        default=None,
        description="Raw source text/value before normalization.",
    )


class GroundTruthEvidenceRecord(BaseModel):
    """A single Ground Truth evidence record for an IGME water point.

    This model captures all documentary evidence discovered from IGME detail
    sources (API enrichment endpoints). Every value traces back to its source.

    This is NOT the same as Candidate Priority Score:
    - Candidate Score: "how useful is this record for enrichment?"
    - Ground Truth Evidence: "what real evidence does this record contain?"
    """

    igme_id: str = Field(description="IGME point identifier (e.g., '3530-5-0174').")
    detail_url: str | None = Field(
        default=None,
        description="URL to the IGME detail page (may be deprecated — API is primary source).",
    )
    retrieval_timestamp: str = Field(
        description="ISO 8601 timestamp of API data retrieval.",
    )

    # ── Lithology ──
    has_lithology: bool = Field(default=False, description="Lithological log available.")
    lithology_intervals: list[MeasurementRecord] = Field(
        default_factory=list,
        description="Lithological intervals with depth, rock type, age, aquifer flag.",
    )

    # ── Water levels ──
    has_piezometry: bool = Field(default=False, description="Water level measurements available.")
    piezometry_measurements: list[MeasurementRecord] = Field(
        default_factory=list,
        description="Static water level measurements over time.",
    )

    # ── Pumping tests ──
    has_pumping_tests: bool = Field(default=False, description="Pumping test records available.")
    pumping_test_records: list[MeasurementRecord] = Field(
        default_factory=list,
        description="Pumping test results (flow, drawdown, duration, transmissivity).",
    )

    # ── Hydrochemistry ──
    has_chemistry: bool = Field(default=False, description="Chemical analysis available.")
    chemistry_records: list[MeasurementRecord] = Field(
        default_factory=list,
        description="Chemical analysis results (major ions, conductivity, pH).",
    )

    # ── Hydrometry ──
    has_hydrometry: bool = Field(default=False, description="Flow measurements available.")
    hydrometry_records: list[MeasurementRecord] = Field(
        default_factory=list,
        description="Flow measurement records.",
    )

    # ── Technical construction ──
    has_technical_data: bool = Field(default=False, description="Borehole construction details.")
    technical_notes: str | None = Field(
        default=None,
        description="Any technical construction data (free text, often null per discovery).",
    )

    # ── Documents ──
    has_documents: bool = Field(default=False, description="Downloadable documents attached.")
    document_references: list[str] = Field(
        default_factory=list,
        description="List of document references (empty for all 20 sampled).",
    )

    # ── Classification ──
    provisional_gt_class: GroundTruthClass | None = Field(
        default=None,
        description="Provisional Ground Truth class based on documentary evidence only.",
    )
    gt_class_rationale: str | None = Field(
        default=None,
        description="Human-readable explanation of why this class was assigned.",
    )
    classification_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the assigned GT class (0-1).",
    )

    # ── Conflicts / missingness ──
    conflicts: list[str] = Field(
        default_factory=list,
        description="Any conflicts detected between sources.",
    )
    known_missing: list[str] = Field(
        default_factory=list,
        description="Important fields known to be missing for this record.",
    )

    # ── Provenance ──
    source_api_version: str = Field(
        default="bdaguas_v2",
        description="Version of the IGME bdaguas API used.",
    )
    extraction_version: str = Field(
        default="v0.1",
        description="Version of the extraction logic.",
    )

    # ── Critical semantic separation ──
    # reference_flow_l_s is NOT pump_test_flow
    # total_depth_m is NOT water depth
    # Sondeo/Pozo does NOT mean productive well
    # reference_flow_l_s = 0 does NOT mean dry

    @field_validator("provisional_gt_class", mode="before")
    @classmethod
    def coerce_gt_class(cls, v: object) -> str | None:
        """Accept both enum values and string representations."""
        if v is None:
            return None
        return str(v)


# ---------------------------------------------------------------------------
# Controlled Ground Truth Enrichment models
# ---------------------------------------------------------------------------


class WaterLevelObservation(BaseModel):
    """A single piezometric water-level observation.

    CRITICAL semantic separation:
    - TARGET_DEPTH_TO_WATER: Nivel — vertical depth below ground reference (m).
    - TARGET_HYDRAULIC_HEAD_ELEVATION: Valor — water elevation/head in the
      source vertical reference (m).

    These are NOT interchangeable. The relation `Cota - Nivel = Valor` is used
    only as an internal consistency check, never to collapse the two targets.
    Original measured fields are preserved verbatim.
    """

    igme_id: str = Field(description="IGME point identifier.")
    observation_date: str | None = Field(
        default=None, description="Observation date (ISO 8601, from Fecha)."
    )
    depth_to_water_m: float | None = Field(
        default=None,
        description="TARGET_DEPTH_TO_WATER: vertical depth below ground reference "
        "reported by IGME Nivel (m).",
    )
    hydraulic_head_elevation_m: float | None = Field(
        default=None,
        description="TARGET_HYDRAULIC_HEAD_ELEVATION: water elevation/head "
        "reported by IGME Valor (m).",
    )
    ground_elevation_m: float | None = Field(
        default=None, description="Cota — ground elevation in source vertical reference (m)."
    )
    source: str = Field(default="QueryAGPZ", description="Source endpoint.")
    source_record_id: str = Field(description="Provenance link to source record (IGME Id).")
    source_hash: str | None = Field(default=None, description="SHA-256 of the raw response.")
    consistency_residual_m: float | None = Field(
        default=None,
        description="Valor - (Cota - Nivel). Internal consistency check only.",
    )
    artesian: bool | None = Field(
        default=None, description="Whether the point is artesian (Surgencia)."
    )
    aquifer_designation: str | None = Field(
        default=None, description="Aquifer designation (Acuifero field)."
    )
    raw: dict = Field(default_factory=dict, description="Original measured fields preserved.")


class LithologyInterval(BaseModel):
    """A single lithological interval.

    The `aquifer_flagged_lithology_interval` boolean indicates a geological
    classification of aquifer material — NOT a measured water entry.
    """

    igme_id: str = Field(description="IGME point identifier.")
    top_depth_m: float | None = Field(default=None, description="Top depth (ProfTecho, m).")
    bottom_depth_m: float | None = Field(default=None, description="Bottom depth (ProfMuro, m).")
    lithology: str | None = Field(default=None, description="Lithology description (Litologia).")
    geological_age: str | None = Field(default=None, description="Geological age (Edad).")
    aquifer_flagged_lithology_interval: bool = Field(
        default=False,
        description="Aquifer-flagged lithology interval (Acuifero). NOT a measured water entry.",
    )
    order_index: int = Field(default=0, description="Original ordering index (Numorden).")
    source: str = Field(default="QueryAGLI", description="Source endpoint.")
    source_hash: str | None = Field(default=None, description="SHA-256 of the raw response.")
    raw: dict = Field(default_factory=dict, description="Original measured fields preserved.")


class PumpingTestRecord(BaseModel):
    """A single pumping test record (QueryAGEB)."""

    igme_id: str = Field(description="IGME point identifier.")
    test_date: str | None = Field(default=None, description="Test date (ISO 8601, from Fecha).")
    flow_l_s: float | None = Field(
        default=None, description="Pumping-test discharge rate (Caudal, L/s)."
    )
    drawdown_m: float | None = Field(default=None, description="Drawdown (DepresionM, m).")
    duration_minutes: float | None = Field(
        default=None, description="Duration (Horasdur*60 + Mindur, minutes)."
    )
    specific_capacity_l_s_m: float | None = Field(
        default=None,
        description=(
            "DERIVED specific capacity = flow / drawdown. Only when flow>=0 and drawdown>0."
        ),
    )
    specific_capacity_is_derived: bool = Field(
        default=False, description="Whether specific capacity was derived (always marked DERIVED)."
    )
    transmissivity_m2_day: float | None = Field(
        default=None, description="Transmissivity (Transmisividad, m²/day)."
    )
    storativity: float | None = Field(default=None, description="Storativity (Coefalmace).")
    source: str = Field(default="QueryAGEB", description="Source endpoint.")
    source_hash: str | None = Field(default=None, description="SHA-256 of the raw response.")
    raw: dict = Field(default_factory=dict, description="Original measured fields preserved.")


class TemporalSummary(BaseModel):
    """Derived temporal summary for a well's water-level observations.

    These are DERIVED ARTIFACTS ONLY — they do not replace the raw time series.
    """

    num_measurements: int = Field(default=0, description="Number of water-level observations.")
    first_measurement_date: str | None = Field(default=None, description="First observation date.")
    last_measurement_date: str | None = Field(default=None, description="Last observation date.")
    temporal_span_days: float | None = Field(
        default=None, description="Days between first and last observation."
    )
    min_depth_to_water_m: float | None = Field(default=None, description="Minimum depth to water.")
    max_depth_to_water_m: float | None = Field(default=None, description="Maximum depth to water.")
    median_depth_to_water_m: float | None = Field(
        default=None, description="Median depth to water."
    )
    temporal_relevance_status: TemporalRelevanceStatus = Field(
        default=TemporalRelevanceStatus.DATE_UNKNOWN,
        description="Temporal relevance classification.",
    )


class LithologyValidationResult(BaseModel):
    """Result of lithological interval validation (step 7)."""

    anomalies: list[LithologyAnomaly] = Field(
        default_factory=list, description="Anomalies detected (flagged, not repaired)."
    )
    is_valid: bool = Field(default=True, description="Whether no anomalies were detected.")


class TargetReadinessAssessment(BaseModel):
    """Documentary readiness for a single AquaScan target.

    Candidate Priority Score MUST NOT influence these classes.
    """

    target: str = Field(description="Target identifier (e.g., TARGET_DEPTH_TO_WATER).")
    readiness: TargetReadiness = Field(description="READY / PARTIAL / NOT_READY.")
    evidence_available: bool = Field(
        default=False, description="Whether any evidence is available."
    )
    evidence_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Evidence quality score (0-1)."
    )
    documentary_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Documentary confidence (0-1)."
    )
    temporal_relevance: TemporalRelevanceStatus | None = Field(
        default=None, description="Temporal relevance where applicable."
    )
    missing_critical_fields: list[str] = Field(
        default_factory=list, description="Critical fields missing for this target."
    )
    reasons: list[str] = Field(default_factory=list, description="Deterministic rule explanations.")


class WaterPresenceDetermination(BaseModel):
    """Evidence-based water presence determination (step 10).

    MUST NOT be derived from point nature, reference flow, or aquifer flag alone.
    """

    status: WaterPresenceStatus = Field(default=WaterPresenceStatus.UNKNOWN)
    evidence_basis: list[str] = Field(
        default_factory=list,
        description="Actual hydrogeological observations supporting the status.",
    )


class NormalizedGroundTruthRecord(BaseModel):
    """One normalized Ground Truth evidence record per IGME point.

    Maintains independent evidence dimensions and preserves raw observations as
    repeated/time-indexed measurements. Time series are never summarized away.
    """

    igme_id: str = Field(description="IGME point identifier.")
    domain: str = Field(description="Phase 0 domain.")
    candidate_tier: str | None = Field(
        default=None,
        description="Candidate P1/P2/P3/P4 (for reporting only).",
    )
    candidate_priority_score: float | None = Field(
        default=None,
        description="Candidate score (for reporting only).",
    )
    detail_url: str | None = Field(default=None, description="IGME detail page URL.")

    # ── Evidence dimensions ──
    geological_evidence: list[LithologyInterval] = Field(default_factory=list)
    water_level_evidence: list[WaterLevelObservation] = Field(default_factory=list)
    hydraulic_evidence: list[PumpingTestRecord] = Field(default_factory=list)
    water_quality_evidence: list[dict] = Field(default_factory=list)
    hydrometry_evidence: list[dict] = Field(default_factory=list)
    documentary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    documentary_notes: list[str] = Field(default_factory=list)

    # ── Temporal summary (derived) ──
    temporal_summary: TemporalSummary = Field(default_factory=TemporalSummary)

    # ── Lithology validation ──
    lithology_validation: LithologyValidationResult = Field(
        default_factory=LithologyValidationResult
    )

    # ── Water presence (non-naive) ──
    water_presence: WaterPresenceDetermination = Field(default_factory=WaterPresenceDetermination)

    # ── Target readiness ──
    target_readiness: dict[str, TargetReadinessAssessment] = Field(default_factory=dict)

    # ── Provenance ──
    provenance: dict = Field(
        default_factory=dict,
        description="Provenance links for every dimension.",
    )

    # ── Source response change flags ──
    source_response_changed: bool = Field(
        default=False,
        description="Whether any endpoint returned different content on re-request.",
    )
    changed_endpoints: list[str] = Field(
        default_factory=list, description="Endpoints whose content changed across requests."
    )
