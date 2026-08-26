"""AquaScan data layer: models, normalization, validation, and provenance."""

from aquascan.data.models import (
    CampaignRecord,
    DatasetLevel,
    EvidenceScore,
    EvidenceType,
    ExtractionMethod,
    GroundTruthClass,
    GroundTruthEvidenceRecord,
    IGMEWaterPointRecord,
    MeasurementRecord,
    Provenance,
    QAFlags,
    WaterPrediction,
)
from aquascan.data.provenance import (
    ProvenanceTracker,
    compute_sha256,
    create_provenance_record,
)
from aquascan.data.validation import (
    validate_campaign,
    validate_dataset,
    validate_schema_version,
)

__all__ = [
    # Models
    "CampaignRecord",
    "DatasetLevel",
    "EvidenceScore",
    "EvidenceType",
    "ExtractionMethod",
    "GroundTruthClass",
    "GroundTruthEvidenceRecord",
    "IGMEWaterPointRecord",
    "MeasurementRecord",
    "Provenance",
    "QAFlags",
    "WaterPrediction",
    # Provenance
    "ProvenanceTracker",
    "compute_sha256",
    "create_provenance_record",
    # Validation
    "validate_campaign",
    "validate_dataset",
    "validate_schema_version",
]
