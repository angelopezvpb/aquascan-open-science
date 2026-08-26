"""Shared infrastructure for the AquaScan physical encoder stack.

Provides the deterministic processing-stage vocabulary, artifact
references with parent links, and SHA-256 hashing used across the
ERT and seismic pipelines.

Every derived artifact must reference its parent artifact hash,
processing version, parameter hash, and timestamp.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ProcessingStage(StrEnum):
    """Deterministic processing stages."""

    # ERT stages
    ERT_RAW = "ERT_RAW"
    ERT_CALIBRATED = "ERT_CALIBRATED"
    ERT_QC = "ERT_QC"
    ERT_APPARENT_RESISTIVITY = "ERT_APPARENT_RESISTIVITY"
    ERT_INVERSION_INPUT = "ERT_INVERSION_INPUT"
    ERT_INVERTED_MODEL = "ERT_INVERTED_MODEL"
    ERT_FEATURES = "ERT_FEATURES"
    # Seismic stages
    SEISMIC_RAW = "SEISMIC_RAW"
    SEISMIC_CALIBRATED = "SEISMIC_CALIBRATED"
    SEISMIC_QC = "SEISMIC_QC"
    SEISMIC_FILTERED_VIEW = "SEISMIC_FILTERED_VIEW"
    SEISMIC_FIRST_ARRIVAL_PICKS = "SEISMIC_FIRST_ARRIVAL_PICKS"
    SEISMIC_TRAVEL_TIMES = "SEISMIC_TRAVEL_TIMES"
    SEISMIC_VELOCITY_MODEL = "SEISMIC_VELOCITY_MODEL"
    SEISMIC_FEATURES = "SEISMIC_FEATURES"


def compute_artifact_hash(payload: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable payload."""
    if isinstance(payload, bytes):
        return hashlib.sha256(payload).hexdigest()
    try:
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = str(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ArtifactRef:
    """A versioned artifact with full provenance lineage.

    Every derived representation carries its parent artifact hash so the
    chain RAW -> calibrated -> QC -> ... -> features is fully traceable.
    """

    __slots__ = (
        "artifact_id",
        "stage",
        "parent_artifact",
        "content_hash",
        "processing_version",
        "parameter_hash",
        "timestamp",
    )

    def __init__(
        self,
        stage: ProcessingStage,
        content: Any,
        *,
        parent_artifact: str | None = None,
        processing_version: str = "0.1.0",
        parameter_hash: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.parent_artifact = parent_artifact
        self.content_hash = compute_artifact_hash(content)
        self.processing_version = processing_version
        self.parameter_hash = parameter_hash
        self.timestamp = _now_iso()
        self.artifact_id = artifact_id or f"{stage.value}-{self.content_hash[:12]}"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage.value,
            "parent_artifact": self.parent_artifact,
            "content_hash": self.content_hash,
            "processing_version": self.processing_version,
            "parameter_hash": self.parameter_hash,
            "timestamp": self.timestamp,
        }


def make_artifact_ref(
    stage: ProcessingStage,
    content: Any,
    *,
    parent_artifact: str | None = None,
    processing_version: str = "0.1.0",
    parameter_hash: str | None = None,
) -> ArtifactRef:
    """Convenience factory for ArtifactRef."""
    return ArtifactRef(
        stage,
        content,
        parent_artifact=parent_artifact,
        processing_version=processing_version,
        parameter_hash=parameter_hash,
    )
