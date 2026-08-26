"""Provenance tracking — every processed value must trace back to RAW."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from aquascan.data.models import Provenance


def compute_sha256(data: bytes | str) -> str:
    """Compute SHA-256 hash of data.

    Args:
        data: Raw bytes or string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def create_provenance_record(
    processing_step: str,
    source_uri: str | None = None,
    source_hash: str | None = None,
    parent_artifact: str | None = None,
    processing_version: str = "0.1.0",
    instrument_version: str | None = None,
) -> Provenance:
    """Create a provenance record for a data transformation.

    Args:
        processing_step: Name of the step (e.g., 'igme_normalize').
        source_uri: URI of the source data.
        source_hash: SHA-256 hash of the source data.
        parent_artifact: ID of the parent artifact.
        processing_version: Version of the processing code.
        instrument_version: Version of the instrument.

    Returns:
        A Provenance instance.
    """
    return Provenance(
        source_uri=source_uri,
        source_hash=source_hash,
        processing_step=processing_step,
        processing_version=processing_version,
        parent_artifact=parent_artifact,
        instrument_version=instrument_version,
        timestamp=datetime.now(UTC),
    )


class ProvenanceTracker:
    """Tracks provenance across a chain of data transformations.

    Each transformation step records what it received and what it produced,
    maintaining a complete audit trail from RAW to final product.
    """

    def __init__(self) -> None:
        self._steps: list[Provenance] = []

    def record(
        self,
        processing_step: str,
        source_uri: str | None = None,
        source_hash: str | None = None,
        parent_artifact: str | None = None,
    ) -> Provenance:
        """Record a processing step and return its provenance record.

        Args:
            processing_step: Name of the processing step.
            source_uri: Source URI.
            source_hash: SHA-256 of source.
            parent_artifact: Parent artifact ID.

        Returns:
            The provenance record for this step.
        """
        record = create_provenance_record(
            processing_step=processing_step,
            source_uri=source_uri,
            source_hash=source_hash,
            parent_artifact=parent_artifact,
        )
        self._steps.append(record)
        return record

    @property
    def steps(self) -> list[Provenance]:
        """Return all recorded provenance steps."""
        return list(self._steps)

    def to_json(self) -> str:
        """Export the provenance chain as JSON."""
        return json.dumps(
            [s.model_dump(mode="json") for s in self._steps],
            indent=2,
            default=str,
        )

    def latest(self) -> Provenance | None:
        """Return the most recent provenance record, if any."""
        return self._steps[-1] if self._steps else None
