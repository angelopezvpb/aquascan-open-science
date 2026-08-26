"""Data validation utilities for AquaScan records and datasets."""

from __future__ import annotations

from collections import Counter

from aquascan.data.models import CampaignRecord


class ValidationIssue:
    """A single validation issue found in a record or dataset."""

    def __init__(
        self,
        level: str,
        field: str,
        message: str,
        record_id: str | None = None,
    ) -> None:
        self.level = level  # "error", "warning", "info"
        self.field = field
        self.message = message
        self.record_id = record_id

    def __str__(self) -> str:
        prefix = f"[{self.level.upper()}]"
        if self.record_id:
            prefix += f" [{self.record_id}]"
        return f"{prefix} {self.field}: {self.message}"

    def __repr__(self) -> str:
        return self.__str__()


class ValidationReport:
    """Result of validating a dataset or record."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []
        self._records_validated = 0

    def add(
        self,
        level: str,
        field: str,
        message: str,
        record_id: str | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(level, field, message, record_id))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def records_validated(self) -> int:
        return self._records_validated

    def summary(self) -> str:
        """Return a human-readable summary."""
        parts = [f"Records validated: {self._records_validated}"]
        counts = Counter(i.level for i in self.issues)
        for level in ("error", "warning", "info"):
            if counts.get(level):
                parts.append(f"{level}s: {counts[level]}")
        if not self.issues:
            parts.append("No issues found.")
        return ", ".join(parts)

    def __str__(self) -> str:
        lines = [self.summary()]
        if self.issues:
            lines.append("---")
            for issue in self.issues[:20]:
                lines.append(str(issue))
            if len(self.issues) > 20:
                lines.append(f"... and {len(self.issues) - 20} more issues")
        return "\n".join(lines)


def validate_campaign(record: CampaignRecord) -> ValidationReport:
    """Validate a single CampaignRecord for data quality issues.

    Args:
        record: The campaign record to validate.

    Returns:
        A ValidationReport with any issues found.
    """
    report = ValidationReport()
    report._records_validated = 1
    rid = record.campaign_id

    # Required identity fields
    if not record.campaign_id:
        report.add("error", "campaign_id", "Missing campaign ID", rid)
    if not record.site_id:
        report.add("error", "site_id", "Missing site ID", rid)

    # Spatial validation
    if record.latitude is not None:
        if not (-90.0 <= record.latitude <= 90.0):
            report.add("error", "latitude", f"Out of range: {record.latitude}", rid)
    else:
        report.add("warning", "latitude", "Missing latitude", rid)

    if record.longitude is not None:
        if not (-180.0 <= record.longitude <= 180.0):
            report.add("error", "longitude", f"Out of range: {record.longitude}", rid)
    else:
        report.add("warning", "longitude", "Missing longitude", rid)

    # Quality range validation
    for field_name, value in [
        ("ert_quality", record.ert_quality),
        ("seismic_quality", record.seismic_quality),
    ]:
        if value is not None and not (0.0 <= value <= 1.0):
            report.add("warning", field_name, f"Outside [0,1]: {value}", rid)

    # Physical sanity checks
    if record.total_depth is not None and record.total_depth < 0:
        report.add("error", "total_depth", f"Negative depth: {record.total_depth}", rid)

    if record.water_depth_min is not None and record.water_depth_max is not None:
        if record.water_depth_min > record.water_depth_max:
            report.add(
                "error",
                "water_depth",
                f"water_depth_min ({record.water_depth_min}) > "
                f"water_depth_max ({record.water_depth_max})",
                rid,
            )

    if record.total_depth is not None and record.static_water_level is not None:
        if record.static_water_level > record.total_depth:
            report.add(
                "warning",
                "static_water_level",
                f"Static water level ({record.static_water_level}m) "
                f"deeper than total depth ({record.total_depth}m)",
                rid,
            )

    # Ground truth validation
    if record.ground_truth_score is not None:
        if not (0.0 <= record.ground_truth_score <= 1.0):
            report.add(
                "error",
                "ground_truth_score",
                f"Out of range [0,1]: {record.ground_truth_score}",
                rid,
            )

    # Schema version check
    if not record.schema_version:
        report.add("warning", "schema_version", "Missing schema version", rid)

    return report


def validate_dataset(records: list[CampaignRecord]) -> ValidationReport:
    """Validate a full dataset of campaign records.

    Args:
        records: List of campaign records.

    Returns:
        A ValidationReport with all issues found.
    """
    report = ValidationReport()
    report._records_validated = len(records)

    if not records:
        report.add("error", "dataset", "Empty dataset")
        return report

    # Check for duplicate campaign IDs
    ids = [r.campaign_id for r in records]
    dupes = [cid for cid, count in Counter(ids).items() if count > 1]
    if dupes:
        report.add(
            "error",
            "campaign_id",
            f"Duplicate campaign IDs found: {dupes[:5]}...",
        )

    # Validate each record
    for record in records:
        record_report = validate_campaign(record)
        report.issues.extend(record_report.issues)

    # Dataset-level checks
    has_coords = sum(1 for r in records if r.latitude is not None)
    if has_coords < len(records):
        report.add(
            "warning",
            "dataset",
            f"{len(records) - has_coords}/{len(records)} records missing coordinates",
        )

    has_gt = sum(1 for r in records if r.ground_truth_score is not None)
    if has_gt == 0:
        report.add("info", "dataset", "No ground truth scores found in dataset")
    else:
        cls_counts = Counter(
            r.ground_truth_class.value if r.ground_truth_class else "unset" for r in records
        )
        report.add(
            "info",
            "dataset",
            f"Ground truth class distribution: {dict(cls_counts)}",
        )

    return report


def validate_schema_version(record: CampaignRecord, expected: str = "1.0.0") -> bool:
    """Check that a record conforms to the expected schema version.

    Args:
        record: The record to check.
        expected: Expected schema version string.

    Returns:
        True if the record's schema version matches.
    """
    return record.schema_version == expected
