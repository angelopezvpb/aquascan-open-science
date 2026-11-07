"""Final assembly of the national dataset (Gate 25).

Only when enrichment is complete:

    data/processed/aquascan_national_dataset_v1.parquet

FINALIZE validates record counts, uniqueness, schemas, hashes, endpoint
coverage, QA and provenance. The final dataset is never rewritten during
acquisition — only immutable partitions are written during the run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from aquascan.national import ENGINE_VERSION
from aquascan.national.constants import TERMINAL_RECORD_STATUSES
from aquascan.national.io_utils import atomic_write_json, atomic_write_parquet, sha256_file
from aquascan.national.state import StateDB, now_iso
from aquascan.national.verify import _verify_partitions


def finalize(cfg, db: StateDB) -> dict[str, Any]:
    """Assemble and validate the final national dataset.

    Returns a report dict with 'valid' and 'errors'. Raises on missing
    partitions; never silently drops records.
    """
    report: dict[str, Any] = {
        "timestamp": now_iso(),
        "engine_version": ENGINE_VERSION,
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # ── 1. All records must be terminal ─────────────────────────────
    counts = db.record_status_counts()
    non_terminal = sum(v for k, v in counts.items() if k not in TERMINAL_RECORD_STATUSES)
    if non_terminal > 0:
        report["valid"] = False
        report["errors"].append(f"{non_terminal} records are not terminal — enrichment incomplete")
        return report

    # ── 2. All batches must be complete (partitions written) ────────
    incomplete = db.incomplete_batches()
    if incomplete:
        report["valid"] = False
        report["errors"].append(f"{len(incomplete)} batches incomplete: {incomplete[:5]}")
        return report

    partitions = sorted(cfg.interim_dir.glob("part-*.parquet"))
    if not partitions:
        report["valid"] = False
        report["errors"].append("no partitions found to assemble")
        return report

    # ── 3. Partition hash verification ──────────────────────────────
    partition_errors = _verify_partitions(cfg)
    if partition_errors:
        report["valid"] = False
        report["errors"].append(f"{len(partition_errors)} partition hash mismatches")
        return report
    report["partitions"] = len(partitions)

    # ── 4. Assemble ─────────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for part in partitions:
        frames.append(pd.read_parquet(part))
    df = pd.concat(frames, ignore_index=True)

    expected = cfg.source_record_count or db.record_count()
    report["record_count"] = len(df)
    if len(df) != expected:
        report["valid"] = False
        report["errors"].append(f"assembled {len(df)} records but expected {expected}")
        return report

    # ── 5. Uniqueness / missing IDs ─────────────────────────────────
    if "igme_id" not in df.columns:
        report["valid"] = False
        report["errors"].append("igme_id column missing from assembled dataset")
        return report
    dupes = int(df["igme_id"].duplicated().sum())
    report["duplicate_ids"] = dupes
    if dupes:
        report["valid"] = False
        report["errors"].append(f"{dupes} duplicate igme_id rows in assembled dataset")

    source_ids = None
    if cfg.source.exists():
        source_ids = set(pd.read_parquet(cfg.source, columns=["igme_id"])["igme_id"])
        missing = source_ids - set(df["igme_id"])
        report["missing_ids"] = len(missing)
        if missing:
            report["valid"] = False
            report["errors"].append(f"{len(missing)} source ids missing from dataset")

    # ── 6. Endpoint coverage + QA accounting ────────────────────────
    status_counts = df["record_status"].value_counts().to_dict() if "record_status" in df else {}
    report["record_status_counts"] = {str(k): int(v) for k, v in status_counts.items()}
    report["qa_counts"] = db.qa_counts()
    report["failures"] = db.failure_count()

    endpoint_cols = [
        "no_lithology",
        "no_piezometry",
        "no_pumping_test",
        "no_chemistry",
        "no_hydrometry",
        "no_technical",
        "no_documents",
    ]
    coverage = {}
    for col in endpoint_cols:
        if col in df.columns:
            coverage[col] = int((~df[col]).sum())
    report["endpoint_coverage"] = coverage

    # ── 7. Provenance present ───────────────────────────────────────
    if "provenance_json" in df.columns:
        has_prov = df["provenance_json"].notna().sum()
        report["provenance_present"] = int(has_prov)
        if has_prov < len(df):
            report["warnings"].append(f"{len(df) - has_prov} records lack provenance_json")

    if not report["valid"]:
        return report

    # ── 8. Write final dataset (atomic) ─────────────────────────────
    final_path = cfg.final_dataset
    atomic_write_parquet(final_path, df)
    final_hash = sha256_file(final_path)
    report["dataset_path"] = str(final_path)
    report["dataset_sha256"] = final_hash

    # ── 9. Manifest ─────────────────────────────────────────────────
    manifest = {
        "dataset": "aquascan_national_dataset_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "engine_version": ENGINE_VERSION,
        "schema_version": cfg.schema_version,
        "source": str(cfg.source),
        "source_hash": cfg.source_hash,
        "source_record_count": cfg.source_record_count,
        "record_count": len(df),
        "dataset_sha256": final_hash,
        "partitions": [{"path": str(p), "sha256": sha256_file(p)} for p in partitions],
        "record_status_counts": report["record_status_counts"],
        "endpoint_coverage": coverage,
        "qa_counts": report["qa_counts"],
        "failures": report["failures"],
        "verification": {
            "uniqueness": report["duplicate_ids"] == 0,
            "missing_ids": report.get("missing_ids", 0) == 0,
            "partition_hashes_valid": True,
        },
        "note": (
            "Ground Truth targets are represented by readiness_* fields "
            "(per-target, independent). No global success/failure label is created. "
            "Observation dates and temporality are preserved (RECENT/HISTORICAL/"
            "MULTI_DECADE_SERIES/DATE_UNKNOWN)."
        ),
    }
    atomic_write_json(cfg.reports_dir / "national_enrichment_manifest_v1.json", manifest)
    report["manifest"] = str(cfg.reports_dir / "national_enrichment_manifest_v1.json")

    # Mark engine COMPLETE (only after successful assembly)
    db.set_engine_state("COMPLETE")
    return report
