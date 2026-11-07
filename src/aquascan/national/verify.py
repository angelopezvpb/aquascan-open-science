"""Verification for the National Enrichment Engine (Gate 32).

aquascan national verify checks, WITHOUT making any network requests:
- state DB integrity
- cache hashes
- partition hashes
- duplicate IDs / missing IDs
- stale RUNNING jobs
- telemetry invariants
- source snapshot hash
- (optional) 4744-record regression canary equivalence
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from aquascan.national.cache import DurableCache
from aquascan.national.canary import canary_report, run_canary
from aquascan.national.io_utils import sha256_file
from aquascan.national.state import StateDB, now_iso


def verify_state(
    cfg,
    db: StateDB,
    *,
    check_canary: bool = True,
    check_partitions: bool = True,
) -> dict[str, Any]:
    """Run all offline checks and return a structured report.

    check_canary: run the 4744-record regression canary (offline, cache only).
    """
    report: dict[str, Any] = {
        "timestamp": now_iso(),
        "checks": {},
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # ── 1. State DB integrity ────────────────────────────────────────
    integrity = db.integrity_check()
    ok = all(row == "ok" for row in integrity)
    report["checks"]["state_db_integrity"] = {
        "valid": ok,
        "result": integrity,
    }
    if not ok:
        report["valid"] = False
        report["errors"].append("state DB integrity check failed")

    # ── 2. Telemetry invariant ───────────────────────────────────────
    inv = db.telemetry_invariant_holds()
    report["checks"]["telemetry_invariant"] = {
        "valid": inv,
        "result": db.telemetry(),
    }
    if not inv:
        report["valid"] = False
        report["errors"].append("telemetry invariant violated: logical != network + cache_hits")

    # ── 3. Source snapshot hash ──────────────────────────────────────
    source_hash_mismatch = False
    if cfg.source_hash and cfg.source.exists():
        actual = sha256_file(cfg.source)
        source_hash_mismatch = actual != cfg.source_hash
        report["checks"]["source_snapshot_hash"] = {
            "valid": not source_hash_mismatch,
            "expected": cfg.source_hash,
            "actual": actual,
        }
        if source_hash_mismatch:
            report["valid"] = False
            report["errors"].append("source snapshot hash changed since init")
    else:
        report["checks"]["source_snapshot_hash"] = {
            "valid": True,
            "note": "source snapshot not available; skipped",
        }

    # ── 4. Record counts / duplicates / missing IDs ──────────────────
    db_count = db.record_count()
    expected = cfg.source_record_count
    counts = db.record_status_counts()
    dupes = db._query_one(
        "SELECT igme_id, COUNT(*) AS n FROM records GROUP BY igme_id HAVING n > 1"
    )
    report["checks"]["record_counts"] = {
        "valid": (dupes is None) and (expected == 0 or db_count >= expected),
        "db_count": db_count,
        "expected": expected,
        "status_counts": counts,
        "duplicates": dupes is not None,
    }
    if dupes is not None:
        report["valid"] = False
        report["errors"].append("duplicate igme_id rows in records table")
    if expected and db_count < expected:
        report["valid"] = False
        report["errors"].append(f"records table has {db_count} rows but source has {expected}")

    # Missing IDs: source ids absent from the records table
    missing_ids: list[str] = []
    if cfg.source.exists():
        src = pd.read_parquet(cfg.source, columns=["igme_id"])
        present = {r["igme_id"] for r in db._query("SELECT igme_id FROM records")}
        missing_ids = [i for i in src["igme_id"].tolist() if i not in present]
    report["checks"]["missing_ids"] = {
        "valid": len(missing_ids) == 0,
        "count": len(missing_ids),
        "sample": missing_ids[:10],
    }
    if missing_ids:
        report["valid"] = False
        report["errors"].append(f"{len(missing_ids)} source ids missing from state DB")

    # ── 5. Stale RUNNING jobs ────────────────────────────────────────
    stale_before = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=cfg.stale_timeout_seconds)
    ).isoformat()
    stale = db.stale_running_jobs(stale_before)
    report["checks"]["stale_running_jobs"] = {
        "valid": len(stale) == 0,
        "count": len(stale),
        "sample": stale[:10],
    }
    if stale:
        report["warnings"].append(
            f"{len(stale)} stale RUNNING job(s) detected — recoverable on resume"
        )

    # ── 6. Cache hash verification ───────────────────────────────────
    cache = DurableCache(cfg.state_dir / "cache", cfg.request_version)
    cache_check = cache.verify_integrity()
    report["checks"]["cache_hashes"] = cache_check
    if not cache_check["valid"]:
        report["valid"] = False
        report["errors"].append(f"{cache_check['corrupted']} corrupted cache entries")

    # ── 7. Partition hash verification ───────────────────────────────
    if check_partitions:
        partition_errors = _verify_partitions(cfg)
        report["checks"]["partition_hashes"] = {
            "valid": len(partition_errors) == 0,
            "partitions_checked": len(list(cfg.interim_dir.glob("part-*.parquet"))),
            "errors": partition_errors[:20],
        }
        if partition_errors:
            report["valid"] = False
            report["errors"].append(f"{len(partition_errors)} partition hash mismatches")
    else:
        report["checks"]["partition_hashes"] = {"valid": True, "skipped": True}

    # ── 8. Regression canary ─────────────────────────────────────────
    if check_canary and cfg.canary.enabled:
        canary = run_canary(cfg.canary.cohort_path, cfg.canary.cache_path)
        canary_report(canary, cfg.reports_dir)
        report["checks"]["canary"] = {
            "valid": canary.valid,
            "compared": canary.compared,
            "total": canary.total,
            "mismatches": canary.mismatches,
            "detail": canary.detail,
        }
        if canary.inputs_missing:
            # Missing canary inputs are a warning, not a hard stop.
            report["warnings"].append("regression canary inputs missing — canary skipped")
        elif not canary.valid and cfg.canary.fail_on_mismatch:
            report["valid"] = False
            report["errors"].append(
                f"regression canary failed: {canary.detail} — STOP NATIONAL RUN"
            )
    else:
        report["checks"]["canary"] = {"valid": True, "skipped": True}

    return report


def _verify_partitions(cfg) -> list[str]:
    """Re-hash every partition file against its sidecar hash JSON."""
    errors: list[str] = []
    for part in sorted(cfg.interim_dir.glob("part-*.parquet")):
        sidecar = part.with_suffix(".parquet.hash.json")
        if not sidecar.exists():
            errors.append(f"missing sidecar for {part.name}")
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError):
            errors.append(f"unreadable sidecar for {part.name}")
            continue
        actual = sha256_file(part)
        if actual != meta.get("partition_sha256"):
            errors.append(
                f"{part.name}: hash mismatch ({actual[:12]} != "
                f"{meta.get('partition_sha256', '')[:12]})"
            )
    return errors


def verify_endpoint_statuses(cfg, db: StateDB) -> dict[str, int]:
    """Endpoint coverage summary used by status/finalize."""
    return db.job_status_counts()


def run_verify(cfg, check_canary: bool = True) -> dict[str, Any]:
    """Convenience: open DB and verify."""
    db = StateDB(cfg.state_db, create=False)
    try:
        return verify_state(cfg, db, check_canary=check_canary)
    finally:
        db.close()


def verify_checkpoint_integrity(db: StateDB) -> bool:
    """True when the latest checkpoint's hash matches its content."""
    cp = db.latest_checkpoint()
    if cp is None:
        return True
    return db.verify_checkpoint(cp["checkpoint_id"])
