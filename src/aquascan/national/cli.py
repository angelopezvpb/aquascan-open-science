"""CLI for the National Enrichment Engine (Gate 35).

Commands:

    aquascan national init
    aquascan national start
    aquascan national pause
    aquascan national stop
    aquascan national resume
    aquascan national status
    aquascan national retry-failed
    aquascan national verify
    aquascan national finalize

Exact continuation from durable state is supported: resume never restarts
from record 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from aquascan.national import ENGINE_VERSION
from aquascan.national.config import (
    DEFAULT_SOURCE,
    NationalConfig,
    load_config,
    save_config,
    stamp_config,
)
from aquascan.national.engine import Controls, NationalEnrichmentEngine
from aquascan.national.finalize import finalize
from aquascan.national.io_utils import sha256_file
from aquascan.national.progress import render_status_text
from aquascan.national.state import StateDB
from aquascan.national.verify import run_verify


def _config_path() -> Path:
    """Config path — overridable via AQUASCAN_NATIONAL_CONFIG (tests/ops)."""
    import os

    env = os.environ.get("AQUASCAN_NATIONAL_CONFIG")
    return Path(env) if env else Path("reports/national_enrichment_config_v1.json")


def _load_cfg(require: bool = True) -> NationalConfig | None:
    path = _config_path()
    if not path.exists():
        if require:
            raise click.ClickException(
                "national engine not initialized — run `aquascan national init` first"
            )
        return None
    return load_config(path)


def _open_db(cfg: NationalConfig) -> StateDB:
    return StateDB(cfg.state_db, create=True)


def _print_config_warning(cfg: NationalConfig) -> None:
    if cfg.api_mode == "simulated":
        click.echo(
            click.style(
                "WARNING: API mode is SIMULATED — synthetic data only. "
                "This never validates real-world performance.",
                fg="yellow",
            )
        )


@click.group()
def national() -> None:
    """AquaScan National Enrichment Engine v1 (resumable, crash-safe)."""


@national.command("init")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SOURCE,
    help="Normalized national snapshot (Parquet).",
)
@click.option(
    "--limit", type=int, default=None, help="Seed only the first N records (dry-run testing)."
)
@click.option(
    "--simulated",
    is_flag=True,
    help="Use the deterministic offline API "
    "(dry runs / tests only — synthetic data is NOT field validation).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-initialize even if the state DB belongs to a different source snapshot.",
)
@click.option("--batch-size", type=int, default=500)
@click.option("--concurrency", type=int, default=4)
def init_cmd(
    source: Path,
    limit: int | None,
    simulated: bool,
    force: bool,
    batch_size: int,
    concurrency: int,
) -> None:
    """Initialize the national state DB from the normalized snapshot."""
    # Respect an existing config (paths can be overridden via
    # AQUASCAN_NATIONAL_CONFIG for tests/ops); otherwise use defaults.
    if _config_path().exists():
        cfg = load_config(_config_path())
        cfg.source = source
    else:
        cfg = NationalConfig(source=source)
    cfg.batch_size = batch_size
    cfg.concurrency = concurrency
    if simulated:
        cfg.api_mode = "simulated"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(source)
    df = pd.read_parquet(source, columns=["igme_id"])
    all_ids = [str(i) for i in df["igme_id"].tolist()]
    ids = all_ids[:limit] if limit is not None else all_ids
    if not ids:
        raise click.ClickException("source snapshot has no records")

    if cfg.state_db.exists():
        existing = StateDB(cfg.state_db, create=False)
        try:
            old_hash = existing.get_meta("source_hash")
        finally:
            existing.close()
        if old_hash and old_hash != source_hash and not force:
            raise click.ClickException(
                "state DB belongs to a different source snapshot — use --force "
                "to re-initialize (this discards progress) or point at the same source"
            )
        if old_hash == source_hash:
            click.echo(
                "State DB already initialized for this source snapshot — "
                "adding any missing records."
            )

    cfg = stamp_config(cfg, source_hash, len(ids))
    save_config(cfg)

    db = _open_db(cfg)
    try:
        db.set_meta("source_hash", source_hash)
        db.set_meta("engine_version", ENGINE_VERSION)
        db.set_meta("schema_version", cfg.schema_version)
        db.set_meta("api_mode", cfg.api_mode)
        db.set_engine_state("STOPPED")
        batch_ids = _assign_batches(ids, batch_size)
        inserted, existing = db.seed_records(ids, batch_ids, source_hash)
        db.create_batches(batch_size)
        db.set_meta("source_record_count", str(len(ids)))
        db.set_meta("source_total", str(len(all_ids)))
    finally:
        db.close()

    _print_config_warning(cfg)
    click.echo(
        f"Initialized national state: {len(ids)} records (inserted {inserted}, existing {existing})"
    )
    click.echo(f"  source: {source}")
    click.echo(f"  source hash: {source_hash[:16]}…")
    click.echo(f"  state DB: {cfg.state_db}")
    click.echo(f"  config: {_config_path()}")
    if limit is not None:
        click.echo(
            click.style(
                f"NOTE: seeded only {len(ids)} records (--limit {limit}) — "
                f"the full universe has {len(all_ids)}.",
                fg="yellow",
            )
        )


def _assign_batches(ids: list[str], batch_size: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(ids), batch_size):
        batch_id = f"B{i // batch_size + 1:05d}"
        for g in ids[i : i + batch_size]:
            out[g] = batch_id
    return out


@national.command("start")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process at most N records this session (dry-run testing).",
)
@click.option(
    "--simulated",
    is_flag=True,
    help="Override API mode to simulated (synthetic data is NOT field validation).",
)
@click.option("--no-canary", is_flag=True, help="Skip the 4744-record regression canary.")
@click.option("--quiet", is_flag=True, help="Disable the live progress display.")
def start_cmd(limit: int | None, simulated: bool, no_canary: bool, quiet: bool) -> None:
    """Start (or continue) national enrichment. Ctrl+C checkpoints safely."""
    cfg = _load_cfg()
    if cfg is None:
        return
    if simulated:
        cfg.api_mode = "simulated"
    if no_canary:
        cfg.canary.enabled = False
    _print_config_warning(cfg)
    engine = NationalEnrichmentEngine(cfg)
    final_state = engine.run(limit=limit, quiet=quiet)
    click.echo(f"Engine finished: {final_state}")
    _print_remaining(cfg)


@national.command("resume")
@click.option("--limit", type=int, default=None)
@click.option("--simulated", is_flag=True)
@click.option("--no-canary", is_flag=True)
@click.option("--quiet", is_flag=True)
def resume_cmd(limit: int | None, simulated: bool, no_canary: bool, quiet: bool) -> None:
    """Validate durable state/cache, recover stale tasks, and continue exactly."""
    cfg = _load_cfg()
    if cfg is None:
        return
    if simulated:
        cfg.api_mode = "simulated"
    if no_canary:
        cfg.canary.enabled = False
    _print_config_warning(cfg)

    db = _open_db(cfg)
    try:
        counts = db.record_status_counts()
        total = db.record_count()
        complete = counts.get("COMPLETE", 0)
        remaining = total - complete
        failed = counts.get("FAILED", 0) + counts.get("QUARANTINED", 0)
        click.echo("National enrichment v1")
        click.echo(f"Total:      {total:>10,}")
        click.echo(f"Complete:   {complete:>10,}")
        click.echo(f"Remaining:  {remaining:>10,}")
        click.echo(f"Failed:     {failed:>10,}")
        click.echo("Resuming from durable checkpoint...")
    finally:
        db.close()

    engine = NationalEnrichmentEngine(cfg)
    final_state = engine.run(limit=limit, quiet=quiet)
    click.echo(f"Engine finished: {final_state}")
    _print_remaining(cfg)


@national.command("pause")
def pause_cmd() -> None:
    """Gracefully pause a running enrichment (no data loss)."""
    cfg = _load_cfg()
    if cfg is None:
        return
    controls = Controls(cfg.state_dir / "control")
    db = _open_db(cfg)
    try:
        if db.engine_state() == "RUNNING":
            controls.request_pause()
            click.echo(
                "Pause requested — the engine will stop accepting new jobs, "
                "finish active jobs, persist state and verify the checkpoint."
            )
        else:
            db.set_engine_state("PAUSED", "paused by user (engine not running)")
            click.echo("Engine not running — state marked PAUSED.")
    finally:
        db.close()


@national.command("stop")
def stop_cmd() -> None:
    """Stop the engine (checkpoint + exit, resumable later)."""
    cfg = _load_cfg()
    if cfg is None:
        return
    controls = Controls(cfg.state_dir / "control")
    db = _open_db(cfg)
    try:
        if db.engine_state() == "RUNNING":
            controls.request_stop()
            click.echo("Stop requested — the engine will checkpoint and exit.")
        else:
            db.set_engine_state("STOPPED", "stopped by user (engine not running)")
            click.echo("Engine not running — state marked STOPPED.")
    finally:
        db.close()


@national.command("status")
@click.option("--json-output", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def status_cmd(as_json: bool) -> None:
    """Show enrichment status. Works even when the engine is not running."""
    cfg = _load_cfg(require=False)
    if cfg is None or not cfg.state_db.exists():
        raise click.ClickException("national engine not initialized")
    from aquascan.national.cache import DurableCache
    from aquascan.national.progress import compute_status

    db = _open_db(cfg)
    try:
        cache = DurableCache(cfg.state_dir / "cache", cfg.request_version)
        status = compute_status(cfg, db, cache)
        status["engine_status"] = db.engine_state()
    finally:
        db.close()

    if as_json:
        click.echo(json.dumps(status, indent=2, default=str))
        return
    click.echo(render_status_text(status))


@national.command("retry-failed")
@click.option("--all", "retry_all", is_flag=True, help="Retry every failed endpoint job.")
@click.option(
    "--endpoint", type=str, default=None, help="Retry only this endpoint (e.g. chemistry)."
)
@click.option("--record", "record_id", type=str, default=None, help="Retry only this record.")
@click.option(
    "--max-attempts", type=int, default=None, help="Only retry jobs with fewer than N attempts."
)
def retry_failed_cmd(
    retry_all: bool, endpoint: str | None, record_id: str | None, max_attempts: int | None
) -> None:
    """Re-open failed/RETRY_WAIT endpoint jobs (original errors are preserved)."""
    if not (retry_all or endpoint or record_id):
        raise click.ClickException("specify --all, --endpoint NAME, or --record ID")
    cfg = _load_cfg()
    if cfg is None:
        return
    db = _open_db(cfg)
    try:
        # Capture affected records BEFORE resetting jobs (they lose their
        # FAILED/RETRY_WAIT status once re-opened).
        clauses = ["status IN ('FAILED', 'RETRY_WAIT')"]
        params: list[Any] = []
        if endpoint:
            clauses.append("endpoint = ?")
            params.append(endpoint)
        if record_id:
            clauses.append("record_id = ?")
            params.append(record_id)
        if max_attempts is not None:
            clauses.append("attempts < ?")
            params.append(max_attempts)
        affected = [
            r["record_id"]
            for r in db._query(
                "SELECT DISTINCT record_id FROM endpoint_jobs WHERE " + " AND ".join(clauses),
                tuple(params),
            )
        ]
        reset = db.reset_failed_endpoints(
            endpoint=endpoint, record_id=record_id, max_attempts=max_attempts
        )
        # Re-open affected records so they are re-scheduled on resume.
        records_reset = db.reset_records_to_pending(affected)
    finally:
        db.close()
    click.echo(f"Re-opened {reset} endpoint job(s); {records_reset} record(s) marked PENDING.")
    click.echo("Run `aquascan national resume` to continue.")


@national.command("verify")
@click.option("--no-canary", is_flag=True, help="Skip the regression canary.")
@click.option("--json-output", "as_json", is_flag=True)
def verify_cmd(no_canary: bool, as_json: bool) -> None:
    """Verify state DB, cache hashes, partition hashes, invariants (no network)."""
    cfg = _load_cfg()
    if cfg is None:
        return
    report = run_verify(cfg, check_canary=not no_canary)
    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        click.echo("National enrichment verify")
        click.echo("=" * 60)
        for name, check in report["checks"].items():
            mark = "PASS" if check.get("valid", True) else "FAIL"
            color = "green" if mark == "PASS" else "red"
            extra = ""
            if name == "telemetry_invariant" and check.get("result"):
                t = check["result"]
                extra = (
                    f" (logical={t['logical_endpoint_queries']} "
                    f"network={t['network_requests']} cache={t['endpoint_cache_hits']})"
                )
            if name == "canary":
                extra = (
                    f" ({check.get('compared', 0)} compared, "
                    f"{check.get('mismatches', 0)} mismatches)"
                )
            click.echo(f"  [{mark}] {name}{extra}", color=color)  # type: ignore[arg-type]
        for w in report["warnings"]:
            click.echo(click.style(f"  [WARN] {w}", fg="yellow"))
        click.echo("=" * 60)
        if report["valid"]:
            click.echo(click.style("VERIFY: ALL CHECKS PASSED", fg="green"))
        else:
            for e in report["errors"]:
                click.echo(click.style(f"  ERROR: {e}", fg="red"))
            sys.exit(1)


@national.command("finalize")
def finalize_cmd() -> None:
    """Assemble the final dataset (only when enrichment is complete)."""
    cfg = _load_cfg()
    if cfg is None:
        return
    db = _open_db(cfg)
    try:
        report = finalize(cfg, db)
    finally:
        db.close()

    click.echo("National enrichment finalize")
    click.echo("=" * 60)
    click.echo(f"Partitions assembled: {report.get('partitions', 0)}")
    click.echo(f"Record count: {report.get('record_count', '?')}")
    click.echo(f"Duplicate IDs: {report.get('duplicate_ids', 0)}")
    click.echo(f"Missing IDs: {report.get('missing_ids', 0)}")
    click.echo(f"Record status: {report.get('record_status_counts', {})}")
    click.echo(f"Endpoint coverage: {report.get('endpoint_coverage', {})}")
    click.echo(f"QA: {report.get('qa_counts', {})}")
    if report.get("dataset_path"):
        click.echo(f"Dataset: {report['dataset_path']}")
        click.echo(f"SHA-256: {report['dataset_sha256']}")
    if not report["valid"]:
        for e in report["errors"]:
            click.echo(click.style(f"  ERROR: {e}", fg="red"))
        sys.exit(1)
    click.echo(click.style("FINALIZE: dataset assembled and validated", fg="green"))


def _print_remaining(cfg: NationalConfig) -> None:
    db = _open_db(cfg)
    try:
        counts = db.record_status_counts()
        terminal = sum(
            v
            for k, v in counts.items()
            if k in {"COMPLETE", "PARTIAL", "FAILED", "QUARANTINED", "SKIPPED"}
        )
        remaining = db.record_count() - terminal
        if remaining > 0:
            click.echo(
                f"Remaining records: {remaining:,} — run `aquascan national resume` to continue."
            )
        else:
            click.echo("All records terminal — run `aquascan national finalize` to assemble.")
    finally:
        db.close()


# Ensure the module exports what cli.py needs
__all__ = [
    "national",
    "init_cmd",
    "start_cmd",
    "resume_cmd",
    "pause_cmd",
    "stop_cmd",
    "status_cmd",
    "retry_failed_cmd",
    "verify_cmd",
    "finalize_cmd",
]
