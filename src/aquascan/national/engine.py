"""National Enrichment Engine core (AquaScan National Dataset v1).

Operational requirements implemented here:
- persistent SQLite execution state with endpoint-level granularity
- durable per-record cache with atomic writes
- exact resume from durable state (never restarts from record 1)
- stale RUNNING recovery after crashes
- graceful PAUSE / STOP via control files and SIGINT/SIGTERM
- bounded retries with exponential backoff + per-attempt history
- global rate limiting, bounded concurrency, network circuit breaker
- explicit checkpoints (records / time), immutable batch partitions
- incremental QA (never silently fixes), quarantine on critical issues
- disk-space monitoring (DISK_CRITICAL → safe pause)
- session history and telemetry correctness (Gate 21 invariant)
- 4744-record regression canary as a pre-flight gate

Scientific semantics are the validated Gate 4/5 semantics. This module
implements operational scale and reliability only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aquascan.national import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION
from aquascan.national.api import CircuitBreaker, make_fetcher
from aquascan.national.cache import DurableCache
from aquascan.national.config import NationalConfig
from aquascan.national.constants import (
    COMPLETE,
    CONTROL_PAUSE,
    CONTROL_STOP,
    ENDPOINT_NAMES,
    ENGINE_COMPLETE,
    ENGINE_DISK_CRITICAL,
    ENGINE_FAILED,
    ENGINE_NETWORK_DEGRADED,
    ENGINE_PAUSED,
    ENGINE_RUNNING,
    ENGINE_STOPPED,
    EP_COMPLETE,
    EP_FAILED,
    EP_RETRY_WAIT,
    FAILED,
    PARTIAL,
    QA_CRITICAL,
    QUARANTINED,
    TERMINAL_RECORD_STATUSES,
)
from aquascan.national.flatten import flatten_normalized_record, run_qa_checks
from aquascan.national.io_utils import (
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    sha256_text,
)
from aquascan.national.progress import ProgressDisplay, compute_status
from aquascan.national.state import StateDB, now_iso
from aquascan.national.verify import verify_state

logger = logging.getLogger("aquascan.national")


class Controls:
    """Cross-process control: pause/stop request files polled by the engine."""

    def __init__(self, control_dir: Path) -> None:
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.control_dir / name

    def request_pause(self) -> None:
        self._path(CONTROL_PAUSE).touch()

    def request_stop(self) -> None:
        self._path(CONTROL_STOP).touch()

    def clear(self) -> None:
        for name in (CONTROL_PAUSE, CONTROL_STOP):
            self._path(name).unlink(missing_ok=True)

    def poll(self) -> tuple[str | None, str | None]:
        """Return (pause_request_name, stop_request_name) currently pending."""
        pause = CONTROL_PAUSE if self._path(CONTROL_PAUSE).exists() else None
        stop = CONTROL_STOP if self._path(CONTROL_STOP).exists() else None
        return pause, stop


class NationalEnrichmentEngine:
    """Resumable, crash-safe national enrichment engine."""

    def __init__(self, cfg: NationalConfig) -> None:
        self.cfg = cfg
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = StateDB(cfg.state_db, create=True)
        self.cache = DurableCache(cfg.state_dir / "cache", cfg.request_version)
        self.controls = Controls(cfg.state_dir / "control")
        self.fetcher, self._rate_limiter = make_fetcher(cfg)
        self.circuit_breaker = CircuitBreaker(cfg.circuit_breaker)

        self._session_id: str | None = None
        self._session_started_at: float | None = None
        self._graceful_signal = False
        self._signal_count = 0
        self._records_since_checkpoint = 0
        self._last_checkpoint_time = time.monotonic()
        self._last_disk_check = 0.0
        self._last_disk_state = "DISK_OK"
        self._in_flight: dict[Future, str] = {}
        self._source_rows: dict[str, dict[str, Any]] = {}
        self._completed_this_session = 0
        self._stopping_reason = ""
        self._degraded_cycles = 0

    # ── signals ──────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self._signal_count += 1
            if self._signal_count >= 2:
                # Second forced signal: immediate exit (documented unsafe).
                logger.warning("second signal received — forced immediate exit (UNSAFE)")
                os._exit(1)
            self._graceful_signal = True
            logger.warning("signal %d received — graceful shutdown (checkpoint + exit)", signum)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    # ── run ──────────────────────────────────────────────────────────

    def run(self, limit: int | None = None, quiet: bool = False) -> str:
        """Run the engine until completion, pause, stop or signal.

        Returns the final engine state (COMPLETE / PAUSED / STOPPED /
        NETWORK_DEGRADED / DISK_CRITICAL / FAILED).
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("engine.run() must execute on the main thread (signals)")

        self._install_signal_handlers()
        self._configure_logging()

        # ── pre-flight validation ───────────────────────────────────
        if not self.cfg.source_hash or not self.cfg.source_record_count:
            logger.error("engine not initialized (run `aquascan national init` first)")
            self.state.set_engine_state(ENGINE_FAILED, "not initialized")
            return ENGINE_FAILED

        if not self.cfg.source.exists():
            logger.error("source snapshot missing: %s", self.cfg.source)
            self.state.set_engine_state(ENGINE_FAILED, "source snapshot missing")
            return ENGINE_FAILED

        report = verify_state(self.cfg, self.state, check_canary=self.cfg.canary.enabled)
        if not report["valid"]:
            logger.error("pre-flight verification failed: %s", report["errors"])
            self.state.set_engine_state(ENGINE_FAILED, "; ".join(report["errors"]))
            return ENGINE_FAILED

        # ── session bookkeeping ─────────────────────────────────────
        self._session_id = self.state.start_session(
            socket.gethostname(), ENGINE_VERSION, self.cfg.source_hash
        )
        self._session_started_at = time.monotonic()
        self.state.snapshot_session_counters(self._session_id)
        logger.info(
            "session %s started (engine %s, source hash %s)",
            self._session_id,
            ENGINE_VERSION,
            self.cfg.source_hash[:12],
        )

        # ── load source rows for metadata join ──────────────────────
        self._load_source_rows()

        # ── recover stale RUNNING jobs left by a crash ──────────────
        stale_before = (
            datetime.now(UTC) - timedelta(seconds=self.cfg.stale_timeout_seconds)
        ).isoformat()
        recovered = self.state.recover_stale_running(stale_before, ENDPOINT_NAMES)
        if recovered["jobs"] or recovered["records"]:
            logger.warning(
                "recovered stale state: %d jobs, %d records",
                recovered["jobs"],
                recovered["records"],
            )

        # ── disk pre-flight ─────────────────────────────────────────
        if self._disk_state() == "DISK_CRITICAL":
            logger.error("disk space critical at start — refusing to run")
            self.state.set_engine_state(ENGINE_DISK_CRITICAL, "insufficient free disk")
            self._end_session(ENGINE_DISK_CRITICAL)
            return ENGINE_DISK_CRITICAL

        self.controls.clear()
        self._graceful_signal = False
        self._signal_count = 0
        self.state.set_engine_state(ENGINE_RUNNING)
        logger.info(
            "engine RUNNING (concurrency=%d, rps=%.1f, batch=%d, api_mode=%s)",
            self.cfg.concurrency,
            self.cfg.rate_limit.requests_per_second,
            self.cfg.batch_size,
            self.cfg.api_mode,
        )

        display = ProgressDisplay(enabled=not quiet)
        try:
            final_state = self._run_loop(limit=limit, display=display)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("engine crashed: %s", exc)
            self.state.set_engine_state(ENGINE_FAILED, str(exc))
            final_state = ENGINE_FAILED
        finally:
            display.close()

        self._end_session(final_state)
        self._checkpoint(force=True)
        self._write_progress_json()
        self.export_reports()
        logger.info("engine finished with state %s", final_state)
        return final_state

    # ── main loop ────────────────────────────────────────────────────

    def _run_loop(self, limit: int | None, display: ProgressDisplay) -> str:
        executor = ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency))
        self._in_flight = {}
        self._degraded_cycles = 0
        try:
            while True:
                pause_req, stop_req = self.controls.poll()
                stop_pending = stop_req is not None
                pause_pending = pause_req is not None or self._graceful_signal

                if (stop_pending or pause_pending) and not self._in_flight:
                    self._stopping_reason = (
                        "stop request" if stop_pending else "pause request / signal"
                    )
                    return ENGINE_STOPPED if stop_pending else ENGINE_PAUSED

                # ── disk monitor ────────────────────────────────────
                disk_state = self._disk_state()
                if disk_state == "DISK_CRITICAL":
                    logger.error("DISK_CRITICAL — pausing safely")
                    self.state.set_engine_state(ENGINE_DISK_CRITICAL, "free disk critical")
                    if not self._in_flight:
                        return ENGINE_DISK_CRITICAL

                # ── circuit breaker ─────────────────────────────────
                if self.circuit_breaker.is_open():
                    self.state.set_engine_state(ENGINE_NETWORK_DEGRADED, "circuit breaker open")
                    if self._in_flight:
                        # Let in-flight work finish before backing off.
                        self._drain_done(display, timeout=0.5)
                        self._maybe_checkpoint()
                        continue
                    self._degraded_cycles += 1
                    logger.warning(
                        "NETWORK_DEGRADED — backing off (%d/%d)",
                        self._degraded_cycles,
                        10,
                    )
                    if self._degraded_cycles >= 10:
                        logger.error("network unavailable for too long — failing run")
                        return ENGINE_FAILED
                    time.sleep(min(self.cfg.circuit_breaker.cooldown_seconds, 5.0))
                    continue

                self.state.set_engine_state(ENGINE_RUNNING)

                # ── schedule new records ────────────────────────────
                limit_reached = limit is not None and self._completed_this_session >= limit
                if not stop_pending and not pause_pending and not limit_reached:
                    self._schedule(executor, limit)

                if not self._in_flight:
                    if limit_reached:
                        self._stopping_reason = f"session limit ({limit} records)"
                        return ENGINE_PAUSED
                    remaining = len(
                        self.state.eligible_records(now_iso(), self.cfg.resume_retries_failed)
                    )
                    if remaining == 0:
                        return ENGINE_COMPLETE
                    # Only RETRY_WAIT jobs not yet due remain — wait briefly.
                    time.sleep(1.0)
                    continue

                # ── wait for progress ───────────────────────────────
                self._drain_done(display, timeout=0.5)

                # checkpoint cadence (records or time, whichever first)
                self._maybe_checkpoint()

        finally:
            # Wait for in-flight work, cancel queued futures.
            executor.shutdown(wait=True, cancel_futures=True)

    def _drain_done(self, display: ProgressDisplay, timeout: float) -> None:
        """Wait briefly for worker futures and process any completed records."""
        done, _ = wait(self._in_flight, timeout=timeout)
        for future in done:
            igme_id = self._in_flight.pop(future)
            exc = future.exception()
            if exc is not None:
                # A worker crash quarantines the record (never silently fixed)
                # so the batch can still complete and the record is preserved.
                logger.error("record %s crashed in worker: %s", igme_id, exc)
                self.state.set_record_error(igme_id, str(exc), None)
                self.state.add_qa_violation(igme_id, "worker_crash", QA_CRITICAL, str(exc))
                self.state.complete_record(
                    igme_id,
                    QUARANTINED,
                    "",
                    None,
                    last_error=str(exc),
                    quarantine_reason=f"worker_crash: {exc}",
                )
            self._after_record(igme_id, display)

    def _schedule(self, executor: ThreadPoolExecutor, limit: int | None) -> None:
        eligible = self.state.eligible_records(now_iso(), self.cfg.resume_retries_failed)
        for igme_id in eligible:
            if igme_id in self._in_flight:
                continue
            if limit is not None and self._completed_this_session >= limit:
                break
            future = executor.submit(
                self._process_record, igme_id, self._source_rows.get(igme_id, {})
            )
            self._in_flight[future] = igme_id
            if len(self._in_flight) >= max(1, self.cfg.concurrency):
                break

    def _after_record(self, igme_id: str, display: ProgressDisplay) -> None:
        rec = self.state.get_record(igme_id)
        if rec is None:
            return
        if rec["status"] in TERMINAL_RECORD_STATUSES:
            self._records_since_checkpoint += 1
            self._completed_this_session += 1
            batch_id = rec["batch_id"]
            if batch_id:
                self._maybe_complete_batch(batch_id)
        status_dict = compute_status(
            self.cfg,
            self.state,
            self.cache,
            session_started_at=self._session_started_at,
            session_completed=self._completed_this_session,
        )
        display.render(status_dict)

    # ── per-record processing ────────────────────────────────────────

    def _process_record(self, igme_id: str, source_row: dict[str, Any]) -> str | None:
        """Process one record: endpoint-level jobs → normalization → output.

        Returns the terminal record status, or None if work remains pending
        (RETRY_WAIT backoff not yet due, or graceful shutdown in progress).
        """
        db = self.state
        db.ensure_endpoint_jobs(igme_id, list(ENDPOINT_NAMES))
        rec = db.get_record(igme_id)
        db.mark_record_running(igme_id, int(rec["attempt_count"] or 0) + 1)

        endpoint_results: dict[str, dict[str, Any]] = {}
        changed_endpoints: list[str] = []
        any_failed = False
        cache_hits_this = 0
        network_this = 0
        work_remaining = False

        for ep in ENDPOINT_NAMES:
            if self._graceful_signal:
                work_remaining = True
                break

            job = db.get_endpoint_job(igme_id, ep)
            job_status = job["status"]

            if job_status == EP_COMPLETE:
                data, r_hash = self.cache.get(igme_id, ep)
                endpoint_results[ep] = {
                    "status": "COMPLETE",
                    "response_hash": r_hash,
                    "count": len(data) if data is not None else 0,
                }
                continue

            if job_status == EP_FAILED:
                any_failed = True
                endpoint_results[ep] = {
                    "status": "FAILED",
                    "response_hash": "",
                    "count": 0,
                    "error": job.get("last_error"),
                }
                continue

            if job_status == EP_RETRY_WAIT:
                if job.get("next_retry_at") and job["next_retry_at"] > now_iso():
                    work_remaining = True
                    endpoint_results[ep] = {
                        "status": "RETRY_WAIT",
                        "response_hash": "",
                        "count": 0,
                        "error": job.get("last_error"),
                    }
                    continue
                # backoff elapsed → retry now
                db._execute(
                    "UPDATE endpoint_jobs SET status = 'PENDING' "
                    "WHERE record_id = ? AND endpoint = ?",
                    (igme_id, ep),
                )

            # PENDING (or RUNNING left by a crashed process) → resolve
            # (durable cache first, then network).
            db.mark_job_running(igme_id, ep)

            data, r_hash = self.cache.get(igme_id, ep)
            if data is not None and r_hash:
                db.add_counter("logical_endpoint_queries", 1)
                db.add_counter("endpoint_cache_hits", 1)
                db.mark_job_complete(igme_id, ep, r_hash, cache_hit=True)
                cache_hits_this += 1
                endpoint_results[ep] = {
                    "status": "COMPLETE",
                    "response_hash": r_hash,
                    "count": len(data),
                }
                continue

            db.add_counter("logical_endpoint_queries", 1)
            db.add_counter("network_requests", 1)
            network_this += 1
            result = self.fetcher.fetch(igme_id, ep)
            db.add_counter("http_requests_total", result.http_attempts)
            db.append_job_attempt(igme_id, ep, result.http_attempts)
            if result.http_attempts > 1:
                db.add_counter("retries_total", result.http_attempts - 1)

            if result.data is not None:
                self.circuit_breaker.record(True)
                put_res = self.cache.put(igme_id, ep, result.data, result.response_hash)
                if put_res.get("changed"):
                    changed_endpoints.append(ep)
                db.mark_job_complete(igme_id, ep, result.response_hash, cache_hit=False)
                endpoint_results[ep] = {
                    "status": "COMPLETE",
                    "response_hash": result.response_hash,
                    "count": len(result.data),
                }
            else:
                self.circuit_breaker.record(False)
                db.add_counter("failed_network_requests", 1)
                error = result.error or "endpoint query failed"
                db.append_job_error_history(igme_id, ep, error)
                job_now = db.get_endpoint_job(igme_id, ep)
                attempts_used = int(job_now["attempts"] or 0) + 1
                if attempts_used >= self.cfg.retry.max_attempts:
                    db.mark_job_failed(igme_id, ep, error, attempts_used)
                    db.add_failure(
                        igme_id, ep, attempts_used, "endpoint_failed", error, self._session_id
                    )
                    any_failed = True
                    endpoint_results[ep] = {
                        "status": "FAILED",
                        "response_hash": "",
                        "count": 0,
                        "error": error,
                    }
                else:
                    backoff = min(
                        self.cfg.retry.backoff_base_seconds * (2 ** (attempts_used - 1)),
                        self.cfg.retry.backoff_max_seconds,
                    )
                    next_at = (datetime.now(UTC) + timedelta(seconds=backoff)).isoformat()
                    db.mark_job_retry_wait(igme_id, ep, error, attempts_used, next_at)
                    db.add_failure(
                        igme_id, ep, attempts_used, "retry_wait", error, self._session_id
                    )
                    work_remaining = True
                    endpoint_results[ep] = {
                        "status": "RETRY_WAIT",
                        "response_hash": "",
                        "count": 0,
                        "error": error,
                    }

        if work_remaining or self._graceful_signal:
            # Record is mid-flight; leave it resumable.
            db.set_record_error(igme_id, "pending work remains", None)
            return None

        # ── derive record status ─────────────────────────────────────
        statuses = {r["status"] for r in endpoint_results.values()}
        if statuses == {EP_COMPLETE}:
            record_status = COMPLETE
        elif EP_FAILED in statuses:
            record_status = PARTIAL if EP_COMPLETE in statuses else FAILED
        else:
            record_status = PARTIAL if EP_COMPLETE in statuses else FAILED

        return self._finalize_record(
            igme_id,
            source_row,
            record_status,
            endpoint_results,
            changed_endpoints,
            any_failed,
            cache_hits_this,
            network_this,
        )

    def _finalize_record(
        self,
        igme_id: str,
        source_row: dict[str, Any],
        record_status: str,
        endpoint_results: dict[str, dict[str, Any]],
        changed_endpoints: list[str],
        any_failed: bool,
        cache_hits_this: int,
        network_this: int,
    ) -> str:
        db = self.state

        # Build raw dict from durable cache + job state
        raw: dict[str, Any] = {"igme_id": igme_id}
        for ep in ENDPOINT_NAMES:
            res = endpoint_results.get(ep, {})
            if res.get("status") == "COMPLETE":
                data, r_hash = self.cache.get(igme_id, ep)
                raw[ep] = data if data is not None else []
                raw[f"{ep}_sha256"] = r_hash
                raw[f"{ep}_count"] = len(raw[ep])
                raw[f"{ep}_cached"] = True
            else:
                raw[ep] = []
                raw[f"{ep}_sha256"] = ""
                raw[f"{ep}_count"] = 0
                raw[f"{ep}_error"] = True
        raw["_source_changed"] = len(changed_endpoints) > 0
        raw["_changed_endpoints"] = changed_endpoints

        domain = str(source_row.get("hydrogeological_unit") or "Unknown")
        tier = source_row.get("candidate_tier")
        score = source_row.get("candidate_priority_score")
        url = source_row.get("detail_url")

        flat: dict[str, Any] | None = None
        last_error: str | None = None
        quarantine_reason: str | None = None

        try:
            from aquascan.enrichment.normalize import build_normalized_record

            rec = build_normalized_record(
                igme_id=igme_id,
                domain=domain,
                candidate_tier=tier,
                candidate_score=score,
                detail_url=url,
                raw=raw,
            )
            violations = run_qa_checks(rec, source_row)
            critical = [v for v in violations if v["severity"] == QA_CRITICAL]
            for v in violations:
                db.add_qa_violation(igme_id, v["check_name"], v["severity"], v["message"])
            if critical:
                record_status = QUARANTINED
                quarantine_reason = "; ".join(v["message"] for v in critical)
            flat = flatten_normalized_record(rec, record_status, endpoint_results, source_row)
        except Exception as exc:  # normalization failure → quarantine, never silently fix
            logger.error("normalization error for %s: %s", igme_id, exc)
            last_error = str(exc)
            record_status = QUARANTINED
            quarantine_reason = f"normalization_error: {exc}"
            db.add_qa_violation(igme_id, "normalization_error", QA_CRITICAL, str(exc))
            flat = self._minimal_flat(
                igme_id,
                domain,
                tier,
                score,
                record_status,
                endpoint_results,
                source_row,
                quarantine_reason,
            )

        if flat is None:
            flat = {}

        output_json = json.dumps(flat, sort_keys=True, default=str)
        result_hash = sha256_text(output_json)
        db.complete_record(
            igme_id,
            record_status,
            result_hash,
            output_json,
            last_error=last_error or ("failed endpoints" if any_failed else None),
            quarantine_reason=quarantine_reason,
        )

        db.add_counter("records_completed", 1)
        if cache_hits_this > 0 and network_this == 0:
            db.add_counter("records_reused_from_cache", 1)
        if network_this > 0:
            db.add_counter("records_requiring_network", 1)
        return record_status

    @staticmethod
    def _minimal_flat(
        igme_id: str,
        domain: str,
        tier: Any,
        score: Any,
        record_status: str,
        endpoint_results: dict[str, dict[str, Any]],
        source_row: dict[str, Any],
        reason: str | None,
    ) -> dict[str, Any]:
        """Fallback row for records that cannot be normalized (preserved, never dropped)."""
        from aquascan.national.flatten import NO_PREFIX

        flat: dict[str, Any] = {
            "igme_id": igme_id,
            "domain": domain,
            "candidate_tier": tier,
            "candidate_priority_score": score,
            "documentary_confidence": 0.0,
            "water_presence_status": "UNKNOWN",
            "record_status": record_status,
            "quarantine_reason": reason,
            "endpoint_status_json": json.dumps(dict(endpoint_results), sort_keys=True, default=str),
        }
        for no_col in NO_PREFIX.values():
            flat[no_col] = True
        if source_row:
            for col in (
                "latitude",
                "longitude",
                "elevation_m",
                "watershed",
                "hydrogeological_unit",
                "aquifer_system",
                "province",
                "municipality",
                "point_nature",
                "total_depth_m",
                "reference_flow_l_s",
                "drilling_method",
                "construction_date",
                "pump_type",
                "water_use",
                "organization",
                "qa_flag_count",
            ):
                flat[f"source_{col}"] = source_row.get(col)
        return flat

    # ── batches / partitions ─────────────────────────────────────────

    def _maybe_complete_batch(self, batch_id: str) -> None:
        batch = self.state.batch_status(batch_id)
        if batch is None or batch["status"] == "COMPLETE":
            return
        recs = self.state.get_records_by_batch(batch_id)
        if not recs:
            return
        if all(r["status"] in TERMINAL_RECORD_STATUSES for r in recs):
            self._write_partition(batch_id, recs)

    def _write_partition(self, batch_id: str, recs: list[dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for r in sorted(recs, key=lambda x: x["igme_id"]):
            if r.get("output_json"):
                row = json.loads(r["output_json"])
                row["igme_id"] = r["igme_id"]
                rows.append(row)
            else:
                rows.append(
                    {
                        "igme_id": r["igme_id"],
                        "record_status": r["status"],
                        "quarantine_reason": r.get("quarantine_reason"),
                    }
                )
        df = pd.DataFrame(rows)
        part_path = self.cfg.interim_dir / f"part-{batch_id}.parquet"
        atomic_write_parquet(part_path, df)
        partition_hash = sha256_file(part_path)
        completion_hash = sha256_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "igme_ids": sorted(r["igme_id"] for r in recs),
                    "result_hashes": {
                        r["igme_id"]: r["result_hash"] for r in recs if r.get("result_hash")
                    },
                },
                sort_keys=True,
            )
        )
        sidecar = {
            "batch_id": batch_id,
            "record_count": len(recs),
            "partition_sha256": partition_hash,
            "completion_hash": completion_hash,
            "created_at": now_iso(),
            "engine_version": ENGINE_VERSION,
            "source_hash": self.cfg.source_hash,
        }
        atomic_write_json(part_path.with_suffix(".parquet.hash.json"), sidecar)
        self.state.mark_batch_complete(batch_id, completion_hash, str(part_path), partition_hash)
        logger.info(
            "batch %s complete: %d records → %s (%s)",
            batch_id,
            len(recs),
            part_path.name,
            partition_hash[:12],
        )

    # ── checkpoints ──────────────────────────────────────────────────

    def _maybe_checkpoint(self, force: bool = False) -> None:
        elapsed = time.monotonic() - self._last_checkpoint_time
        due = (
            force
            or self._records_since_checkpoint >= self.cfg.checkpoint.every_n_records
            or elapsed >= self.cfg.checkpoint.every_n_seconds
        )
        if not due:
            return
        self._checkpoint()
        self._records_since_checkpoint = 0
        self._last_checkpoint_time = time.monotonic()

    def _checkpoint(self, force: bool = False) -> int:
        del force  # force is accepted for API symmetry; checkpoints are idempotent
        counts = self.state.record_status_counts()
        terminal = sum(counts.get(s, 0) for s in TERMINAL_RECORD_STATUSES)
        summary = {
            "timestamp": now_iso(),
            "session_id": self._session_id,
            "engine": ENGINE_NAME,
            "software_version": ENGINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "records_total": self.state.record_count(),
            "records_complete": terminal,
            "records_terminal": terminal,
            "status_counts": counts,
            "endpoint_jobs_complete": self.state.job_status_counts().get("COMPLETE", 0),
            "failures": self.state.failure_count(),
            "cache_entries": self.cache.entry_count(),
            "logical_queries": self.state.get_counter("logical_endpoint_queries"),
            "network_requests": self.state.get_counter("network_requests"),
            "cache_hits": self.state.get_counter("endpoint_cache_hits"),
            "source_hash": self.cfg.source_hash,
        }
        cid = self.state.write_checkpoint(summary)
        self._write_progress_json()
        logger.info(
            "checkpoint %d: %d/%d records complete",
            cid,
            summary["records_complete"],
            summary["records_total"],
        )
        return cid

    # ── disk monitor ─────────────────────────────────────────────────

    def _disk_state(self) -> str:
        now = time.monotonic()
        if now - self._last_disk_check < self.cfg.disk.check_interval_seconds:
            return self._last_disk_state
        self._last_disk_check = now
        try:
            usage = shutil.disk_usage(self.cfg.state_dir)
        except OSError:
            return self._last_disk_state
        free_gb = usage.free / 1e9
        if free_gb < self.cfg.disk.critical_free_gb:
            self._last_disk_state = "DISK_CRITICAL"
        elif free_gb < self.cfg.disk.warning_free_gb:
            self._last_disk_state = "DISK_WARNING"
        else:
            self._last_disk_state = "DISK_OK"
        return self._last_disk_state

    # ── helpers ──────────────────────────────────────────────────────

    def _load_source_rows(self) -> None:
        df = pd.read_parquet(self.cfg.source)
        self._source_rows = {str(r["igme_id"]): dict(r) for r in df.to_dict("records")}

    def _configure_logging(self) -> None:
        log_path = self.cfg.state_dir / "engine.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("aquascan.national")
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    def _close_fetcher(self) -> None:
        close = getattr(self.fetcher, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _end_session(self, final_state: str) -> None:
        if self._session_id is None:
            return
        delta = self.state.session_delta(self._session_id)
        self.state.end_session(
            self._session_id,
            status=final_state,
            records_completed=delta.get("records_completed", 0),
            records_processed=delta.get("records_processed", 0),
            logical_queries=delta.get("logical_queries", 0),
            network_requests=delta.get("network_requests", 0),
            cache_hits=delta.get("cache_hits", 0),
            failures=delta.get("failed_network_requests", 0),
            retries=self.state.get_counter("retries_total") - self._session_start_retries(),
            note=self._stopping_reason or "",
        )
        self.state.set_engine_state(final_state)

    def _session_start_retries(self) -> int:
        meta = self.state.get_meta_json(f"session_start_{self._session_id}", {})
        return int(meta.get("retries_total", 0))

    # ── reports ──────────────────────────────────────────────────────

    def _write_progress_json(self) -> None:
        status = compute_status(
            self.cfg,
            self.state,
            self.cache,
            session_started_at=self._session_started_at,
            session_completed=self._completed_this_session,
        )
        status["engine_status"] = self.state.engine_state()
        atomic_write_json(self.cfg.reports_dir / "national_enrichment_progress_v1.json", status)

    def export_reports(self) -> None:
        """Export sessions + failures CSVs (atomic writes)."""
        sessions = self.state.sessions(100000)
        failures = self.state.failures(100000)
        if sessions:
            pd.DataFrame(sessions).to_csv(
                self.cfg.reports_dir / "national_enrichment_sessions_v1.csv", index=False
            )
        if failures:
            pd.DataFrame(failures).to_csv(
                self.cfg.reports_dir / "national_enrichment_failures_v1.csv", index=False
            )
