"""Persistent execution state database (SQLite, WAL mode, thread-safe).

Tracks at minimum per record:
    igme_record_id, status, attempt_count, started_at, completed_at,
    last_error, last_endpoint, source_hash, result_hash

plus endpoint-level state, batches, sessions, checkpoints, failures and
QA violations. A record can be COMPLETE on some endpoints and FAILED on
others, and resume only the missing branches.

Crash safety: SQLite WAL journaling provides atomic transactions; every
durable mutation happens inside a single transaction so a crash never
corrupts previously valid state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aquascan.national.constants import (
    PENDING,
)
from aquascan.national.io_utils import sha256_text

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    igme_id          TEXT PRIMARY KEY,
    status           TEXT NOT NULL DEFAULT 'PENDING',
    attempt_count    INTEGER NOT NULL DEFAULT 0,
    batch_id         TEXT,
    started_at       TEXT,
    completed_at     TEXT,
    last_error       TEXT,
    last_endpoint    TEXT,
    source_hash      TEXT,
    result_hash      TEXT,
    output_json      TEXT,
    quarantine_reason TEXT
);

CREATE TABLE IF NOT EXISTS endpoint_jobs (
    record_id       TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    attempts        INTEGER NOT NULL DEFAULT 0,
    network_requests INTEGER NOT NULL DEFAULT 0,
    cache_hits      INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    response_hash   TEXT,
    next_retry_at   TEXT,
    last_attempt_at TEXT,
    completed_at    TEXT,
    error_history_json TEXT,
    PRIMARY KEY (record_id, endpoint)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON endpoint_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_record ON endpoint_jobs(record_id);

CREATE TABLE IF NOT EXISTS batches (
    batch_id        TEXT PRIMARY KEY,
    record_count    INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    completion_hash TEXT,
    partition_path  TEXT,
    partition_hash  TEXT,
    created_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    host                TEXT,
    software_version    TEXT,
    source_snapshot_hash TEXT,
    status              TEXT,
    records_completed   INTEGER NOT NULL DEFAULT 0,
    records_processed   INTEGER NOT NULL DEFAULT 0,
    logical_queries     INTEGER NOT NULL DEFAULT 0,
    network_requests    INTEGER NOT NULL DEFAULT 0,
    cache_hits          INTEGER NOT NULL DEFAULT 0,
    failures            INTEGER NOT NULL DEFAULT 0,
    retries             INTEGER NOT NULL DEFAULT 0,
    note                TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT NOT NULL,
    records_complete     INTEGER NOT NULL,
    endpoint_jobs_complete INTEGER NOT NULL,
    failures             INTEGER NOT NULL,
    cache_entries        INTEGER NOT NULL,
    logical_queries      INTEGER NOT NULL,
    network_requests     INTEGER NOT NULL,
    cache_hits           INTEGER NOT NULL,
    software_version     TEXT NOT NULL,
    source_hash          TEXT NOT NULL,
    checkpoint_hash      TEXT NOT NULL,
    summary_json         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    record_id    TEXT NOT NULL,
    endpoint     TEXT NOT NULL,
    attempt      INTEGER NOT NULL,
    error_type   TEXT,
    error_message TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_record ON failures(record_id);

CREATE TABLE IF NOT EXISTS qa_violations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT NOT NULL,
    check_name TEXT NOT NULL,
    severity   TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_record ON qa_violations(record_id);
"""

# Cumulative telemetry counters stored in meta (survive restarts).
COUNTER_KEYS = (
    "logical_endpoint_queries",
    "network_requests",
    "endpoint_cache_hits",
    "http_requests_total",
    "records_reused_from_cache",
    "records_requiring_network",
    "records_completed",
    "failed_network_requests",
    "retries_total",
)

CONTROL_DIR = "control"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StateDB:
    """Thread-safe SQLite state store with WAL journaling."""

    def __init__(self, path: Path, create: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        if create:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        for key in COUNTER_KEYS:
            self.set_meta(f"counter_{key}", str(self.get_meta_int(f"counter_{key}", 0)))

    # ── low level ────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.ProgrammingError:
                pass

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]

    def _query_one(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ── meta / counters ──────────────────────────────────────────────

    def set_meta(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def get_meta_int(self, key: str, default: int = 0) -> int:
        val = self.get_meta(key)
        try:
            return int(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def get_meta_json(self, key: str, default: Any = None) -> Any:
        val = self.get_meta(key)
        if val is None:
            return default
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return default

    def add_counter(self, key: str, delta: int) -> None:
        if delta == 0:
            return
        meta_key = f"counter_{key}"
        self._execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER) + ?",
            (meta_key, str(delta), delta),
        )

    def get_counter(self, key: str) -> int:
        return self.get_meta_int(f"counter_{key}", 0)

    # ── engine state ─────────────────────────────────────────────────

    def set_engine_state(self, state: str, detail: str = "") -> None:
        self.set_meta("engine_state", state)
        if detail:
            self.set_meta("engine_state_detail", detail)
        self.set_meta("engine_state_updated_at", now_iso())

    def engine_state(self) -> str:
        return self.get_meta("engine_state") or "STOPPED"

    def set_status_summary(self, summary: dict[str, Any]) -> None:
        self.set_meta("status_summary", json.dumps(summary, sort_keys=True, default=str))

    def status_summary(self) -> dict[str, Any]:
        return self.get_meta_json("status_summary", {})

    # ── init / seeding ───────────────────────────────────────────────

    def seed_records(
        self,
        igme_ids: list[str],
        batch_ids: dict[str, str],
        source_hash: str,
    ) -> tuple[int, int]:
        """Insert PENDING records; returns (inserted, existing)."""
        rows = [
            (
                igme_id,
                PENDING,
                0,
                batch_ids.get(igme_id),
                source_hash,
            )
            for igme_id in igme_ids
        ]
        with self._lock:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO records"
                "(igme_id, status, attempt_count, batch_id, source_hash)"
                " VALUES(?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            total = len(rows)
            inserted = max(0, cur.rowcount if cur.rowcount >= 0 else total)
        existing = total - inserted
        self.set_meta("source_record_count", str(self.record_count()))
        return inserted, existing

    def create_batches(self, batch_size: int) -> int:
        """Ensure batch rows exist for every batch_id on records and assign
        batches to any records without one. Returns batches created."""
        # 1. Register batch rows for batch_ids already assigned to records.
        assigned = self._query(
            "SELECT batch_id, COUNT(*) AS n FROM records "
            "WHERE batch_id IS NOT NULL GROUP BY batch_id"
        )
        for row in assigned:
            self._execute(
                "INSERT OR IGNORE INTO batches(batch_id, record_count, status, created_at)"
                " VALUES(?, ?, 'PENDING', ?)",
                (row["batch_id"], int(row["n"]), now_iso()),
            )

        # 2. Assign new batches to records without one (numbered after existing).
        unassigned = [
            r["igme_id"]
            for r in self._query(
                "SELECT igme_id FROM records WHERE batch_id IS NULL ORDER BY igme_id"
            )
        ]
        base = self.get_meta_int("next_batch_number", 1)
        created = 0
        for i in range(0, len(unassigned), batch_size):
            chunk = unassigned[i : i + batch_size]
            batch_id = f"B{base:05d}"
            base += 1
            self._execute(
                "INSERT OR IGNORE INTO batches(batch_id, record_count, status, created_at)"
                " VALUES(?, ?, 'PENDING', ?)",
                (batch_id, len(chunk), now_iso()),
            )
            with self._lock:
                self._conn.executemany(
                    "UPDATE records SET batch_id = ? WHERE igme_id = ?",
                    [(batch_id, g) for g in chunk],
                )
                self._conn.commit()
            created += 1
        if created:
            self.set_meta("next_batch_number", str(base))
        return created

    # ── records ──────────────────────────────────────────────────────

    def record_count(self) -> int:
        row = self._query_one("SELECT COUNT(*) AS n FROM records")
        return int(row["n"])

    def record_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._query("SELECT status, COUNT(*) AS n FROM records GROUP BY status"):
            counts[row["status"]] = int(row["n"])
        return counts

    def get_record(self, igme_id: str) -> dict | None:
        return self._query_one("SELECT * FROM records WHERE igme_id = ?", (igme_id,))

    def get_records_by_batch(self, batch_id: str) -> list[dict]:
        return self._query("SELECT * FROM records WHERE batch_id = ?", (batch_id,))

    def mark_record_running(self, igme_id: str, attempt_count: int) -> None:
        self._execute(
            "UPDATE records SET status = 'RUNNING', started_at = ?, attempt_count = ? "
            "WHERE igme_id = ?",
            (now_iso(), attempt_count, igme_id),
        )

    def complete_record(
        self,
        igme_id: str,
        status: str,
        result_hash: str,
        output_json: str | None,
        last_error: str | None = None,
        quarantine_reason: str | None = None,
    ) -> None:
        self._execute(
            "UPDATE records SET status = ?, completed_at = ?, result_hash = ?, "
            "output_json = ?, last_error = ?, quarantine_reason = ? WHERE igme_id = ?",
            (status, now_iso(), result_hash, output_json, last_error, quarantine_reason, igme_id),
        )

    def set_record_error(self, igme_id: str, error: str, endpoint: str | None) -> None:
        self._execute(
            "UPDATE records SET last_error = ?, last_endpoint = ? WHERE igme_id = ?",
            (error, endpoint, igme_id),
        )

    def reset_records_to_pending(self, igme_ids: list[str]) -> int:
        if not igme_ids:
            return 0
        with self._lock:
            cur = self._conn.executemany(
                "UPDATE records SET status = 'PENDING', completed_at = NULL WHERE igme_id = ?",
                [(g,) for g in igme_ids],
            )
            self._conn.commit()
            return max(0, cur.rowcount)

    def reset_for_reprocess(self, igme_ids: list[str]) -> int:
        """Reset records AND their endpoint jobs so they are fully reprocessed
        (used to demonstrate durable-cache reuse with zero network).
        """
        if not igme_ids:
            return 0
        with self._lock:
            self._conn.executemany(
                "DELETE FROM endpoint_jobs WHERE record_id = ?", [(g,) for g in igme_ids]
            )
            cur = self._conn.executemany(
                "UPDATE records SET status = 'PENDING', completed_at = NULL, "
                "output_json = NULL, result_hash = NULL WHERE igme_id = ?",
                [(g,) for g in igme_ids],
            )
            self._conn.commit()
            return max(0, cur.rowcount)

    # ── endpoint jobs ────────────────────────────────────────────────

    def ensure_endpoint_jobs(self, record_id: str, endpoints: list[str]) -> int:
        """Create PENDING endpoint jobs if missing; returns number created."""
        existing = {
            r["endpoint"]
            for r in self._query(
                "SELECT endpoint FROM endpoint_jobs WHERE record_id = ?", (record_id,)
            )
        }
        missing = [ep for ep in endpoints if ep not in existing]
        if not missing:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO endpoint_jobs(record_id, endpoint, status)"
                " VALUES(?, ?, 'PENDING')",
                [(record_id, ep) for ep in missing],
            )
            self._conn.commit()
        return len(missing)

    def get_endpoint_job(self, record_id: str, endpoint: str) -> dict | None:
        return self._query_one(
            "SELECT * FROM endpoint_jobs WHERE record_id = ? AND endpoint = ?",
            (record_id, endpoint),
        )

    def mark_job_running(self, record_id: str, endpoint: str) -> None:
        self._execute(
            "UPDATE endpoint_jobs SET status = 'RUNNING', last_attempt_at = ? "
            "WHERE record_id = ? AND endpoint = ?",
            (now_iso(), record_id, endpoint),
        )

    def mark_job_complete(
        self,
        record_id: str,
        endpoint: str,
        response_hash: str,
        cache_hit: bool,
    ) -> None:
        self._execute(
            "UPDATE endpoint_jobs SET status = 'COMPLETE', response_hash = ?, "
            "cache_hits = cache_hits + ?, completed_at = ?, last_error = NULL, "
            "next_retry_at = NULL WHERE record_id = ? AND endpoint = ?",
            (response_hash, 1 if cache_hit else 0, now_iso(), record_id, endpoint),
        )

    def mark_job_retry_wait(
        self,
        record_id: str,
        endpoint: str,
        error: str,
        attempts: int,
        next_retry_at: str,
    ) -> None:
        self._execute(
            "UPDATE endpoint_jobs SET status = 'RETRY_WAIT', last_error = ?, "
            "attempts = ?, next_retry_at = ? WHERE record_id = ? AND endpoint = ?",
            (error, attempts, next_retry_at, record_id, endpoint),
        )

    def mark_job_failed(
        self,
        record_id: str,
        endpoint: str,
        error: str,
        attempts: int,
    ) -> None:
        self._execute(
            "UPDATE endpoint_jobs SET status = 'FAILED', last_error = ?, "
            "attempts = ?, completed_at = ?, next_retry_at = NULL "
            "WHERE record_id = ? AND endpoint = ?",
            (error, attempts, now_iso(), record_id, endpoint),
        )

    def append_job_attempt(self, record_id: str, endpoint: str, attempts: int) -> None:
        """Track network request count for the job (HTTP attempts, incl. retries)."""
        self._execute(
            "UPDATE endpoint_jobs SET network_requests = network_requests + ? "
            "WHERE record_id = ? AND endpoint = ?",
            (attempts, record_id, endpoint),
        )

    def append_job_error_history(self, record_id: str, endpoint: str, error: str) -> None:
        row = self.get_endpoint_job(record_id, endpoint)
        history: list[str] = []
        if row and row.get("error_history_json"):
            try:
                history = json.loads(row["error_history_json"])
            except json.JSONDecodeError:
                history = []
        history.append(error)
        self._execute(
            "UPDATE endpoint_jobs SET error_history_json = ? WHERE record_id = ? AND endpoint = ?",
            (json.dumps(history[-100:]), record_id, endpoint),
        )

    def job_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._query("SELECT status, COUNT(*) AS n FROM endpoint_jobs GROUP BY status"):
            counts[row["status"]] = int(row["n"])
        return counts

    def pending_job_count(self, record_id: str) -> int:
        row = self._query_one(
            "SELECT COUNT(*) AS n FROM endpoint_jobs "
            "WHERE record_id = ? AND status IN ('PENDING', 'RUNNING', 'RETRY_WAIT')",
            (record_id,),
        )
        return int(row["n"])

    def retry_wait_jobs_due(self, record_id: str, now: str) -> int:
        row = self._query_one(
            "SELECT COUNT(*) AS n FROM endpoint_jobs "
            "WHERE record_id = ? AND status = 'RETRY_WAIT' AND next_retry_at <= ?",
            (record_id, now),
        )
        return int(row["n"])

    def reset_failed_endpoints(
        self,
        endpoint: str | None = None,
        record_id: str | None = None,
        max_attempts: int | None = None,
    ) -> int:
        """Reset FAILED/RETRY_WAIT endpoint jobs to PENDING (retry-failed command).

        Original errors are preserved in error_history_json and the failures
        table; this only re-opens the job.
        """
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
        with self._lock:
            cur = self._conn.execute(
                "UPDATE endpoint_jobs SET status = 'PENDING', next_retry_at = NULL "
                f"WHERE {' AND '.join(clauses)}",
                tuple(params),
            )
            self._conn.commit()
            return max(0, cur.rowcount)

    # ── eligibility ──────────────────────────────────────────────────

    def eligible_records(self, now: str, resume_retries_failed: bool = False) -> list[str]:
        """Records with work remaining, ordered deterministically by igme_id.

        A record is eligible if it has:
        - no endpoint jobs yet (fresh PENDING record), or
        - at least one endpoint job PENDING, or
        - at least one endpoint job RETRY_WAIT whose backoff has elapsed.
        FAILED records are only eligible when resume_retries_failed is set.
        """
        rows = self._query(
            "SELECT r.igme_id, r.status FROM records r "
            "WHERE r.status IN ('PENDING', 'PARTIAL', 'RETRY_WAIT', 'RUNNING') "
            "   OR (r.status = 'FAILED' AND ? = 1) "
            "ORDER BY r.igme_id",
            (1 if resume_retries_failed else 0,),
        )
        eligible: list[str] = []
        for row in rows:
            rec = row["igme_id"]
            jobs = self._query(
                "SELECT status, next_retry_at FROM endpoint_jobs WHERE record_id = ?",
                (rec,),
            )
            if not jobs:
                eligible.append(rec)
                continue
            for j in jobs:
                # PENDING and RUNNING jobs are pending work: RUNNING jobs left
                # by a crashed process are recovered by reprocessing them
                # (records in flight in a live process are never re-scheduled).
                if j["status"] in ("PENDING", "RUNNING"):
                    eligible.append(rec)
                    break
                if j["status"] == "RETRY_WAIT" and j["next_retry_at"] is not None:
                    if j["next_retry_at"] <= now:
                        eligible.append(rec)
                        break
            else:
                # All endpoint jobs are terminal. A record left PENDING/RUNNING
                # after the last job completed (crash or signal before the
                # record row was updated) still needs finalization. Without
                # this, the engine can declare COMPLETE with stuck records.
                if all(j["status"] == "COMPLETE" for j in jobs):
                    eligible.append(rec)

        return eligible

    # ── stale recovery ───────────────────────────────────────────────

    def recover_stale_running(
        self,
        stale_before: str,
        endpoints: list[str],
    ) -> dict[str, int]:
        """Recover jobs/records left RUNNING by a crashed process.

        A job is stale when its last_attempt_at predates *stale_before*.
        Returns counts of recovered jobs and records.
        """
        jobs = self._query(
            "SELECT record_id, endpoint FROM endpoint_jobs "
            "WHERE status = 'RUNNING' AND (last_attempt_at IS NULL OR last_attempt_at <= ?)",
            (stale_before,),
        )
        with self._lock:
            for j in jobs:
                self._conn.execute(
                    "UPDATE endpoint_jobs SET status = 'PENDING' "
                    "WHERE record_id = ? AND endpoint = ?",
                    (j["record_id"], j["endpoint"]),
                )
            self._conn.commit()
        recovered_jobs = len(jobs)

        # Recover records stuck RUNNING with no running jobs left
        recs = self._query("SELECT igme_id FROM records WHERE status = 'RUNNING'")
        recovered_records = 0
        for r in recs:
            pending = self.pending_job_count(r["igme_id"])
            # Only touch records that have jobs; leave pure-PENDING records alone
            has_jobs = self._query_one(
                "SELECT COUNT(*) AS n FROM endpoint_jobs WHERE record_id = ?", (r["igme_id"],)
            )
            if int(has_jobs["n"]) == 0 or pending > 0:
                continue
            self._execute(
                "UPDATE records SET status = 'PENDING' WHERE igme_id = ?", (r["igme_id"],)
            )
            recovered_records += 1
        return {"jobs": recovered_jobs, "records": recovered_records}

    def stale_running_jobs(self, stale_before: str) -> list[dict]:
        return self._query(
            "SELECT record_id, endpoint, last_attempt_at FROM endpoint_jobs "
            "WHERE status = 'RUNNING' AND (last_attempt_at IS NULL OR last_attempt_at <= ?)",
            (stale_before,),
        )

    # ── batches ──────────────────────────────────────────────────────

    def batch_records(self, batch_id: str) -> list[dict]:
        return self._query("SELECT * FROM records WHERE batch_id = ?", (batch_id,))

    def batch_status(self, batch_id: str) -> dict | None:
        return self._query_one("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))

    def mark_batch_complete(
        self,
        batch_id: str,
        completion_hash: str,
        partition_path: str,
        partition_hash: str,
    ) -> None:
        self._execute(
            "UPDATE batches SET status = 'COMPLETE', completion_hash = ?, "
            "partition_path = ?, partition_hash = ?, completed_at = ? "
            "WHERE batch_id = ?",
            (completion_hash, partition_path, partition_hash, now_iso(), batch_id),
        )

    def incomplete_batches(self) -> list[str]:
        return [
            r["batch_id"]
            for r in self._query("SELECT batch_id FROM batches WHERE status != 'COMPLETE'")
        ]

    # ── sessions ─────────────────────────────────────────────────────

    def start_session(self, host: str, software_version: str, source_hash: str) -> str:
        session_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO sessions(session_id, started_at, host, software_version,"
            " source_snapshot_hash, status)"
            " VALUES(?, ?, ?, ?, ?, 'RUNNING')",
            (session_id, now_iso(), host, software_version, source_hash),
        )
        return session_id

    def end_session(self, session_id: str, **stats: Any) -> None:
        fields = ", ".join(f"{k} = ?" for k in stats)
        # Placeholder order: <stat values> + ended_at + WHERE session_id
        params = list(stats.values()) + [now_iso(), session_id]
        self._execute(
            f"UPDATE sessions SET {fields}, ended_at = ? WHERE session_id = ?",
            tuple(params),
        )

    def sessions(self, limit: int = 50) -> list[dict]:
        return self._query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))

    def session_delta(
        self,
        session_id: str,
    ) -> dict[str, int]:
        """Per-session counter deltas (start-of-session snapshot vs current)."""
        key = f"session_start_{session_id}"
        start = self.get_meta_json(key, None)
        if start is None:
            return {}
        out: dict[str, int] = {}
        for k in (
            "records_completed",
            "logical_queries",
            "network_requests",
            "cache_hits",
            "records_processed",
            "failed_network_requests",
        ):
            out[k] = self.get_counter(k) - int(start.get(k, 0))
        return out

    def snapshot_session_counters(self, session_id: str) -> None:
        self.set_meta(
            f"session_start_{session_id}",
            json.dumps(
                {
                    "records_completed": self.get_counter("records_completed"),
                    "logical_queries": self.get_counter("logical_endpoint_queries"),
                    "network_requests": self.get_counter("network_requests"),
                    "cache_hits": self.get_counter("endpoint_cache_hits"),
                    "records_processed": self.get_counter("records_requiring_network")
                    + self.get_counter("records_reused_from_cache"),
                    "failed_network_requests": self.get_counter("failed_network_requests"),
                    "retries_total": self.get_counter("retries_total"),
                },
                sort_keys=True,
            ),
        )

    # ── checkpoints ──────────────────────────────────────────────────

    def write_checkpoint(self, summary: dict[str, Any]) -> int:
        """Write a checkpoint row; the row itself is hashable/verifiable."""
        checkpoint_hash = sha256_text(
            json.dumps(summary, sort_keys=True, ensure_ascii=False, default=str)
        )
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO checkpoints(timestamp, records_complete,"
                " endpoint_jobs_complete, failures, cache_entries, logical_queries,"
                " network_requests, cache_hits, software_version, source_hash,"
                " checkpoint_hash, summary_json)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    summary.get("timestamp", now_iso()),
                    int(summary.get("records_complete", 0)),
                    int(summary.get("endpoint_jobs_complete", 0)),
                    int(summary.get("failures", 0)),
                    int(summary.get("cache_entries", 0)),
                    int(summary.get("logical_queries", 0)),
                    int(summary.get("network_requests", 0)),
                    int(summary.get("cache_hits", 0)),
                    str(summary.get("software_version", "")),
                    str(summary.get("source_hash", "")),
                    checkpoint_hash,
                    json.dumps(summary, sort_keys=True, default=str),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def latest_checkpoint(self) -> dict | None:
        return self._query_one("SELECT * FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1")

    def checkpoint_count(self) -> int:
        row = self._query_one("SELECT COUNT(*) AS n FROM checkpoints")
        return int(row["n"])

    def verify_checkpoint(self, checkpoint_id: int) -> bool:
        """Verify a checkpoint: hash matches content AND columns match summary."""
        row = self._query_one("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        if row is None:
            return False
        summary = json.loads(row["summary_json"])
        recomputed = sha256_text(
            json.dumps(summary, sort_keys=True, ensure_ascii=False, default=str)
        )
        if recomputed != row["checkpoint_hash"]:
            return False
        # Denormalized columns must agree with the summary content.
        for col, key in (
            ("records_complete", "records_complete"),
            ("endpoint_jobs_complete", "endpoint_jobs_complete"),
            ("failures", "failures"),
            ("cache_entries", "cache_entries"),
            ("logical_queries", "logical_queries"),
            ("network_requests", "network_requests"),
            ("cache_hits", "cache_hits"),
            ("software_version", "software_version"),
            ("source_hash", "source_hash"),
        ):
            if str(row[col]) != str(summary.get(key, "")):
                return False
        return True

    # ── failures / QA ────────────────────────────────────────────────

    def add_failure(
        self,
        record_id: str,
        endpoint: str,
        attempt: int,
        error_type: str,
        error_message: str,
        session_id: str | None = None,
    ) -> None:
        self._execute(
            "INSERT INTO failures(session_id, record_id, endpoint, attempt,"
            " error_type, error_message, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (session_id, record_id, endpoint, attempt, error_type, error_message, now_iso()),
        )

    def failures(self, limit: int = 500) -> list[dict]:
        return self._query("SELECT * FROM failures ORDER BY id DESC LIMIT ?", (limit,))

    def failure_count(self) -> int:
        row = self._query_one("SELECT COUNT(*) AS n FROM failures")
        return int(row["n"])

    def add_qa_violation(
        self, record_id: str, check_name: str, severity: str, message: str
    ) -> None:
        self._execute(
            "INSERT INTO qa_violations(record_id, check_name, severity, message, created_at)"
            " VALUES(?, ?, ?, ?, ?)",
            (record_id, check_name, severity, message, now_iso()),
        )

    def qa_violations(self, record_id: str | None = None) -> list[dict]:
        if record_id:
            return self._query("SELECT * FROM qa_violations WHERE record_id = ?", (record_id,))
        return self._query("SELECT * FROM qa_violations ORDER BY id")

    def qa_counts(self) -> dict[str, int]:
        counts = {"WARNING": 0, "CRITICAL": 0}
        for row in self._query(
            "SELECT severity, COUNT(*) AS n FROM qa_violations GROUP BY severity"
        ):
            counts[row["severity"]] = int(row["n"])
        return counts

    # ── integrity ────────────────────────────────────────────────────

    def integrity_check(self) -> list[str]:
        rows = self._query("PRAGMA integrity_check")
        return [r["integrity_check"] for r in rows]

    def telemetry(self) -> dict[str, int]:
        return {
            "logical_endpoint_queries": self.get_counter("logical_endpoint_queries"),
            "network_requests": self.get_counter("network_requests"),
            "endpoint_cache_hits": self.get_counter("endpoint_cache_hits"),
            "http_requests_total": self.get_counter("http_requests_total"),
            "records_reused_from_cache": self.get_counter("records_reused_from_cache"),
            "records_requiring_network": self.get_counter("records_requiring_network"),
            "records_completed": self.get_counter("records_completed"),
            "failed_network_requests": self.get_counter("failed_network_requests"),
            "retries_total": self.get_counter("retries_total"),
        }

    def telemetry_invariant_holds(self) -> bool:
        t = self.telemetry()
        return t["logical_endpoint_queries"] == t["network_requests"] + t["endpoint_cache_hits"]
