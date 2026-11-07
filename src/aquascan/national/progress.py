"""Terminal progress display and status computation.

Produces the compact progress UI (Gate 14) and the structured status used by
`aquascan national status` (Gate 13). Status must work even when the engine
is not running (it reads durable state only).
"""

from __future__ import annotations

import sys
import time
from typing import Any


class ProgressDisplay:
    """Clean single-line terminal progress rendering, throttled.

    Avoids excessive per-request logging. When stdout is not a TTY the
    display is a no-op (the engine logs periodic summary lines instead).
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = sys.stdout.isatty()
        self.enabled = enabled
        self._last_render = 0.0
        self._last_line_len = 0

    def render(self, status: dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_render < 0.4:
            return
        self._last_render = now

        total = max(1, status.get("total", 1))
        done = status.get("complete", 0)
        pct = 100.0 * done / total
        bar_len = 24
        filled = int(bar_len * pct / 100.0)
        bar = "█" * filled + "-" * (bar_len - filled)

        eta = status.get("eta_seconds")
        eta_str = _fmt_duration(eta) if eta is not None else "--:--"

        line = (
            f"\rAquaScan National Dataset v1  [{bar}] {pct:5.1f}%  "
            f"{done:>10,} / {total:,} records   "
            f"Net: {status.get('network_requests', 0):>9,}   "
            f"Cache: {status.get('cache_hits', 0):>9,}   "
            f"Failed: {status.get('failed', 0):>6,}   "
            f"Rate: {status.get('records_per_second', 0):.2f} rec/s   "
            f"ETA: {eta_str}   "
            f"[Ctrl+C for safe checkpoint and exit]"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        self._last_line_len = len(line)

    def close(self) -> None:
        if self.enabled and self._last_line_len:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_line_len = 0


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def compute_status(
    cfg,
    db,
    cache=None,
    *,
    session_started_at: float | None = None,
    session_completed: int = 0,
) -> dict[str, Any]:
    """Build the full status dict (Gate 13) from durable state only."""
    counts = db.record_status_counts()
    total = db.record_count()
    complete = counts.get("COMPLETE", 0)
    partial = counts.get("PARTIAL", 0)
    pending = counts.get("PENDING", 0) + counts.get("RUNNING", 0) + counts.get("RETRY_WAIT", 0)
    failed = counts.get("FAILED", 0) + counts.get("QUARANTINED", 0)

    job_counts = db.job_status_counts()
    telemetry = db.telemetry()

    elapsed_runtime = 0.0
    if session_started_at is not None:
        elapsed_runtime = time.monotonic() - session_started_at

    records_per_second = 0.0
    if elapsed_runtime > 0 and session_completed > 0:
        records_per_second = session_completed / elapsed_runtime

    remaining = max(0, total - complete - partial - failed)
    eta_seconds = None
    if records_per_second > 0 and remaining > 0:
        eta_seconds = remaining / records_per_second

    last_checkpoint = db.latest_checkpoint()
    engine_state = db.engine_state()

    cache_stats = {}
    if cache is not None:
        try:
            cache_stats = cache.cache_stats()
        except OSError:
            cache_stats = {}

    requests_per_second = 0.0
    if elapsed_runtime > 0:
        requests_per_second = telemetry["network_requests"] / max(elapsed_runtime, 1e-9)

    return {
        "engine": "National enrichment v1",
        "engine_status": engine_state,
        "total": total,
        "complete": complete,
        "partial": partial,
        "pending": pending,
        "failed": failed,
        "running": counts.get("RUNNING", 0),
        "status_counts": counts,
        "endpoint_counts": job_counts,
        "logical_endpoint_queries": telemetry["logical_endpoint_queries"],
        "network_requests": telemetry["network_requests"],
        "cache_hits": telemetry["endpoint_cache_hits"],
        "records_reused_from_cache": telemetry["records_reused_from_cache"],
        "records_requiring_network": telemetry["records_requiring_network"],
        "failed_network_requests": telemetry["failed_network_requests"],
        "http_requests_total": telemetry["http_requests_total"],
        "records_per_second": round(records_per_second, 2),
        "requests_per_second": round(requests_per_second, 2),
        "elapsed_runtime_seconds": round(elapsed_runtime, 1),
        "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
        "eta_estimate_note": (
            "ETA is an estimate from observed performance; it updates as the run progresses."
        ),
        "last_checkpoint": {
            "id": last_checkpoint["checkpoint_id"] if last_checkpoint else None,
            "timestamp": last_checkpoint["timestamp"] if last_checkpoint else None,
            "hash": last_checkpoint["checkpoint_hash"] if last_checkpoint else None,
        },
        "cache_stats": cache_stats,
        "source_record_count": cfg.source_record_count,
        "source_hash": cfg.source_hash,
    }


def render_status_text(status: dict[str, Any]) -> str:
    """Human-readable status block (Gate 13 example format)."""
    lines = [
        "=" * 64,
        "AquaScan National Enrichment v1 — STATUS",
        "=" * 64,
        f"Engine status:  {status['engine_status']}",
        "",
        f"Total:      {status['total']:>10,}",
        f"Complete:   {status['complete']:>10,}",
        f"Partial:    {status['partial']:>10,}",
        f"Pending:    {status['pending']:>10,}",
        f"Failed:     {status['failed']:>10,}",
        f"Running:    {status['running']:>10,}",
        "",
        "Endpoint jobs:",
    ]
    for ep, n in sorted(status["endpoint_counts"].items()):
        lines.append(f"  {ep:<14} {n:>10,}")
    lines += [
        "",
        f"Logical endpoint queries: {status['logical_endpoint_queries']:>12,}",
        f"Network requests:         {status['network_requests']:>12,}",
        f"Cache hits:               {status['cache_hits']:>12,}",
        f"Records reused from cache: {status['records_reused_from_cache']:>10,}",
        f"Records needing network:  {status['records_requiring_network']:>12,}",
        f"Failed network requests:  {status['failed_network_requests']:>12,}",
        f"HTTP requests total:      {status['http_requests_total']:>12,}",
        "",
        f"Records/second: {status['records_per_second']:.2f}",
        f"Requests/second: {status['requests_per_second']:.2f}",
        f"Elapsed runtime: {_fmt_duration(status['elapsed_runtime_seconds'])}",
    ]
    if status.get("eta_seconds"):
        lines.append(f"Estimated remaining: {_fmt_duration(status['eta_seconds'])} (estimate)")
    last_cp = status["last_checkpoint"]
    lines.append(
        f"Last checkpoint: id={last_cp['id']} at {last_cp['timestamp']} "
        f"hash={str(last_cp['hash'])[:12]}"
    )
    if status.get("eta_estimate_note"):
        lines.append(f"Note: {status['eta_estimate_note']}")
    lines.append("=" * 64)
    return "\n".join(lines)
