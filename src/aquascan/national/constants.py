"""Shared constants for the National Enrichment Engine."""

from __future__ import annotations

# Endpoint order follows the validated Gate 4/5 pipeline (ENDPOINTS dict).
ENDPOINT_NAMES: tuple[str, ...] = (
    "piezometry",
    "lithology",
    "pumping_tests",
    "hydrometry",
    "chemistry",
    "technical",
    "documents",
)

# Record-level statuses
PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
PARTIAL = "PARTIAL"
FAILED = "FAILED"
RETRY_WAIT = "RETRY_WAIT"
SKIPPED = "SKIPPED"
QUARANTINED = "QUARANTINED"

TERMINAL_RECORD_STATUSES: frozenset[str] = frozenset(
    {COMPLETE, PARTIAL, FAILED, QUARANTINED, SKIPPED}
)

# Endpoint-job statuses
EP_PENDING = "PENDING"
EP_RUNNING = "RUNNING"
EP_COMPLETE = "COMPLETE"
EP_FAILED = "FAILED"
EP_RETRY_WAIT = "RETRY_WAIT"

# Engine states
ENGINE_STOPPED = "STOPPED"
ENGINE_RUNNING = "RUNNING"
ENGINE_PAUSED = "PAUSED"
ENGINE_COMPLETE = "COMPLETE"
ENGINE_FAILED = "FAILED"
ENGINE_NETWORK_DEGRADED = "NETWORK_DEGRADED"
ENGINE_DISK_CRITICAL = "DISK_CRITICAL"

# Control request filenames (written by pause/stop CLI, polled by the engine)
CONTROL_PAUSE = "pause.request"
CONTROL_STOP = "stop.request"

# QA severities
QA_WARNING = "WARNING"
QA_CRITICAL = "CRITICAL"

__all__ = [
    "ENDPOINT_NAMES",
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "PARTIAL",
    "FAILED",
    "RETRY_WAIT",
    "SKIPPED",
    "QUARANTINED",
    "TERMINAL_RECORD_STATUSES",
    "EP_PENDING",
    "EP_RUNNING",
    "EP_COMPLETE",
    "EP_FAILED",
    "EP_RETRY_WAIT",
    "ENGINE_STOPPED",
    "ENGINE_RUNNING",
    "ENGINE_PAUSED",
    "ENGINE_COMPLETE",
    "ENGINE_FAILED",
    "ENGINE_NETWORK_DEGRADED",
    "ENGINE_DISK_CRITICAL",
    "CONTROL_PAUSE",
    "CONTROL_STOP",
    "QA_WARNING",
    "QA_CRITICAL",
]
