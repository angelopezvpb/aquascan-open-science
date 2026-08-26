"""Gate 4: IGME bdaguas enrichment API client.

Implements resilient HTTP calls with:
- timeout
- retry with bounded exponential backoff
- rate limiting
- deterministic request logging
- HTTP/error capture
- resumability
- cache reuse
- SHA-256 hashing

A failed endpoint request must not destroy or invalidate data already
retrieved from other endpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_BASE = "http://info.igme.es/bdaguas/api/Agma"
HEADERS = {"Content-Type": "application/json; charset=utf-8"}

ENDPOINTS: dict[str, dict[str, Any]] = {
    "piezometry": {"path": "/QueryAGPZ", "params": {"FechaIni": "", "FechaFin": ""}},
    "lithology": {"path": "/QueryAGLI", "params": {}},
    "pumping_tests": {"path": "/QueryAGEB", "params": {"FechaIni": "", "FechaFin": ""}},
    "hydrometry": {"path": "/QueryAGHD", "params": {"FechaIni": "", "FechaFin": ""}},
    "chemistry": {"path": "/QueryAGAQ", "params": {"FechaIni": "", "FechaFin": ""}},
    "technical": {"path": "/QueryAGTE", "params": {}},
    "documents": {"path": "/QueryDocsByID", "params": {}},
}

RECENT_YEAR_THRESHOLD = 5
MULTI_DECADE_MIN_SPAN_YEARS = 20
MULTI_DECADE_MIN_MEASUREMENTS = 5


def sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute SHA-256 hash of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(raw)


@dataclass
class APIRequestMetrics:
    """Per-call metrics for operational measurement (Gate 4, step 13)."""

    endpoint: str = ""
    igme_id: str = ""
    success: bool = False
    status_code: int = 0
    elapsed_ms: float = 0.0
    response_bytes: int = 0
    retry_count: int = 0
    cache_hit: bool = False
    error_message: str = ""


@dataclass
class EnrichmentAPIClient:
    """Client for IGME bdaguas enrichment endpoints.

    Every API request supports:
    - timeout
    - retry with bounded exponential backoff
    - rate limiting
    - deterministic request logging
    - HTTP/error capture
    - resumability
    - cache reuse
    - SHA-256 hashing

    Telemetry architecture:
    - APIRequestMetrics (self.metrics): one per actual HTTP request.
      Do NOT create fake metrics for cache hits.
    - Explicit counters (below): track cache hits, logical queries,
      and per-record reuse independently of network metrics.

    Invariant:
      logical_endpoint_queries == network_requests + endpoint_cache_hits
    """

    timeout: float = 30.0
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    requests_per_second: float = 4.0
    session: requests.Session | None = field(default=None, repr=False)

    # Network request metrics (one per actual HTTP call)
    metrics: list[APIRequestMetrics] = field(default_factory=list)

    # Explicit counters for cache / logical query telemetry
    logical_endpoint_queries: int = field(default=0, init=False)
    endpoint_cache_hits: int = field(default=0, init=False)
    records_processed: int = field(default=0, init=False)
    records_reused_from_cache: int = field(default=0, init=False)
    records_requiring_network: int = field(default=0, init=False)
    failed_network_requests: int = field(default=0, init=False)

    _last_request: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(HEADERS)

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.monotonic() - self._last_request
        min_interval = 1.0 / self.requests_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()

    def _post(self, endpoint: str, body: dict[str, Any]) -> requests.Response:
        """Issue a POST request with timeout."""
        url = f"{API_BASE}{endpoint}"
        return requests.post(
            url,
            data=json.dumps(body),
            headers=HEADERS,
            timeout=self.timeout,
        )

    def _retryable_post(
        self, endpoint: str, body: dict[str, Any], metric: APIRequestMetrics
    ) -> requests.Response | None:
        """POST with retry and bounded exponential backoff."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            metric.retry_count = attempt

            try:
                resp = self._post(endpoint, body)
                metric.status_code = resp.status_code
                metric.elapsed_ms = resp.elapsed.total_seconds() * 1000
                metric.response_bytes = len(resp.content)

                if resp.status_code == 200:
                    metric.success = True
                    return resp

                if 400 <= resp.status_code < 500:
                    # Client error — don't retry
                    metric.error_message = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return resp

                # Server error — retry
                metric.error_message = f"HTTP {resp.status_code}"
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    time.sleep(delay)

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exception = e
                metric.error_message = str(e)[:200]
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    time.sleep(delay)

        if last_exception:
            metric.error_message = str(last_exception)[:200]
        return None

    def query_endpoint(
        self,
        igme_id: str,
        endpoint_name: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Query a single endpoint for a single IGME point.

        Args:
            igme_id: IGME point identifier.
            endpoint_name: Internal endpoint key (e.g., 'piezometry').

        Returns:
            Tuple of (data_list_or_None, sha256_hash).
            Returns (None, "") if the request failed.
        """
        ep_config = ENDPOINTS[endpoint_name]
        path = ep_config["path"]
        body: dict[str, Any] = {"Ids": [igme_id]}
        body.update(ep_config["params"])

        metric = APIRequestMetrics(endpoint=endpoint_name, igme_id=igme_id)
        t0 = time.monotonic()

        try:
            resp = self._retryable_post(path, body, metric)
        finally:
            metric.elapsed_ms = (time.monotonic() - t0) * 1000
            self.metrics.append(metric)

        if resp is None or resp.status_code != 200:
            return None, ""

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            metric.success = False
            metric.error_message = "JSON parse error"
            return None, ""

        if not isinstance(data, list):
            metric.success = True
            # Some endpoints may return non-list on error
            return [], sha256_json(data)

        result_hash = sha256_json(data)
        return data, result_hash

    def query_all(
        self,
        igme_id: str,
        cache: Any | None = None,
    ) -> dict[str, Any]:
        """Query all enrichment endpoints for a single IGME point.

        A failed endpoint request must not destroy or invalidate data
        already retrieved from other endpoints.

        Args:
            igme_id: IGME point identifier.
            cache: Optional ResponseCache instance for caching.

        Returns:
            Dict with raw responses keyed by endpoint name.
        """
        self.records_processed += 1

        raw: dict[str, Any] = {
            "igme_id": igme_id,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }

        n_cached = 0
        n_network = 0

        for ep_name in ENDPOINTS:
            self.logical_endpoint_queries += 1

            # Check cache first
            cached_data = None
            cached_hash = None
            if cache is not None:
                cached_data, cached_hash = cache.get(igme_id, ep_name)

            if cached_data is not None and cached_hash:
                self.endpoint_cache_hits += 1
                n_cached += 1
                raw[ep_name] = cached_data
                raw[f"{ep_name}_sha256"] = cached_hash
                raw[f"{ep_name}_count"] = len(cached_data)
                raw[f"{ep_name}_cached"] = True
                continue

            # Query API
            n_network += 1
            data, data_hash = self.query_endpoint(igme_id, ep_name)

            if data is not None:
                raw[ep_name] = data
                raw[f"{ep_name}_sha256"] = data_hash
                raw[f"{ep_name}_count"] = len(data)
                raw[f"{ep_name}_cached"] = False

                # Store in cache
                if cache is not None:
                    cache.put(igme_id, ep_name, data, data_hash)
            else:
                self.failed_network_requests += 1
                raw[ep_name] = []
                raw[f"{ep_name}_sha256"] = ""
                raw[f"{ep_name}_count"] = 0
                raw[f"{ep_name}_error"] = True

        # Per-record classification
        if n_cached == len(ENDPOINTS):
            self.records_reused_from_cache += 1
        elif n_network > 0:
            self.records_requiring_network += 1

        # Persist cache once per record (resumability without O(n^2) writes)
        if cache is not None:
            cache.save()

        return raw

    @property
    def total_requests(self) -> int:
        return len(self.metrics)

    @property
    def successful_requests(self) -> int:
        return sum(1 for m in self.metrics if m.success)

    @property
    def failed_requests(self) -> int:
        return sum(1 for m in self.metrics if not m.success and m.status_code > 0)

    def get_latency_stats(self) -> dict[str, float]:
        """Return latency statistics across all requests."""
        if not self.metrics:
            return {}
        lats = sorted(m.elapsed_ms for m in self.metrics if m.elapsed_ms > 0)
        if not lats:
            return {}
        return {
            "mean_ms": sum(lats) / len(lats),
            "median_ms": lats[len(lats) // 2],
            "p95_ms": lats[int(len(lats) * 0.95)] if len(lats) >= 20 else lats[-1],
        }

    def get_metrics_summary(self) -> dict[str, Any]:
        """Generate API operational metrics (Gate 5 corrected telemetry).

        Distinguishes:
        - logical_endpoint_queries: total endpoint lookups attempted
        - network_requests: actual HTTP calls (len(self.metrics))
        - endpoint_cache_hits: cache retrievals (no network)
        - records_fully_reused_from_cache: records needing 0 network calls
        - records_requiring_network: records needing >=1 network call
        - failed_network_requests: network calls that failed

        Invariant:
          logical_endpoint_queries == network_requests + endpoint_cache_hits
        """
        total_retries = sum(m.retry_count for m in self.metrics)
        total_bytes = sum(m.response_bytes for m in self.metrics)
        latency = self.get_latency_stats()

        # Endpoint-specific failure rates (network only)
        endpoint_stats: dict[str, dict[str, Any]] = {}
        for ep_name in ENDPOINTS:
            ep_metrics = [m for m in self.metrics if m.endpoint == ep_name]
            n = len(ep_metrics)
            if n == 0:
                continue
            n_fail = sum(1 for m in ep_metrics if not m.success)
            endpoint_stats[ep_name] = {
                "network_total": n,
                "network_failed": n_fail,
                "failure_rate": round(n_fail / n, 4) if n > 0 else 0.0,
            }

        network_requests = len(self.metrics)

        return {
            # Logical vs. physical breakdown
            "logical_endpoint_queries": self.logical_endpoint_queries,
            "network_requests": network_requests,
            "endpoint_cache_hits": self.endpoint_cache_hits,
            # Per-record classification
            "records_processed": self.records_processed,
            "records_fully_reused_from_cache": self.records_reused_from_cache,
            "records_requiring_network": self.records_requiring_network,
            # Legacy-compatible network metrics
            "total_requests": network_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failed_network_requests": self.failed_network_requests,
            "retries": total_retries,
            "cache_hits": self.endpoint_cache_hits,
            # Invariant check
            "telemetry_invariant_holds": (
                self.logical_endpoint_queries == network_requests + self.endpoint_cache_hits
            ),
            # Latency (network only — cache has no latency)
            "mean_latency_ms": latency.get("mean_ms", 0),
            "median_latency_ms": latency.get("median_ms", 0),
            "p95_latency_ms": latency.get("p95_ms", 0),
            "total_elapsed_ms": sum(m.elapsed_ms for m in self.metrics),
            "response_bytes": total_bytes,
            "endpoint_failure_rates": endpoint_stats,
        }
