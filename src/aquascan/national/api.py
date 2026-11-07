"""Network plumbing for the National Enrichment Engine.

- RateLimiter: thread-safe global minimum-interval rate limiting so the
  engine stays respectful of IGME infrastructure regardless of concurrency.
- CircuitBreaker: pauses network acquisition during systemic failures.
- LiveFetcher: wraps the validated Gate 4/5 EnrichmentAPIClient.
- SimulatedFetcher: deterministic offline API for dry runs / tests ONLY.
  Synthetic data never validates real-world performance.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from aquascan.national.config import CircuitBreakerConfig, RetryConfig


@dataclass
class FetchResult:
    """Result of one endpoint lookup."""

    data: list[dict[str, Any]] | None
    response_hash: str
    http_attempts: int = 1  # actual HTTP calls performed (incl. internal retries)
    error: str | None = None


class RateLimiter:
    """Thread-safe minimum-interval rate limiter (global across workers)."""

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._last = now


class CircuitBreaker:
    """Pauses network acquisition when a recent window shows systemic failure."""

    def __init__(self, cfg: CircuitBreakerConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._recent: list[bool] = []  # True = success
        self._open = False
        self._open_at = 0.0
        self._trip_count = 0

    def record(self, success: bool) -> None:
        with self._lock:
            self._recent.append(success)
            if len(self._recent) > self.cfg.window_size:
                self._recent.pop(0)
            # Evaluate only once the recent window is reasonably populated so
            # a short burst of failures does not trip the breaker.
            if len(self._recent) >= min(self.cfg.window_size, 20):
                failures = sum(1 for ok in self._recent if not ok)
                ratio = failures / len(self._recent)
                if ratio > self.cfg.failure_ratio_threshold and not self._open:
                    self._open = True
                    self._open_at = time.monotonic()
                    self._trip_count += 1

    def is_open(self) -> bool:
        with self._lock:
            if not self._open:
                return False
            if time.monotonic() - self._open_at >= self.cfg.cooldown_seconds:
                self._open = False
                self._recent.clear()
            return self._open

    def state(self) -> dict[str, Any]:
        with self._lock:
            failures = sum(1 for ok in self._recent if not ok)
            return {
                "open": self._open,
                "window": len(self._recent),
                "window_failures": failures,
                "window_failure_ratio": round(failures / len(self._recent), 4)
                if self._recent
                else 0.0,
                "trip_count": self._trip_count,
                "cooldown_remaining_seconds": round(
                    max(0.0, self.cfg.cooldown_seconds - (time.monotonic() - self._open_at)), 2
                )
                if self._open
                else 0.0,
            }


class LiveFetcher:
    """Production fetcher wrapping the validated Gate 4/5 API client.

    The underlying client performs bounded retries with exponential backoff
    and records per-request metrics; the engine adds endpoint-level state,
    its own retry policy and the global rate limiter on top.
    """

    def __init__(
        self,
        retry: RetryConfig,
        rate_limiter: RateLimiter,
    ) -> None:
        from aquascan.enrichment.api import EnrichmentAPIClient

        self._client_factory = lambda: EnrichmentAPIClient(
            timeout=30.0,
            max_retries=max(0, retry.max_attempts - 1),
            base_delay=retry.backoff_base_seconds,
            max_delay=retry.backoff_max_seconds,
            requests_per_second=1000.0,  # engine-level limiter governs globally
        )
        self._rate_limiter = rate_limiter
        self._local = threading.local()

    @property
    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._client_factory()
            self._local.client = client
        return client

    def fetch(self, igme_id: str, endpoint: str) -> FetchResult:
        self._rate_limiter.wait()
        data, r_hash = self._client.query_endpoint(igme_id, endpoint)
        attempts = 1
        error: str | None = None
        if data is None:
            error = "endpoint query failed after retries"
            attempts = max(1, self._client.max_retries + 1)
        else:
            metrics = self._client.metrics
            if metrics:
                attempts = max(1, metrics[-1].retry_count + 1)
        return FetchResult(data=data, response_hash=r_hash, http_attempts=attempts, error=error)

    def close(self) -> None:
        client = getattr(self._local, "client", None)
        if client is not None and client.session is not None:
            try:
                client.session.close()
            except Exception:
                pass


class SimulatedFetcher:
    """Deterministic offline API simulation for dry runs and tests ONLY.

    Responses are generated from a stable hash of (igme_id, endpoint,
    request version) so outputs are reproducible across processes.
    Failure injection (failure_rate / fail_exact) exercises retry paths.
    Synthetic data never validates real-world performance.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        latency_seconds: float = 0.0,
        failure_rate: float = 0.0,
        fail_exact: list[list[str]] | None = None,
        request_version: str = "v1",
    ) -> None:
        self._rate_limiter = rate_limiter
        self.latency = latency_seconds
        self.failure_rate = failure_rate
        # (igme_id, endpoint) pairs that always fail (deterministic injection).
        self.fail_exact = set(map(tuple, fail_exact or []))
        self.request_version = request_version
        self._rng = random.Random(20260826)

    def _digest(self, igme_id: str, endpoint: str) -> int:
        raw = f"{igme_id}|{endpoint}|{self.request_version}".encode()
        return int(hashlib.sha256(raw).hexdigest(), 16)

    def _should_fail(self, igme_id: str, endpoint: str) -> bool:
        if (igme_id, endpoint) in self.fail_exact:
            return True
        if self.failure_rate <= 0:
            return False
        self._rng.seed(self._digest(igme_id, endpoint))
        return self._rng.random() < self.failure_rate

    def _rows(self, igme_id: str, endpoint: str) -> list[dict[str, Any]]:
        h = self._digest(igme_id, endpoint)
        rng = random.Random(h)
        rows: list[dict[str, Any]] = []
        n = rng.randint(0, 3)
        for _ in range(n):
            row: dict[str, Any] = {"Id": igme_id}
            if endpoint == "piezometry":
                year = rng.randint(1980, 2020)
                row.update(
                    {
                        "Fecha": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                        "Nivel": round(rng.uniform(2.0, 120.0), 2),
                        "Valor": round(rng.uniform(0.0, 400.0), 2),
                        "Cota": round(rng.uniform(0.0, 600.0), 2),
                        "Surgencia": "No surgente",
                    }
                )
            elif endpoint == "lithology":
                top = 0.0
                for _ in range(rng.randint(1, 4)):
                    bottom = round(top + rng.uniform(5.0, 60.0), 2)
                    row.update(
                        {
                            "ProfTecho": round(top, 2),
                            "ProfMuro": bottom,
                            "Litologia": rng.choice(
                                ["Calizas", "Margas", "Areniscas", "Aluviales", "Conglomerados"]
                            ),
                            "Acuifero": bool(rng.random() < 0.5),
                        }
                    )
                    top = bottom
            elif endpoint == "pumping_tests":
                flow = round(rng.uniform(0.0, 80.0), 2)
                row.update(
                    {
                        "Fecha": f"{rng.randint(1980, 2020)}-{rng.randint(1, 12):02d}-01",
                        "Caudal": flow,
                        "DepresionM": round(rng.uniform(1.0, 80.0), 2),
                        "Horasdur": rng.randint(0, 6),
                        "Mindur": rng.randint(0, 59),
                    }
                )
            elif endpoint == "chemistry":
                row.update(
                    {
                        "Fecha": f"{rng.randint(1980, 2020)}-{rng.randint(1, 12):02d}-15",
                        "Conductividad": round(rng.uniform(200.0, 5000.0), 1),
                        "pH": round(rng.uniform(6.0, 8.5), 2),
                    }
                )
            elif endpoint == "hydrometry":
                row.update(
                    {
                        "Fecha": f"{rng.randint(1980, 2020)}-{rng.randint(1, 12):02d}-10",
                        "Valor": round(rng.uniform(0.0, 200.0), 2),
                    }
                )
            elif endpoint == "technical":
                row.update({"Prof": round(rng.uniform(10.0, 400.0), 2)})
            elif endpoint == "documents":
                row.update({"Nombre": f"doc_{h % 1000}.pdf"})
            rows.append(row)
        return rows

    def fetch(self, igme_id: str, endpoint: str) -> FetchResult:
        self._rate_limiter.wait()
        if self.latency > 0:
            time.sleep(self.latency)
        if self._should_fail(igme_id, endpoint):
            return FetchResult(
                data=None, response_hash="", http_attempts=1, error="simulated failure"
            )
        data = self._rows(igme_id, endpoint)
        from aquascan.enrichment.api import sha256_json

        return FetchResult(data=data, response_hash=sha256_json(data), http_attempts=1)


def make_fetcher(
    cfg,
) -> tuple[Any, RateLimiter]:
    """Build the fetcher + shared rate limiter from engine config.

    Returns (fetcher, rate_limiter). Synthetic mode is clearly labelled
    by cfg.api_mode == 'simulated'.
    """
    limiter = RateLimiter(cfg.rate_limit.requests_per_second)
    if cfg.api_mode == "simulated":
        return (
            SimulatedFetcher(
                rate_limiter=limiter,
                latency_seconds=cfg.simulated.latency_seconds,
                failure_rate=cfg.simulated.failure_rate,
                fail_exact=cfg.simulated.fail_exact,
                request_version=cfg.request_version,
            ),
            limiter,
        )
    return LiveFetcher(cfg.retry, limiter), limiter
