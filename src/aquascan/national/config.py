"""Engine configuration for the National Enrichment Engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aquascan.national import ENGINE_VERSION

DEFAULT_SOURCE = Path("data/processed/igme_candidates_v1.parquet")
STATE_DIR = Path("data/state")
STATE_DB = Path("data/state/national_enrichment_v1.sqlite")
INTERIM_DIR = Path("data/interim/national_v1")
FINAL_DATASET = Path("data/processed/aquascan_national_dataset_v1.parquet")
REPORTS_DIR = Path("reports")


@dataclass
class RateLimitConfig:
    """Global rate limiting (shared across worker threads)."""

    requests_per_second: float = 2.0  # conservative default; respectful of IGME infra


@dataclass
class RetryConfig:
    """Bounded retry with exponential backoff."""

    max_attempts: int = 4
    backoff_base_seconds: float = 5.0
    backoff_max_seconds: float = 300.0


@dataclass
class CircuitBreakerConfig:
    """Network health circuit breaker."""

    window_size: int = 50
    failure_ratio_threshold: float = 0.5
    cooldown_seconds: float = 60.0


@dataclass
class DiskConfig:
    """Disk space monitoring."""

    warning_free_gb: float = 5.0
    critical_free_gb: float = 1.0
    check_interval_seconds: float = 60.0


@dataclass
class CheckpointConfig:
    """Explicit checkpoint cadence (whichever occurs first)."""

    every_n_records: int = 100
    every_n_seconds: float = 300.0


@dataclass
class SimulatedConfig:
    """Deterministic offline API simulation (testing / dry runs only).

    Synthetic data is NEVER field validation — simulated runs must be
    labelled as such in every report they produce.
    """

    enabled: bool = False
    latency_seconds: float = 0.0
    failure_rate: float = 0.0
    # Optional fixed list of (igme_id, endpoint) pairs that always fail.
    fail_exact: list[list[str]] = field(default_factory=list)


@dataclass
class CanaryConfig:
    """4744-record regression canary (Gate 22)."""

    enabled: bool = True
    cohort_path: Path = Path("data/processed/ground_truth_cohort5000_v0_4.parquet")
    cache_path: Path = Path("data/discovery_cache/gate5_cohort5000/enrichment_cache.json")
    # Stop the national run if any semantic mismatch is detected.
    fail_on_mismatch: bool = True


@dataclass
class NationalConfig:
    """Full engine configuration. Serialized to reports/national_enrichment_config_v1.json."""

    engine_version: str = ENGINE_VERSION
    schema_version: str = "1"
    name: str = "national_v1"

    source: Path = DEFAULT_SOURCE
    state_db: Path = STATE_DB
    state_dir: Path = Path("data/state/national_v1")
    interim_dir: Path = INTERIM_DIR
    final_dataset: Path = FINAL_DATASET
    reports_dir: Path = REPORTS_DIR

    request_version: str = "v1"  # cache key component: record + endpoint + version
    endpoint_version: str = "v1"  # IGME bdaguas API contract version
    api_mode: str = "live"  # "live" | "simulated"

    batch_size: int = 500
    concurrency: int = 4  # bounded concurrent network jobs (4-8 recommended)
    stale_timeout_seconds: float = 1800.0
    resume_retries_failed: bool = False

    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    disk: DiskConfig = field(default_factory=DiskConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    simulated: SimulatedConfig = field(default_factory=SimulatedConfig)
    canary: CanaryConfig = field(default_factory=CanaryConfig)

    # Immutable input snapshot metadata (filled at init)
    source_hash: str = ""
    source_record_count: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict (paths as strings)."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if isinstance(obj, tuple):
                return [_convert(v) for v in obj]
            return obj

        return _convert(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NationalConfig:
        """Rebuild a config from a serialized dict (paths restored)."""
        raw = dict(data)
        for key in (
            "source",
            "state_db",
            "state_dir",
            "interim_dir",
            "final_dataset",
            "reports_dir",
        ):
            if key in raw and isinstance(raw[key], str):
                raw[key] = Path(raw[key])
        raw["canary"]["cohort_path"] = Path(raw["canary"]["cohort_path"])
        raw["canary"]["cache_path"] = Path(raw["canary"]["cache_path"])
        raw["rate_limit"] = RateLimitConfig(**raw["rate_limit"])
        raw["retry"] = RetryConfig(**raw["retry"])
        raw["circuit_breaker"] = CircuitBreakerConfig(**raw["circuit_breaker"])
        raw["disk"] = DiskConfig(**raw["disk"])
        raw["checkpoint"] = CheckpointConfig(**raw["checkpoint"])
        raw["simulated"] = SimulatedConfig(**raw["simulated"])
        raw["canary"] = CanaryConfig(**raw["canary"])
        return cls(**raw)


def save_config(cfg: NationalConfig) -> Path:
    """Persist the configuration to reports/national_enrichment_config_v1.json (atomic)."""
    from aquascan.national.io_utils import atomic_write_json

    path = cfg.reports_dir / "national_enrichment_config_v1.json"
    atomic_write_json(path, cfg.to_dict())
    return path


def load_config(path: Path) -> NationalConfig:
    """Load a persisted configuration."""
    with open(path) as f:
        data = json.load(f)
    cfg = NationalConfig.from_dict(data)
    cfg.created_at = data.get("created_at", "")
    return cfg


def stamp_config(cfg: NationalConfig, source_hash: str, record_count: int) -> NationalConfig:
    """Attach immutable source snapshot metadata (filled once at init)."""
    cfg.source_hash = source_hash
    cfg.source_record_count = record_count
    cfg.created_at = datetime.now(UTC).isoformat()
    return cfg
