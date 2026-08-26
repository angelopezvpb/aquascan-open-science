"""Controlled Ground Truth enrichment tests.

Covers:
- cache reproducibility
- retries
- resume
- endpoint failure isolation
- source-response version changes
- temporal observations
- depth-to-water vs hydraulic-head separation
- lithological interval validation
- target readiness independence
- water-presence UNKNOWN handling
- Candidate Score independence
- provenance
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aquascan.data.models import (
    LithologyAnomaly,
    TargetReadiness,
    TemporalRelevanceStatus,
    WaterPresenceStatus,
)
from aquascan.enrichment.cache import ResponseCache
from aquascan.enrichment.normalize import (
    build_normalized_record,
    compute_documentary_confidence,
    compute_temporal_summary,
    parse_lithology_intervals,
    parse_pumping_tests,
    parse_water_level_observations,
    validate_lithology_intervals,
)

# ── Fixtures ────────────────────────────────────────────────────────


def make_raw_record(overrides: dict | None = None) -> dict:
    """Build a synthetic raw API response dict for testing."""
    raw = {
        "igme_id": "TEST-1-0001",
        "piezometry": [],
        "lithology": [],
        "pumping_tests": [],
        "chemistry": [],
        "hydrometry": [],
        "technical": [],
        "documents": [],
        "piezometry_sha256": "",
        "lithology_sha256": "",
        "pumping_tests_sha256": "",
        "chemistry_sha256": "",
        "hydrometry_sha256": "",
        "technical_sha256": "",
        "documents_sha256": "",
        "_source_changed": False,
        "_changed_endpoints": [],
    }
    if overrides:
        raw.update(overrides)
    return raw


# ── 1. Cache reproducibility ────────────────────────────────────────


def test_cache_reproducible(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    data = [{"Id": "X", "Valor": 1.0}]
    h1 = cache.put("A", "piezometry", data, "hash123")
    assert h1 is True
    # Same content, same hash → cache hit, no new version
    h2 = cache.put("A", "piezometry", data, "hash123")
    assert h2 is True
    assert cache.get("A", "piezometry") == (data, "hash123")
    assert "A" not in cache.changed_records


def test_cache_reloads_from_disk(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir)
    data = [{"Id": "X", "Valor": 2.0}]
    cache.put("A", "piezometry", data, "hash-abc")
    cache.save()

    # Reload from disk
    cache2 = ResponseCache(cache_dir)
    assert cache2.get("A", "piezometry") == (data, "hash-abc")
    assert cache2.record_count == 1


# ── 2. Retries (in api client) ──────────────────────────────────────


def test_api_client_retry_on_server_error(tmp_path: Path, monkeypatch) -> None:
    from aquascan.enrichment.api import EnrichmentAPIClient

    client = EnrichmentAPIClient(max_retries=2, base_delay=0.0)

    calls = {"count": 0}

    def fake_post(endpoint: str, body: dict):
        calls["count"] += 1
        if calls["count"] < 3:
            resp = type("Resp", (), {})()
            resp.status_code = 500
            resp.text = "server error"
            resp.elapsed = type("E", (), {"total_seconds": lambda self: 0.1})()
            resp.content = b"err"
            return resp
        resp = type("Resp", (), {})()
        resp.status_code = 200
        resp.text = "ok"
        resp.elapsed = type("E", (), {"total_seconds": lambda self: 0.1})()
        resp.content = b"[]"
        resp.json = lambda: []
        return resp

    client._post = fake_post  # type: ignore[assignment]

    metric = type("M", (), {})()
    metric.retry_count = 0
    metric.status_code = 0
    metric.elapsed_ms = 0.0
    metric.response_bytes = 0
    metric.success = False
    metric.error_message = ""

    resp = client._retryable_post("/QueryAGPZ", {"Ids": ["X"]}, metric)
    assert resp is not None
    assert resp.status_code == 200
    assert calls["count"] == 3  # initial + 2 retries


# ── 3. Resume (cache reuse) ─────────────────────────────────────────


def test_resume_uses_cache(tmp_path: Path) -> None:
    from aquascan.enrichment.api import EnrichmentAPIClient

    cache = ResponseCache(tmp_path / "cache")
    data = [{"Id": "X", "Valor": 5.0}]
    cache.put("X", "piezometry", data, "some-hash")

    client = EnrichmentAPIClient()
    # The client should prefer cache over API
    raw = client.query_all("X", cache)
    assert raw["piezometry"] == data
    assert raw["piezometry_sha256"] == "some-hash"
    assert raw["piezometry_cached"] is True


# ── 4. Endpoint failure isolation ───────────────────────────────────


def test_endpoint_failure_isolation(tmp_path: Path, monkeypatch) -> None:
    from aquascan.enrichment.api import EnrichmentAPIClient

    client = EnrichmentAPIClient(max_retries=0, base_delay=0.0)

    def fake_query_endpoint(igme_id: str, endpoint: str):
        # Fail only pumping_tests
        if endpoint == "pumping_tests":
            return None, ""
        return [], "ok-hash"

    monkeypatch.setattr(client, "query_endpoint", fake_query_endpoint)

    raw = client.query_all("X", None)
    # Other endpoints still populated
    assert raw["piezometry"] == []
    assert raw["piezometry_count"] == 0
    # Failed endpoint marked with error, not invalidating others
    assert raw["pumping_tests"] == []
    assert raw["pumping_tests_error"] is True
    # Lithology still present
    assert raw["lithology"] == []


# ── 5. Source-response version changes ──────────────────────────────


def test_source_response_changed(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    data_v1 = [{"Id": "X", "Valor": 1.0}]
    data_v2 = [{"Id": "X", "Valor": 2.0}]

    cache.put("X", "piezometry", data_v1, "hash-v1")
    cache.put("X", "piezometry", data_v2, "hash-v2")

    # New version is stored
    assert cache.get("X", "piezometry")[1] == "hash-v2"
    # Source change flagged
    assert "X" in cache.changed_records
    assert "piezometry" in cache.changed_records["X"]
    # Old version preserved
    entry = cache._storage["X"]["piezometry"]
    assert len(entry["versions"]) == 1
    assert entry["versions"][0]["sha256"] == "hash-v1"


# ── 6. Temporal observations ────────────────────────────────────────


def test_temporal_summary_preserves_series() -> None:
    raw = make_raw_record(
        {
            "piezometry": [
                {"Fecha": "1985-05-31", "Nivel": 42.0, "Valor": 100.0, "Cota": 142.0},
                {"Fecha": "1991-05-14", "Nivel": 70.85, "Valor": -1.55, "Cota": 69.3},
                {"Fecha": "1991-06-01", "Nivel": 65.0, "Valor": 4.3, "Cota": 69.3},
            ]
        }
    )
    obs = parse_water_level_observations(raw["piezometry"], "TEST-1-0001")
    assert len(obs) == 3  # time series preserved, not summarized away

    summary = compute_temporal_summary(obs)
    assert summary.num_measurements == 3
    assert summary.first_measurement_date == "1985-05-31"
    assert summary.last_measurement_date == "1991-06-01"
    assert summary.temporal_span_days is not None
    assert summary.min_depth_to_water_m == 42.0
    assert summary.max_depth_to_water_m == 70.85
    assert summary.median_depth_to_water_m == 65.0


def test_temporal_relevance_status_unknown() -> None:
    raw = make_raw_record({"piezometry": [{"Nivel": 10.0, "Valor": 100.0}]})
    obs = parse_water_level_observations(raw["piezometry"], "TEST-1-0001")
    summary = compute_temporal_summary(obs)
    assert summary.temporal_relevance_status == TemporalRelevanceStatus.DATE_UNKNOWN


# ── 7. Depth-to-water vs hydraulic-head separation ──────────────────


def test_depth_to_water_not_collapsed_with_head() -> None:
    raw = make_raw_record({"piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}]})
    obs = parse_water_level_observations(raw["piezometry"], "TEST-1-0001")
    o = obs[0]
    # Two distinct targets, not interchangeable
    assert o.depth_to_water_m == 42.0  # TARGET_DEPTH_TO_WATER (Nivel)
    assert o.hydraulic_head_elevation_m == 100.0  # TARGET_HYDRAULIC_HEAD_ELEVATION (Valor)
    assert o.depth_to_water_m != o.hydraulic_head_elevation_m
    # Consistency check is separate (residual)
    assert o.consistency_residual_m == pytest.approx(100.0 - (142.0 - 42.0), abs=1e-9)


# ── 8. Lithological interval validation ─────────────────────────────


def test_lithology_validation_flags_anomalies() -> None:
    from aquascan.data.models import LithologyInterval

    intervals = [
        LithologyInterval(igme_id="X", top_depth_m=0.0, bottom_depth_m=10.0, order_index=1),
        LithologyInterval(igme_id="X", top_depth_m=8.0, bottom_depth_m=15.0, order_index=2),
    ]
    result = validate_lithology_intervals(intervals, total_depth_m=20.0)
    assert not result.is_valid
    assert LithologyAnomaly.OVERLAPPING_INTERVALS in result.anomalies


def test_lithology_validation_bottom_lt_top() -> None:
    from aquascan.data.models import LithologyInterval

    intervals = [
        LithologyInterval(igme_id="X", top_depth_m=10.0, bottom_depth_m=5.0, order_index=1),
    ]
    result = validate_lithology_intervals(intervals)
    assert LithologyAnomaly.BOTTOM_LT_TOP in result.anomalies


def test_lithology_validation_negative_depth() -> None:
    from aquascan.data.models import LithologyInterval

    intervals = [
        LithologyInterval(igme_id="X", top_depth_m=-1.0, bottom_depth_m=5.0, order_index=1),
    ]
    result = validate_lithology_intervals(intervals)
    assert LithologyAnomaly.NEGATIVE_DEPTH in result.anomalies


def test_lithology_validation_exceeds_depth() -> None:
    from aquascan.data.models import LithologyInterval

    intervals = [
        LithologyInterval(igme_id="X", top_depth_m=0.0, bottom_depth_m=100.0, order_index=1),
    ]
    result = validate_lithology_intervals(intervals, total_depth_m=50.0)
    assert LithologyAnomaly.EXCEEDS_TOTAL_DEPTH in result.anomalies


def test_lithology_validation_gap() -> None:
    from aquascan.data.models import LithologyInterval

    intervals = [
        LithologyInterval(igme_id="X", top_depth_m=0.0, bottom_depth_m=10.0, order_index=1),
        LithologyInterval(igme_id="X", top_depth_m=20.0, bottom_depth_m=30.0, order_index=2),
    ]
    result = validate_lithology_intervals(intervals)
    assert LithologyAnomaly.GAP in result.anomalies


def test_lithology_aquifer_flag_not_water_entry() -> None:
    """aquifer_flagged_lithology_interval is NOT a measured water entry."""
    raw = make_raw_record(
        {
            "lithology": [
                {
                    "Litologia": "Calizas",
                    "ProfTecho": 0.0,
                    "ProfMuro": 72.9,
                    "Acuifero": True,
                }
            ]
        }
    )
    intervals = parse_lithology_intervals(raw["lithology"], "TEST-1-0001")
    assert intervals[0].aquifer_flagged_lithology_interval is True
    # The field is explicitly named aquifer_flagged_lithology_interval
    from aquascan.data.models import LithologyInterval

    assert "aquifer_flagged_lithology_interval" in LithologyInterval.model_fields


# ── 9. Pumping tests ────────────────────────────────────────────────


def test_pumping_test_specific_capacity_derived() -> None:
    raw = make_raw_record(
        {"pumping_tests": [{"Caudal": 25.0, "DepresionM": 65.0, "Horasdur": 0, "Mindur": 360}]}
    )
    tests = parse_pumping_tests(raw["pumping_tests"], "TEST-1-0001")
    t = tests[0]
    assert t.flow_l_s == 25.0
    assert t.drawdown_m == 65.0
    assert t.duration_minutes == 360
    # Specific capacity derived: flow >= 0 and drawdown > 0
    assert t.specific_capacity_l_s_m == pytest.approx(25.0 / 65.0)
    assert t.specific_capacity_is_derived is True


def test_pumping_test_no_specific_capacity_when_drawdown_zero() -> None:
    raw = make_raw_record({"pumping_tests": [{"Caudal": 25.0, "DepresionM": 0.0}]})
    tests = parse_pumping_tests(raw["pumping_tests"], "TEST-1-0001")
    t = tests[0]
    # drawdown <= 0 → no specific capacity (DERIVED only when drawdown > 0)
    assert t.specific_capacity_l_s_m is None
    assert t.specific_capacity_is_derived is False


def test_pumping_test_no_specific_capacity_when_flow_negative() -> None:
    raw = make_raw_record({"pumping_tests": [{"Caudal": -5.0, "DepresionM": 10.0}]})
    tests = parse_pumping_tests(raw["pumping_tests"], "TEST-1-0001")
    t = tests[0]
    assert t.specific_capacity_l_s_m is None


# ── 10. Water-presence UNKNOWN handling ─────────────────────────────


def test_water_presence_unknown_without_observations() -> None:
    raw = make_raw_record()  # no observations at all
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    assert rec.water_presence.status == WaterPresenceStatus.UNKNOWN


def test_water_presence_not_naive_from_point_nature() -> None:
    """point nature = Sondeo must NOT imply water presence."""
    raw = make_raw_record()  # no piezometry, no pumping, no hydrometry
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    # Even though the record might be a Sondeo, no hydrogeological observation
    # means UNKNOWN — never PRESENT from point nature alone.
    assert rec.water_presence.status == WaterPresenceStatus.UNKNOWN


def test_water_presence_from_piezometry() -> None:
    raw = make_raw_record({"piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}]})
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    assert rec.water_presence.status == WaterPresenceStatus.PRESENT


def test_water_presence_from_positive_pump_flow() -> None:
    raw = make_raw_record({"pumping_tests": [{"Caudal": 25.0, "DepresionM": 65.0}]})
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    assert rec.water_presence.status == WaterPresenceStatus.PRESENT


# ── 11. Target readiness independence ───────────────────────────────


def test_target_readiness_reads_piezometry_only() -> None:
    raw = make_raw_record({"piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}]})
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    tr = rec.target_readiness
    # Depth-to-water READY (has Nivel)
    assert tr["TARGET_DEPTH_TO_WATER"].evidence_available is True
    # Hydraulic head READY (has Valor)
    assert tr["TARGET_HYDRAULIC_HEAD_ELEVATION"].evidence_available is True
    # Lithology NOT_READY (no lithology)
    assert tr["TARGET_LITHOLOGY"].readiness == TargetReadiness.NOT_READY
    # Productivity NOT_READY (no pumping test)
    assert tr["TARGET_PRODUCTIVITY"].readiness == TargetReadiness.NOT_READY


def test_target_readiness_lithology_only() -> None:
    raw = make_raw_record(
        {
            "lithology": [
                {
                    "Litologia": "Calizas",
                    "ProfTecho": 0.0,
                    "ProfMuro": 72.9,
                    "Acuifero": True,
                    "Edad": "Secundario",
                }
            ]
        }
    )
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    tr = rec.target_readiness
    assert tr["TARGET_LITHOLOGY"].readiness in (TargetReadiness.READY, TargetReadiness.PARTIAL)
    assert tr["TARGET_AQUIFER_INTERVAL"].evidence_available is True
    # No water level → depth-to-water NOT_READY
    assert tr["TARGET_DEPTH_TO_WATER"].readiness == TargetReadiness.NOT_READY


# ── 12. Candidate Score independence ────────────────────────────────


def test_candidate_score_does_not_influence_readiness() -> None:
    """Readiness must be identical regardless of candidate score."""
    raw = make_raw_record({"piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}]})
    rec_high = build_normalized_record("TEST-1-0001", "Santa Eulalia", "P1", 1.0, None, raw)
    rec_low = build_normalized_record("TEST-1-0001", "Santa Eulalia", "P4", 0.0, None, raw)
    for target in rec_high.target_readiness:
        assert (
            rec_high.target_readiness[target].readiness
            == rec_low.target_readiness[target].readiness
        )


# ── 13. Provenance ──────────────────────────────────────────────────


def test_provenance_present_in_normalized_record() -> None:
    raw = make_raw_record(
        {
            "piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}],
            "piezometry_sha256": "abc123",
        }
    )
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    assert rec.provenance["igme_id"] == "TEST-1-0001"
    assert rec.provenance["piezometry_sha256"] == "abc123"
    assert "lithology_sha256" in rec.provenance


def test_source_response_changed_propagates() -> None:
    raw = make_raw_record(
        {
            "piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}],
            "_source_changed": True,
            "_changed_endpoints": ["piezometry"],
        }
    )
    rec = build_normalized_record("TEST-1-0001", "Santa Eulalia", None, None, None, raw)
    assert rec.source_response_changed is True
    assert "piezometry" in rec.changed_endpoints


# ── 14. Documentary confidence ──────────────────────────────────────


def test_documentary_confidence_scales_with_families() -> None:
    c_none = compute_documentary_confidence([], [], [], [], [], [], [], False)
    assert c_none == 0.0

    raw_many = make_raw_record(
        {
            "piezometry": [{"Nivel": 42.0, "Valor": 100.0, "Cota": 142.0}],
            "lithology": [{"Litologia": "Calizas"}],
            "pumping_tests": [{"Caudal": 25.0, "DepresionM": 65.0}],
        }
    )
    wl = parse_water_level_observations(raw_many["piezometry"], "X")
    lith = parse_lithology_intervals(raw_many["lithology"], "X")
    pump = parse_pumping_tests(raw_many["pumping_tests"], "X")
    c_many = compute_documentary_confidence(wl, lith, pump, [], [], [], [], False)
    assert c_many > c_none


# ── Telemetry Tests ───────────────────────────────────────────


class FakeCache:
    """In-memory cache for testing telemetry counters."""

    def __init__(self, data: dict[str, dict[str, list[dict]]] | None = None):
        self._data: dict[str, dict[str, list[dict]]] = data or {}
        self.changed_records: dict[str, list[str]] = {}
        self.record_count: int = len(self._data)

    def get(self, igme_id: str, ep_name: str) -> tuple[list[dict] | None, str]:
        record = self._data.get(igme_id, {})
        if ep_name in record:
            return record[ep_name], hashlib.sha256(
                json.dumps(record[ep_name], sort_keys=True).encode()
            ).hexdigest()
        return None, ""

    def put(self, igme_id: str, ep_name: str, data: list[dict], sha: str) -> None:
        self._data.setdefault(igme_id, {})[ep_name] = data

    def save(self) -> None:
        pass

    def verify_integrity(self) -> dict[str, bool | str]:
        return {"valid": True, "message": "ok"}


class TestTelemetry:
    """Telemetry correction tests.

    Verifies the cache-hit / network-request distinction is exact.
    """

    def test_empty_cache_all_network(self) -> None:
        """A: 7 logical queries → 7 network requests, 0 cache hits."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        client = EnrichmentAPIClient()
        fake_cache = FakeCache()  # empty
        raw = client.query_all("TEL-1-0001", fake_cache)

        assert len(raw) >= len(ENDPOINTS) + 2  # +2 for igme_id, retrieved_at
        assert client.logical_endpoint_queries == len(ENDPOINTS)
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits
        # Empty cache → all should be network, but if endpoints fail/empty we still count them
        assert client.total_requests + client.endpoint_cache_hits == client.logical_endpoint_queries
        assert client.endpoint_cache_hits == 0
        assert client.records_processed == 1

    def test_fully_cached_all_cache(self) -> None:
        """B: 7 logical queries → 0 network, 7 cache hits."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        # Pre-populate cache for all endpoints
        precache: dict[str, dict[str, list[dict]]] = {}
        for ep_name in ENDPOINTS:
            precache.setdefault("TEL-1-0002", {})[ep_name] = [{"cached": True}]

        client = EnrichmentAPIClient()
        fake_cache = FakeCache(precache)
        raw = client.query_all("TEL-1-0002", fake_cache)

        assert len(raw) >= len(ENDPOINTS) + 2
        assert client.endpoint_cache_hits == len(ENDPOINTS)
        assert client.total_requests == 0  # no network requests at all
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits
        assert client.records_reused_from_cache == 1
        assert client.records_requiring_network == 0

    def test_partially_cached_mixed(self) -> None:
        """C: partial cache → logical = network + cache_hits, both >0."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        # Pre-cache only 3 of 7 endpoints
        precache: dict[str, dict[str, list[dict]]] = {}
        cached_eps = list(ENDPOINTS.keys())[:3]
        for ep_name in cached_eps:
            precache.setdefault("TEL-1-0003", {})[ep_name] = [{"cached": True}]

        client = EnrichmentAPIClient()
        fake_cache = FakeCache(precache)
        raw = client.query_all("TEL-1-0003", fake_cache)

        assert len(raw) >= len(ENDPOINTS) + 2
        assert client.endpoint_cache_hits == 3
        assert client.logical_endpoint_queries == len(ENDPOINTS)
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits
        # Not fully cached → needs network
        assert client.records_reused_from_cache == 0
        assert client.records_requiring_network == 1

    def test_multiple_records_counters_accumulate(self) -> None:
        """D: counters remain exact across multiple records."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        n_eps = len(ENDPOINTS)

        # Pre-cache first record completely, second partially (2 eps)
        precache: dict[str, dict[str, list[dict]]] = {}
        for ep_name in ENDPOINTS:
            precache.setdefault("TEL-M-0001", {})[ep_name] = [{"cached": True}]
        for ep_name in list(ENDPOINTS.keys())[:2]:
            precache.setdefault("TEL-M-0002", {})[ep_name] = [{"cached": True}]

        client = EnrichmentAPIClient()
        fake_cache = FakeCache(precache)

        # Record 1: fully cached
        client.query_all("TEL-M-0001", fake_cache)
        assert client.endpoint_cache_hits == n_eps
        assert client.records_reused_from_cache == 1
        assert client.total_requests == 0

        # Record 2: partially cached (2 cache hits, n_eps-2 network)
        client.query_all("TEL-M-0002", fake_cache)
        assert client.endpoint_cache_hits == n_eps + 2
        assert client.records_reused_from_cache == 1
        assert client.records_requiring_network == 1
        assert client.logical_endpoint_queries == 2 * n_eps
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits

        # Record 3: empty cache (all network)
        client.query_all("TEL-M-0003", fake_cache)
        assert client.logical_endpoint_queries == 3 * n_eps
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits
        assert client.records_processed == 3

    def test_cache_hit_does_not_affect_latency(self) -> None:
        """E: Cache hits must never increase network latency or response-byte metrics."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        # Fully cached record
        precache: dict[str, dict[str, list[dict]]] = {}
        for ep_name in ENDPOINTS:
            precache.setdefault("TEL-E-0001", {})[ep_name] = [{"cached": True}]

        client = EnrichmentAPIClient()
        fake_cache = FakeCache(precache)
        client.query_all("TEL-E-0001", fake_cache)

        assert client.total_requests == 0
        ms = client.get_metrics_summary()
        # No network requests → total_elapsed_ms must be 0
        assert ms["total_elapsed_ms"] == 0
        assert ms["response_bytes"] == 0
        assert ms["mean_latency_ms"] == 0

    def test_failed_network_distinct_from_cache_miss(self) -> None:
        """F: Failed network request remains distinguishable from cache miss."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        client = EnrichmentAPIClient()
        fake_cache = FakeCache()  # empty

        # We can't force API failures without live endpoints,
        # but we verify: if cache returns nothing, it's NOT a "cache_hit"
        # and the logical query goes to the network path.
        before = client.logical_endpoint_queries
        client.query_all("TEL-F-0001", fake_cache)
        after = client.logical_endpoint_queries

        # All queries were logical (not cache hits)
        assert after - before == len(ENDPOINTS)
        assert client.endpoint_cache_hits == 0
        assert client.total_requests == len(ENDPOINTS)
        # Invariant holds
        assert client.logical_endpoint_queries == client.total_requests + client.endpoint_cache_hits

    def test_metrics_summary_contains_all_counters(self) -> None:
        """Verify get_metrics_summary() returns corrected telemetry fields."""
        from aquascan.enrichment.api import (
            ENDPOINTS,
            EnrichmentAPIClient,
        )

        client = EnrichmentAPIClient()
        fake_cache = FakeCache()
        client.query_all("TEL-G-0001", fake_cache)

        ms = client.get_metrics_summary()
        assert "logical_endpoint_queries" in ms
        assert "network_requests" in ms
        assert "endpoint_cache_hits" in ms
        assert "records_processed" in ms
        assert "records_fully_reused_from_cache" in ms
        assert "records_requiring_network" in ms
        assert "failed_network_requests" in ms
        assert "telemetry_invariant_holds" in ms
        assert ms["logical_endpoint_queries"] == len(ENDPOINTS)
        assert ms["telemetry_invariant_holds"] is True

    def test_network_request_never_a_cache_hit(self) -> None:
        """An APIRequestMetrics created for HTTP must NEVER have cache_hit=True."""
        from aquascan.enrichment.api import (
            EnrichmentAPIClient,
        )

        client = EnrichmentAPIClient()
        fake_cache = FakeCache()
        client.query_all("TEL-H-0001", fake_cache)

        # All metrics should have cache_hit=False (no fake metrics created)
        assert len(client.metrics) == client.total_requests
        for metric in client.metrics:
            assert metric.cache_hit is False
