"""National Enrichment Engine tests (Gate 38).

Covers pause/resume, SIGINT graceful shutdown, stale RUNNING recovery,
endpoint-level resume, cache reuse, atomic writes, checkpoint integrity,
bounded retries, circuit breaker, rate limiting, telemetry invariant,
batch/partition integrity, disk-critical pause, crash recovery (SIGKILL),
final count, no Phase 0 exposure, and verify.

All network traffic is SIMULATED (deterministic offline fetcher).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from aquascan.national.api import CircuitBreaker, RateLimiter
from aquascan.national.cache import DurableCache
from aquascan.national.config import CircuitBreakerConfig, NationalConfig, stamp_config
from aquascan.national.constants import (
    COMPLETE,
    ENDPOINT_NAMES,
    ENGINE_COMPLETE,
    ENGINE_DISK_CRITICAL,
    ENGINE_FAILED,
    ENGINE_PAUSED,
    FAILED,
    PARTIAL,
    PENDING,
    QUARANTINED,
)
from aquascan.national.engine import NationalEnrichmentEngine
from aquascan.national.io_utils import sha256_file
from aquascan.national.state import StateDB, now_iso

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# ── fixtures / helpers ───────────────────────────────────────────────


def make_source(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "igme_id": f"TEST-{i:04d}",
                "latitude": 39.0 + i / 10000.0,
                "longitude": -2.0 + i / 10000.0,
                "hydrogeological_unit": "San Antonio" if i % 2 else "Región de los arenales",
                "candidate_tier": "P1",
                "candidate_priority_score": 0.5,
                "detail_url": f"https://info.igme.es/detalle/{i}",
                "total_depth_m": 100.0,
                "reference_flow_l_s": None,
                "construction_date": "1990-05-01",
                "qa_flag_count": 0,
                "province": "Test",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def make_config(
    tmp_path: Path,
    n: int = 30,
    batch_size: int = 10,
    **overrides,
) -> NationalConfig:
    source = tmp_path / "source.parquet"
    make_source(source, n)
    cfg = NationalConfig(
        source=source,
        state_db=tmp_path / "state" / "national.sqlite",
        state_dir=tmp_path / "state" / "engine",
        interim_dir=tmp_path / "interim",
        final_dataset=tmp_path / "processed" / "aquascan_national_dataset_v1.parquet",
        reports_dir=tmp_path / "reports",
    )
    cfg.api_mode = "simulated"
    cfg.canary.enabled = False
    cfg.batch_size = batch_size
    cfg.checkpoint.every_n_records = 5
    # Tests exercise engine mechanics, not production politeness: disable
    # the shared rate limiter (it is covered by a dedicated unit test).
    cfg.rate_limit.requests_per_second = 0.0
    cfg = stamp_config(cfg, sha256_file(source), n)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def setup_state(cfg: NationalConfig) -> StateDB:
    db = StateDB(cfg.state_db, create=True)
    db.set_meta("source_hash", cfg.source_hash)
    db.set_meta("engine_version", "test")
    ids = [f"TEST-{i:04d}" for i in range(1, cfg.source_record_count + 1)]
    batch_ids: dict[str, str] = {}
    for i in range(0, len(ids), cfg.batch_size):
        bid = f"B{i // cfg.batch_size + 1:05d}"
        for g in ids[i : i + cfg.batch_size]:
            batch_ids[g] = bid
    db.seed_records(ids, batch_ids, cfg.source_hash)
    db.create_batches(cfg.batch_size)
    return db


def run_engine(
    cfg: NationalConfig, limit: int | None = None
) -> tuple[NationalEnrichmentEngine, str]:
    engine = NationalEnrichmentEngine(cfg)
    final = engine.run(limit=limit, quiet=True)
    return engine, final


def open_db(cfg: NationalConfig) -> StateDB:
    return StateDB(cfg.state_db, create=False)


def counter_delta(db: StateDB, key: str, before: int) -> int:
    return db.get_counter(key) - before


# ── 1. init / status ─────────────────────────────────────────────────


def test_init_seeds_records(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=12, batch_size=5)
    db = setup_state(cfg)
    try:
        assert db.record_count() == 12
        counts = db.record_status_counts()
        assert counts[PENDING] == 12
        assert db.get_meta("source_hash") == cfg.source_hash
        batches = db.incomplete_batches()
        assert len(batches) == 3  # 12 records / batch of 5
    finally:
        db.close()


# ── 2. full run + telemetry invariant ────────────────────────────────


def test_full_run_completes_with_invariant(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=25, batch_size=10)
    db = setup_state(cfg)
    db.close()

    engine, final = run_engine(cfg)
    assert final == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        counts = db.record_status_counts()
        assert counts.get(COMPLETE) == 25
        assert db.telemetry_invariant_holds()
        t = db.telemetry()
        # 25 records × 7 endpoints, all from network on first run
        assert t["logical_endpoint_queries"] == 25 * 7
        assert t["network_requests"] == 25 * 7
        assert t["endpoint_cache_hits"] == 0
        assert t["records_requiring_network"] == 25
        assert t["records_reused_from_cache"] == 0
        assert t["failed_network_requests"] == 0
        # Batches complete with partitions + sidecars
        assert len(db.incomplete_batches()) == 0
        parts = sorted(cfg.interim_dir.glob("part-*.parquet"))
        assert len(parts) == 3
        for part in parts:
            sidecar = part.with_suffix(".parquet.hash.json")
            assert sidecar.exists()
            meta = json.loads(sidecar.read_text())
            assert meta["partition_sha256"] == sha256_file(part)
        # Cache entries == 25 × 7
        assert engine.cache.entry_count() == 25 * 7
        # Checkpoint written and hashable/verifiable
        assert db.checkpoint_count() >= 1
        assert db.verify_checkpoint(db.latest_checkpoint()["checkpoint_id"])
    finally:
        db.close()


def test_verify_passes_after_complete_run(tmp_path: Path) -> None:
    from aquascan.national.verify import run_verify

    cfg = make_config(tmp_path, n=10)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)
    report = run_verify(cfg, check_canary=False)
    assert report["valid"], report["errors"]
    assert report["checks"]["telemetry_invariant"]["valid"]


# ── 3. exact resume / cache reuse / determinism ──────────────────────


def test_eligible_records_includes_record_stuck_after_last_job(tmp_path: Path) -> None:
    """A crash after the last endpoint job completes must not strand the record.

    Regression test: the record stays PENDING (or RUNNING) while every endpoint
    job is COMPLETE. The engine must still pick it up and finalize it, instead
    of declaring COMPLETE with stuck records.
    """
    cfg = make_config(tmp_path, n=1, batch_size=1)
    db = setup_state(cfg)
    try:
        db.ensure_endpoint_jobs("TEST-0001", list(ENDPOINT_NAMES))
        for ep in ENDPOINT_NAMES:
            db.mark_job_complete("TEST-0001", ep, "hash", cache_hit=False)
        assert db.get_record("TEST-0001")["status"] == PENDING
        assert "TEST-0001" in db.eligible_records(now_iso())
    finally:
        db.close()


def test_resume_skips_completed_work_and_is_deterministic(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=20, batch_size=20)
    db = setup_state(cfg)
    db.close()

    engine1, final1 = run_engine(cfg, limit=12)
    assert final1 == ENGINE_PAUSED  # session-limited

    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 12
        net_after_first = db.get_counter("network_requests")
        first_hashes = {
            r["igme_id"]: r["result_hash"]
            for r in db._query("SELECT igme_id, result_hash FROM records WHERE status = 'COMPLETE'")
        }
    finally:
        db.close()

    engine2, final2 = run_engine(cfg, limit=20)
    assert final2 == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        # Second session: only the remaining 8 records required network.
        delta = counter_delta(db, "network_requests", net_after_first)
        assert delta == 8 * 7
        assert db.record_status_counts().get(COMPLETE) == 20
        # Determinism: the 12 records completed in the first session keep
        # byte-identical outputs after the second session.
        for igme_id, expected in first_hashes.items():
            row = db._query_one("SELECT result_hash FROM records WHERE igme_id = ?", (igme_id,))
            assert row["result_hash"] == expected
    finally:
        db.close()


def test_cache_reuse_reprocess_without_network(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=15, batch_size=15)
    db = setup_state(cfg)
    db.close()

    run_engine(cfg)

    db = open_db(cfg)
    try:
        first_hashes = {
            r["igme_id"]: r["result_hash"]
            for r in db._query("SELECT igme_id, result_hash FROM records")
        }
        db.reset_for_reprocess(list(first_hashes))
        net_before = db.get_counter("network_requests")
        cache_before = db.get_counter("endpoint_cache_hits")
        reused_before = db.get_counter("records_reused_from_cache")
    finally:
        db.close()

    engine2, final2 = run_engine(cfg)
    assert final2 == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        # Everything was served from the durable cache: zero network.
        assert counter_delta(db, "network_requests", net_before) == 0
        assert counter_delta(db, "endpoint_cache_hits", cache_before) == 15 * 7
        assert counter_delta(db, "records_reused_from_cache", reused_before) == 15
        # Deterministic outputs: identical hashes to the first run.
        for r in db._query("SELECT igme_id, result_hash FROM records"):
            assert r["result_hash"] == first_hashes[r["igme_id"]]
    finally:
        db.close()


# ── 4. endpoint-level resume / retry-failed ──────────────────────────


def test_endpoint_level_resume_and_retry_failed(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=6, batch_size=6)
    cfg.retry.max_attempts = 1
    cfg.retry.backoff_base_seconds = 0.0
    fail_id = "TEST-0002"
    cfg.simulated.fail_exact = [[fail_id, "chemistry"]]
    db = setup_state(cfg)
    db.close()

    engine1, final1 = run_engine(cfg)
    assert final1 == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        counts = db.record_status_counts()
        assert counts.get(PARTIAL) == 1
        assert counts.get(COMPLETE) == 5
        # chemistry job FAILED; other jobs COMPLETE
        chem = db.get_endpoint_job(fail_id, "chemistry")
        assert chem["status"] == "FAILED"
        assert db.get_endpoint_job(fail_id, "piezometry")["status"] == "COMPLETE"
        piez_hash = db.get_endpoint_job(fail_id, "piezometry")["response_hash"]
        # error history preserved
        history = json.loads(chem["error_history_json"])
        assert len(history) >= 1
        assert db.failure_count() >= 1
        net_before = db.get_counter("network_requests")
    finally:
        db.close()

    # Outage ends: clear the failure injection, retry only chemistry.
    cfg.simulated.fail_exact = []
    db = open_db(cfg)
    try:
        reset = db.reset_failed_endpoints(endpoint="chemistry")
        assert reset == 1
        db.reset_records_to_pending([fail_id])
    finally:
        db.close()

    engine2, final2 = run_engine(cfg)
    assert final2 == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 6
        # Only the chemistry branch was re-fetched (1 network request).
        assert counter_delta(db, "network_requests", net_before) == 1
        # Non-chemistry branches preserved their original response hashes.
        assert db.get_endpoint_job(fail_id, "piezometry")["response_hash"] == piez_hash
    finally:
        db.close()


# ── 5. bounded retries ───────────────────────────────────────────────


def test_bounded_retries_no_endless_loop(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=3, batch_size=3)
    cfg.retry.max_attempts = 2
    cfg.retry.backoff_base_seconds = 0.0
    cfg.retry.backoff_max_seconds = 0.0
    # This test isolates per-endpoint retry bounds; disable breaker tripping
    # (circuit-breaker behavior has its own dedicated tests).
    cfg.circuit_breaker.failure_ratio_threshold = 1.0
    cfg.simulated.fail_exact = [
        [f"TEST-{i:04d}", ep]
        for i in (1, 2)
        for ep in (
            "piezometry",
            "lithology",
            "pumping_tests",
            "hydrometry",
            "chemistry",
            "technical",
            "documents",
        )
    ]
    db = setup_state(cfg)
    db.close()

    engine, final = run_engine(cfg)
    assert final == ENGINE_COMPLETE

    db = open_db(cfg)
    try:
        # Two failed records: each endpoint exactly max_attempts attempts.
        for i in (1, 2):
            rec = f"TEST-{i:04d}"
            assert db.get_record(rec)["status"] == FAILED
            for ep in (
                "piezometry",
                "lithology",
                "pumping_tests",
                "hydrometry",
                "chemistry",
                "technical",
                "documents",
            ):
                job = db.get_endpoint_job(rec, ep)
                assert job["status"] == "FAILED"
                assert job["attempts"] == 2
        # Third record unaffected
        assert db.get_record("TEST-0003")["status"] == COMPLETE
    finally:
        db.close()


# ── 6. stale RUNNING recovery ────────────────────────────────────────


def test_stale_running_recovery(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=8, batch_size=8)
    db = setup_state(cfg)
    try:
        # Simulate a crashed process: mark a record + some jobs RUNNING with
        # an old last_attempt_at.
        db.mark_record_running("TEST-0001", 1)
        db.ensure_endpoint_jobs(
            "TEST-0001",
            [
                "piezometry",
                "lithology",
                "pumping_tests",
                "hydrometry",
                "chemistry",
                "technical",
                "documents",
            ],
        )
        db.mark_job_running("TEST-0001", "piezometry")
        db.mark_job_running("TEST-0001", "lithology")
        # Simulate a crash that happened long ago (old last_attempt_at).
        db._execute(
            "UPDATE endpoint_jobs SET last_attempt_at = '2000-01-01T00:00:00+00:00' "
            "WHERE record_id = 'TEST-0001'"
        )
        stale_before = "2001-01-01T00:00:00+00:00"
        recovered = db.recover_stale_running(
            stale_before,
            [
                "piezometry",
                "lithology",
                "pumping_tests",
                "hydrometry",
                "chemistry",
                "technical",
                "documents",
            ],
        )
        assert recovered["jobs"] == 2
        assert db.get_endpoint_job("TEST-0001", "piezometry")["status"] == "PENDING"
    finally:
        db.close()

    # Full resume after simulated crash completes without duplication.
    engine, final = run_engine(cfg)
    assert final == ENGINE_COMPLETE
    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 8
        assert db.get_record("TEST-0001")["status"] == COMPLETE
    finally:
        db.close()


# ── 7. atomic writes ─────────────────────────────────────────────────


def test_atomic_write_preserves_previous_artifact(tmp_path: Path) -> None:
    from aquascan.national.io_utils import atomic_write_text

    target = tmp_path / "artifact.json"
    atomic_write_text(target, '{"v": 1}')
    before = target.read_text()

    # Simulate a crash during serialization (os.replace fails).
    import aquascan.national.io_utils as io_utils

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(io_utils.os, "replace", failing_replace)
    try:
        with pytest.raises(OSError):
            atomic_write_text(target, '{"v": 2}')
    finally:
        monkeypatch.undo()
    # Previous valid artifact intact; no stray tmp files.
    assert target.read_text() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_never_overwrites_different_response_silently(tmp_path: Path) -> None:
    cache = DurableCache(tmp_path / "cache", "v1")
    r1 = cache.put("A", "piezometry", [{"v": 1}], "hash1")
    assert r1["stored"] is True and r1["changed"] is False
    r2 = cache.put("A", "piezometry", [{"v": 2}], "hash2")
    assert r2["changed"] is True
    # Original preserved under the main key; new version stored separately.
    data, h = cache.get("A", "piezometry")
    assert h == "hash1"
    assert data == [{"v": 1}]
    assert cache.cache_stats()["changed_versions"] == 1
    # Identical content → no-op, no version pile-up.
    r3 = cache.put("A", "piezometry", [{"v": 1}], "hash1")
    assert r3["changed"] is False and r3["stored"] is False
    assert cache.cache_stats()["changed_versions"] == 1


def test_cache_integrity_verification(tmp_path: Path) -> None:
    cache = DurableCache(tmp_path / "cache", "v1")
    cache.put("A", "piezometry", [{"v": 1}])  # hash computed automatically
    check = cache.verify_integrity()
    assert check["valid"]
    # Corrupt the stored raw response
    entry_path = cache._main_path("A", "piezometry")
    entry = json.loads(entry_path.read_text())
    entry["raw_response"] = [{"v": 999}]
    entry_path.write_text(json.dumps(entry))
    check = cache.verify_integrity()
    assert not check["valid"]
    assert check["corrupted"] == 1


# ── 8. checkpoint integrity ──────────────────────────────────────────


def test_checkpoint_integrity(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=10, batch_size=10)
    cfg.checkpoint.every_n_records = 1  # checkpoint every record
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    db = open_db(cfg)
    try:
        assert db.checkpoint_count() >= 2
        assert db.verify_checkpoint(db.latest_checkpoint()["checkpoint_id"])
        # Tamper → verification fails
        cp = db.latest_checkpoint()
        db._execute(
            "UPDATE checkpoints SET records_complete = records_complete + 1 "
            "WHERE checkpoint_id = ?",
            (cp["checkpoint_id"],),
        )
        assert not db.verify_checkpoint(cp["checkpoint_id"])
    finally:
        db.close()


# ── 9. circuit breaker ───────────────────────────────────────────────


def test_circuit_breaker_opens_and_closes() -> None:
    cb = CircuitBreaker(
        CircuitBreakerConfig(window_size=10, failure_ratio_threshold=0.5, cooldown_seconds=0.3)
    )
    for _ in range(6):
        cb.record(False)
    assert not cb.is_open()  # window not full yet
    for _ in range(4):
        cb.record(False)
    assert cb.is_open()  # 10/10 failures → ratio 1.0 > 0.5
    assert cb.state()["trip_count"] == 1
    time.sleep(0.35)
    assert not cb.is_open()  # cooldown elapsed → closed


def test_network_failures_trip_breaker_and_finish(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=20, batch_size=20)
    cfg.retry.max_attempts = 1
    cfg.simulated.failure_rate = 1.0
    cfg.circuit_breaker.window_size = 10
    cfg.circuit_breaker.cooldown_seconds = 0.05
    db = setup_state(cfg)
    db.close()

    engine, final = run_engine(cfg)
    # All records FAILED terminal (no infinite retry); run reaches completion
    # or explicit network-failure verdict — never hangs.
    assert final in (ENGINE_COMPLETE, ENGINE_FAILED)
    assert engine.circuit_breaker.state()["trip_count"] >= 1

    db = open_db(cfg)
    try:
        counts = db.record_status_counts()
        assert counts.get(FAILED, 0) > 0
        assert db.telemetry_invariant_holds()
        assert db.telemetry()["failed_network_requests"] > 0
    finally:
        db.close()


# ── 10. rate limiting ────────────────────────────────────────────────


def test_rate_limiter_enforces_min_interval() -> None:
    rl = RateLimiter(requests_per_second=50.0)  # 20 ms min interval
    t0 = time.monotonic()
    for _ in range(10):
        rl.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.15  # 9 × 0.02 ≈ 0.18s


def test_rate_limiter_zero_disabled() -> None:
    rl = RateLimiter(requests_per_second=0.0)
    t0 = time.monotonic()
    for _ in range(50):
        rl.wait()
    assert time.monotonic() - t0 < 0.5


# ── 11. batch / partition integrity ──────────────────────────────────


def test_partition_integrity_after_run(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=22, batch_size=10)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    parts = sorted(cfg.interim_dir.glob("part-*.parquet"))
    assert len(parts) == 3
    total = 0
    all_ids: list[str] = []
    for part in parts:
        df = pd.read_parquet(part)
        total += len(df)
        all_ids.extend(df["igme_id"].tolist())
        sidecar = json.loads(part.with_suffix(".parquet.hash.json").read_text())
        assert sidecar["partition_sha256"] == sha256_file(part)
        assert sidecar["record_count"] == len(df)
    assert total == 22
    assert len(set(all_ids)) == 22  # no duplicates across partitions


# ── 12. disk critical pause ──────────────────────────────────────────


def test_disk_critical_stops_scheduling_safely(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=10, batch_size=10)
    db = setup_state(cfg)
    db.close()

    engine = NationalEnrichmentEngine(cfg)
    # Force critical disk from the very first check.
    engine._disk_state = lambda: "DISK_CRITICAL"  # type: ignore[method-assign]
    final = engine.run(quiet=True)
    assert final == ENGINE_DISK_CRITICAL

    db = open_db(cfg)
    try:
        # No work lost: records remain PENDING and resumable.
        assert db.record_status_counts().get(PENDING, 0) == 10
        assert db.engine_state() == ENGINE_DISK_CRITICAL
    finally:
        db.close()


# ── 13. QA + quarantine ──────────────────────────────────────────────


def test_qa_warning_and_quarantine_paths(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=4, batch_size=4)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    db = open_db(cfg)
    try:
        # Simulated piezometry rows are well-formed; QA should find no
        # critical violations for the standard fixture.
        qa = db.qa_counts()
        assert qa["CRITICAL"] == 0
    finally:
        db.close()

    # Force a normalization crash → QUARANTINED (never silently fixed).
    cfg2 = make_config(tmp_path / "r2", n=4, batch_size=4)
    db = setup_state(cfg2)
    db.close()
    import aquascan.national.engine as engine_mod

    original = engine_mod.NationalEnrichmentEngine._finalize_record
    calls = {"n": 0}

    def exploding(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom: normalization failure")
        return original(self, *args, **kwargs)

    engine_mod.NationalEnrichmentEngine._finalize_record = exploding  # type: ignore[method-assign]
    try:
        engine2 = NationalEnrichmentEngine(cfg2)
        final = engine2.run(quiet=True)
        assert final == ENGINE_COMPLETE
    finally:
        engine_mod.NationalEnrichmentEngine._finalize_record = original

    db = open_db(cfg2)
    try:
        counts = db.record_status_counts()
        assert counts.get(QUARANTINED) == 1
        assert counts.get(COMPLETE) == 3
        assert db.qa_counts()["CRITICAL"] == 1
        quarantined = db._query_one("SELECT * FROM records WHERE status = 'QUARANTINED'")
        assert "boom" in (quarantined["quarantine_reason"] or "")
        # The quarantined record is preserved in the partition (never dropped).
        df = pd.read_parquet(cfg2.interim_dir / "part-B00001.parquet")
        assert (df["record_status"] == "QUARANTINED").sum() == 1
        assert len(df) == 4
    finally:
        db.close()


# ── 14. finalize ─────────────────────────────────────────────────────


def test_finalize_assembles_valid_dataset(tmp_path: Path) -> None:
    from aquascan.national.finalize import finalize

    cfg = make_config(tmp_path, n=12, batch_size=5)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    db = open_db(cfg)
    try:
        report = finalize(cfg, db)
    finally:
        db.close()

    assert report["valid"], report["errors"]
    assert report["record_count"] == 12
    assert report["duplicate_ids"] == 0
    assert report["missing_ids"] == 0
    assert cfg.final_dataset.exists()
    df = pd.read_parquet(cfg.final_dataset)
    assert len(df) == 12
    assert df["igme_id"].is_unique
    # Manifest written
    manifest = json.loads((cfg.reports_dir / "national_enrichment_manifest_v1.json").read_text())
    assert manifest["dataset_sha256"] == report["dataset_sha256"]
    assert manifest["record_count"] == 12


def test_finalize_refuses_incomplete(tmp_path: Path) -> None:
    from aquascan.national.finalize import finalize

    cfg = make_config(tmp_path, n=12, batch_size=5)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg, limit=6)  # only half the records terminal

    db = open_db(cfg)
    try:
        report = finalize(cfg, db)
    finally:
        db.close()
    assert not report["valid"]
    assert any("not terminal" in e for e in report["errors"])
    assert not cfg.final_dataset.exists()


# ── 15. verify detects corruption ────────────────────────────────────


def test_verify_detects_cache_corruption(tmp_path: Path) -> None:
    from aquascan.national.verify import run_verify

    cfg = make_config(tmp_path, n=6, batch_size=6)
    db = setup_state(cfg)
    db.close()
    engine, _ = run_engine(cfg)

    # Corrupt one cached response
    entry_path = engine.cache._main_path("TEST-0001", "piezometry")
    entry = json.loads(entry_path.read_text())
    entry["raw_response"] = [{"v": -1}]
    entry_path.write_text(json.dumps(entry))

    report = run_verify(cfg, check_canary=False)
    assert not report["valid"]
    assert report["checks"]["cache_hashes"]["corrupted"] == 1


# ── 16. no Phase 0 exposure ──────────────────────────────────────────


def test_no_raw_payloads_in_public_reports(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=6, batch_size=6)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    # The status report and progress JSON never include raw API payloads.
    from aquascan.national.progress import compute_status

    db = open_db(cfg)
    try:
        cache = DurableCache(cfg.state_dir / "cache", cfg.request_version)
        status = compute_status(cfg, db, cache)
    finally:
        db.close()
    blob = json.dumps(status)
    assert "raw_response" not in blob
    assert '"Id":' not in blob
    progress = json.loads((cfg.reports_dir / "national_enrichment_progress_v1.json").read_text())
    assert "raw_response" not in json.dumps(progress)


# ── 17. SIGINT graceful shutdown (subprocess) ────────────────────────


def _cli_env(config_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["AQUASCAN_NATIONAL_CONFIG"] = str(config_path)
    return env


def _cli(args: list[str], config_path: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aquascan.cli", "national", *args],
        cwd=str(REPO),
        env=_cli_env(config_path),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_cli_config(cfg: NationalConfig) -> Path:
    from aquascan.national.config import save_config

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)
    # save_config writes to national_enrichment_config_v1.json inside reports_dir
    return cfg.reports_dir / "national_enrichment_config_v1.json"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_sigint_graceful_shutdown_and_resume(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=60, batch_size=20)
    cfg.simulated.latency_seconds = 0.02
    cfg.canary.enabled = False
    config_path = _write_cli_config(cfg)

    # Seed via the CLI init path (source hash + meta + batches).
    r = _cli(["init", "--source", str(cfg.source), "--simulated"], config_path)
    assert r.returncode == 0, r.stderr

    proc = subprocess.Popen(
        [sys.executable, "-m", "aquascan.cli", "national", "start", "--simulated"],
        cwd=str(REPO),
        env=_cli_env(config_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.2)
    proc.send_signal(signal.SIGINT)
    try:
        out, err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("engine did not exit after SIGINT")
    assert proc.returncode == 0, f"SIGINT not graceful: {out} {err}"
    assert "PAUSED" in out or "Paused" in out or "PAUSED" in err

    # Checkpoint must exist and state must be PAUSED (resumable).
    db = open_db(cfg)
    try:
        assert db.engine_state() == ENGINE_PAUSED
        assert db.latest_checkpoint() is not None
        completed = db.record_status_counts().get(COMPLETE, 0)
        assert 0 < completed < 60  # interrupted mid-run, nothing lost
    finally:
        db.close()

    # Resume to completion; no lost work, no duplicates.
    r = _cli(["resume", "--simulated"], config_path)
    assert r.returncode == 0, r.stderr
    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 60
        assert db.engine_state() == ENGINE_COMPLETE
        assert db.telemetry_invariant_holds()
    finally:
        db.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_sigkill_crash_recovery(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=60, batch_size=20)
    cfg.simulated.latency_seconds = 0.02
    cfg.canary.enabled = False
    config_path = _write_cli_config(cfg)
    r = _cli(["init", "--source", str(cfg.source), "--simulated"], config_path)
    assert r.returncode == 0, r.stderr

    proc = subprocess.Popen(
        [sys.executable, "-m", "aquascan.cli", "national", "start", "--simulated"],
        cwd=str(REPO),
        env=_cli_env(config_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    proc.kill()  # SIGKILL — no chance to checkpoint
    proc.wait(timeout=30)

    # Resume after the hard crash: everything completes exactly once.
    r = _cli(["resume", "--simulated"], config_path)
    assert r.returncode == 0, r.stderr

    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 60
        # No duplicate logical completion: every endpoint job COMPLETE once
        # with a response hash; PRIMARY KEY (record_id, endpoint) guarantees
        # uniqueness, and every job must be COMPLETE.
        rows = db._query("SELECT status, COUNT(*) AS n FROM endpoint_jobs GROUP BY status")
        assert {row["status"]: row["n"] for row in rows} == {"COMPLETE": 60 * 7}
        assert db.telemetry_invariant_holds()
    finally:
        db.close()

    # Finalize the recovered dataset.
    r = _cli(["finalize"], config_path)
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(cfg.final_dataset)
    assert len(df) == 60
    assert df["igme_id"].is_unique


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_pause_via_control_file(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=40, batch_size=20)
    cfg.simulated.latency_seconds = 0.02
    cfg.canary.enabled = False
    config_path = _write_cli_config(cfg)
    r = _cli(["init", "--source", str(cfg.source), "--simulated"], config_path)
    assert r.returncode == 0, r.stderr

    proc = subprocess.Popen(
        [sys.executable, "-m", "aquascan.cli", "national", "start", "--simulated"],
        cwd=str(REPO),
        env=_cli_env(config_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    r = _cli(["pause"], config_path)
    assert r.returncode == 0, r.stderr
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("engine did not pause")

    db = open_db(cfg)
    try:
        assert db.engine_state() == ENGINE_PAUSED
        completed = db.record_status_counts().get(COMPLETE, 0)
        assert 0 < completed < 40
    finally:
        db.close()

    r = _cli(["resume", "--simulated"], config_path)
    assert r.returncode == 0, r.stderr
    db = open_db(cfg)
    try:
        assert db.record_status_counts().get(COMPLETE) == 40
    finally:
        db.close()


# ── 18. status command works when engine is not running ──────────────


def test_status_without_running_engine(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=6, batch_size=6)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg)

    from aquascan.national.progress import compute_status

    db = open_db(cfg)
    try:
        cache = DurableCache(cfg.state_dir / "cache", cfg.request_version)
        status = compute_status(cfg, db, cache)
        assert status["total"] == 6
        assert status["complete"] == 6
        assert status["engine_status"] in ("COMPLETE", "RUNNING", "PAUSED", "STOPPED")
        assert "eta_estimate_note" in status
    finally:
        db.close()


# ── 19. session history ──────────────────────────────────────────────


def test_session_history_recorded(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, n=10, batch_size=10)
    db = setup_state(cfg)
    db.close()
    run_engine(cfg, limit=4)
    run_engine(cfg, limit=10)

    db = open_db(cfg)
    try:
        sessions = db.sessions()
        assert len(sessions) >= 2
        first = sessions[-1]
        assert first["records_completed"] == 4
        assert first["status"] == ENGINE_PAUSED
        second = sessions[0]
        assert second["records_completed"] == 6
        assert second["network_requests"] > 0
    finally:
        db.close()
