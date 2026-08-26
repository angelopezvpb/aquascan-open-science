"""Normalization of raw API responses into normalized evidence.

Covers steps 1, 5, 6, 7, 8, 9, 10.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from aquascan.data.models import (
    LithologyAnomaly,
    LithologyInterval,
    LithologyValidationResult,
    NormalizedGroundTruthRecord,
    PumpingTestRecord,
    TargetReadiness,
    TargetReadinessAssessment,
    TemporalRelevanceStatus,
    TemporalSummary,
    WaterLevelObservation,
    WaterPresenceDetermination,
    WaterPresenceStatus,
)

# ── Temporal thresholds (documented) ────────────────────────────────
RECENT_YEARS = 5.0
MULTI_DECADE_SPAN_YEARS = 20.0
MULTI_DECADE_MIN_MEASUREMENTS = 5


def clean_number(val: object) -> float | None:
    """Normalize Spanish locale numbers (comma decimal) to float."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int | float):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def clean_date(val: object) -> str | None:
    """Normalize a date value to ISO 8601 (date only)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Try ISO parse
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    # Try common formats
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s[:10] if len(s) >= 10 else s


# ── 1. Water-level observations (steps 1, 6) ────────────────────────


def parse_water_level_observations(
    raw: list[dict[str, Any]], igme_id: str
) -> list[WaterLevelObservation]:
    """Parse QueryAGPZ rows into WaterLevelObservation records.

    Preserves:
    - observation date
    - depth to water (Nivel → depth_to_water_m)
    - hydraulic head elevation (Valor → hydraulic_head_elevation_m)
    - source

    The Cota - Nivel = Valor relation is used only as consistency check.
    """
    observations: list[WaterLevelObservation] = []
    for row in raw:
        if not isinstance(row, dict):
            continue

        nivel = clean_number(row.get("Nivel"))
        valor = clean_number(row.get("Valor"))
        cota = clean_number(row.get("Cota"))

        # Consistency check: Valor ≈ Cota - Nivel
        residual = None
        if nivel is not None and valor is not None and cota is not None:
            residual = valor - (cota - nivel)

        # Artesian flag from Surgencia
        surgencia = str(row.get("Surgencia", "")).strip().lower()
        artesian = None
        if "surgente" in surgencia and "no" not in surgencia:
            artesian = True
        elif "no surgente" in surgencia:
            artesian = False

        obs = WaterLevelObservation(
            igme_id=igme_id,
            observation_date=clean_date(row.get("Fecha") or row.get("FechaStr")),
            depth_to_water_m=nivel,
            hydraulic_head_elevation_m=valor,
            ground_elevation_m=cota,
            source="QueryAGPZ",
            source_record_id=igme_id,
            source_hash=None,
            consistency_residual_m=residual,
            artesian=artesian,
            aquifer_designation=str(row.get("Acuifero")) if row.get("Acuifero") else None,
            raw=dict(row),
        )
        observations.append(obs)
    return observations


def compute_temporal_summary(observations: list[WaterLevelObservation]) -> TemporalSummary:
    """Compute derived temporal summary for water-level observations.

    These are DERIVED ARTIFACTS ONLY — they never replace the raw time series.
    """
    n = len(observations)
    if n == 0:
        return TemporalSummary()

    dated = [o for o in observations if o.observation_date is not None]
    dates = sorted(o.observation_date for o in dated)  # type: ignore[arg-type]

    depths = [o.depth_to_water_m for o in observations if o.depth_to_water_m is not None]
    min_depth = min(depths) if depths else None
    max_depth = max(depths) if depths else None
    median_depth = None
    if depths:
        sorted_d = sorted(depths)
        mid = len(sorted_d) // 2
        median_depth = (
            sorted_d[mid] if len(sorted_d) % 2 else (sorted_d[mid - 1] + sorted_d[mid]) / 2
        )

    first_date = dates[0] if dates else None
    last_date = dates[-1] if dates else None
    span_days = None
    if first_date and last_date:
        try:
            d0 = datetime.fromisoformat(first_date)
            d1 = datetime.fromisoformat(last_date)
            span_days = (d1 - d0).days
        except ValueError:
            span_days = None

    # Temporal relevance
    status = TemporalRelevanceStatus.DATE_UNKNOWN
    if last_date:
        try:
            last_dt = datetime.fromisoformat(last_date)
            now = datetime.now(UTC)
            age_years = (now - last_dt.replace(tzinfo=UTC)).days / 365.25
            decades = span_days is not None and span_days >= MULTI_DECADE_SPAN_YEARS * 365
            if decades and n >= MULTI_DECADE_MIN_MEASUREMENTS:
                status = TemporalRelevanceStatus.MULTI_DECADE_SERIES
            elif age_years <= RECENT_YEARS:
                status = TemporalRelevanceStatus.RECENT
            else:
                status = TemporalRelevanceStatus.HISTORICAL
        except ValueError:
            status = TemporalRelevanceStatus.DATE_UNKNOWN

    return TemporalSummary(
        num_measurements=n,
        first_measurement_date=first_date,
        last_measurement_date=last_date,
        temporal_span_days=span_days,
        min_depth_to_water_m=min_depth,
        max_depth_to_water_m=max_depth,
        median_depth_to_water_m=median_depth,
        temporal_relevance_status=status,
    )


# ── 2. Lithology (step 7) ───────────────────────────────────────────


def parse_lithology_intervals(raw: list[dict[str, Any]], igme_id: str) -> list[LithologyInterval]:
    """Parse QueryAGLI rows into LithologyInterval records."""
    intervals: list[LithologyInterval] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        interval = LithologyInterval(
            igme_id=igme_id,
            top_depth_m=clean_number(row.get("ProfTecho")),
            bottom_depth_m=clean_number(row.get("ProfMuro")),
            lithology=str(row.get("Litologia")) if row.get("Litologia") is not None else None,
            geological_age=str(row.get("Edad")) if row.get("Edad") is not None else None,
            aquifer_flagged_lithology_interval=bool(row.get("Acuifero", False)),
            order_index=int(clean_number(row.get("Numorden")) or 0),
            source="QueryAGLI",
            source_hash=None,
            raw=dict(row),
        )
        intervals.append(interval)
    return intervals


def validate_lithology_intervals(
    intervals: list[LithologyInterval],
    total_depth_m: float | None = None,
) -> LithologyValidationResult:
    """Validate lithological intervals (flag anomalies, do not repair).

    Checks:
    - interval ordering
    - overlapping intervals
    - gaps
    - negative depths
    - bottom < top
    - intervals exceeding total borehole depth
    """
    anomalies: list[LithologyAnomaly] = []
    if not intervals:
        return LithologyValidationResult(anomalies=[], is_valid=True)

    ordered = sorted(
        intervals,
        key=lambda i: (i.order_index, i.top_depth_m if i.top_depth_m is not None else 1e9),
    )

    # Check ordering (by order_index vs. depth)
    for i in range(len(ordered) - 1):
        cur = ordered[i]
        nxt = ordered[i + 1]
        if cur.bottom_depth_m is not None and nxt.top_depth_m is not None:
            if cur.bottom_depth_m > nxt.top_depth_m:
                anomalies.append(LithologyAnomaly.ORDERING_VIOLATION)

    # Check individual intervals
    for _i, interval in enumerate(intervals):
        top = interval.top_depth_m
        bottom = interval.bottom_depth_m

        # Negative depths
        if top is not None and top < 0:
            anomalies.append(LithologyAnomaly.NEGATIVE_DEPTH)
        if bottom is not None and bottom < 0:
            anomalies.append(LithologyAnomaly.NEGATIVE_DEPTH)

        # Bottom < top
        if top is not None and bottom is not None and bottom < top:
            anomalies.append(LithologyAnomaly.BOTTOM_LT_TOP)

        # Exceeds total borehole depth
        if total_depth_m is not None:
            if bottom is not None and bottom > total_depth_m:
                anomalies.append(LithologyAnomaly.EXCEEDS_TOTAL_DEPTH)

    # Check overlapping and gaps between consecutive intervals
    for i in range(len(ordered) - 1):
        cur = ordered[i]
        nxt = ordered[i + 1]
        if cur.bottom_depth_m is not None and nxt.top_depth_m is not None:
            if cur.bottom_depth_m > nxt.top_depth_m:
                anomalies.append(LithologyAnomaly.OVERLAPPING_INTERVALS)
            elif cur.bottom_depth_m < nxt.top_depth_m:
                anomalies.append(LithologyAnomaly.GAP)

    # Deduplicate anomalies
    unique_anomalies = list(dict.fromkeys(anomalies))
    return LithologyValidationResult(
        anomalies=unique_anomalies,
        is_valid=len(unique_anomalies) == 0,
    )


# ── 3. Pumping tests (step 8) ───────────────────────────────────────


def parse_pumping_tests(raw: list[dict[str, Any]], igme_id: str) -> list[PumpingTestRecord]:
    """Parse QueryAGEB rows into PumpingTestRecord.

    Specific capacity derived ONLY when flow >= 0 and drawdown > 0.
    """
    tests: list[PumpingTestRecord] = []
    for row in raw:
        if not isinstance(row, dict):
            continue

        flow = clean_number(row.get("Caudal"))
        drawdown = clean_number(row.get("DepresionM"))
        horas = clean_number(row.get("Horasdur"))
        mins = clean_number(row.get("Mindur"))

        duration = None
        if horas is not None or mins is not None:
            duration = (horas or 0) * 60 + (mins or 0)

        # Derive specific capacity
        sc = None
        sc_derived = False
        if flow is not None and drawdown is not None:
            if flow >= 0 and drawdown > 0:
                sc = flow / drawdown
                sc_derived = True

        test = PumpingTestRecord(
            igme_id=igme_id,
            test_date=clean_date(row.get("Fecha") or row.get("FechaStr")),
            flow_l_s=flow,
            drawdown_m=drawdown,
            duration_minutes=duration,
            specific_capacity_l_s_m=sc,
            specific_capacity_is_derived=sc_derived,
            transmissivity_m2_day=clean_number(row.get("Transmisividad")),
            storativity=clean_number(row.get("Coefalmace")),
            source="QueryAGEB",
            source_hash=None,
            raw=dict(row),
        )
        tests.append(test)
    return tests


# ── 4. Chemistry and hydrometry ─────────────────────────────────────


def parse_chemistry(raw: list[dict[str, Any]], igme_id: str) -> list[dict[str, Any]]:
    """Parse QueryAGAQ rows (preserve all fields)."""
    records: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        normalized: dict[str, Any] = {
            "igme_id": igme_id,
            "observation_date": clean_date(row.get("Fecha") or row.get("FechaStr")),
        }
        for key, val in row.items():
            if key in ("Fecha", "FechaStr"):
                continue
            normalized[key] = val
        records.append(normalized)
    return records


def parse_hydrometry(raw: list[dict[str, Any]], igme_id: str) -> list[dict[str, Any]]:
    """Parse QueryAGHD rows (preserve all fields)."""
    records: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        normalized: dict[str, Any] = {
            "igme_id": igme_id,
            "observation_date": clean_date(row.get("Fecha") or row.get("FechaStr")),
        }
        for key, val in row.items():
            if key in ("Fecha", "FechaStr"):
                continue
            normalized[key] = val
        records.append(normalized)
    return records


# ── 5. Water presence (step 10) ─────────────────────────────────────


def determine_water_presence(
    water_levels: list[WaterLevelObservation],
    pumping_tests: list[PumpingTestRecord],
    hydrometry: list[dict[str, Any]],
) -> WaterPresenceDetermination:
    """Determine water presence from actual hydrogeological observations.

    MUST NOT use point nature, reference flow, or aquifer flag alone.
    """
    evidence: list[str] = []

    # Positive evidence: piezometry with a depth to water observation
    has_wl = any(o.depth_to_water_m is not None for o in water_levels)
    if has_wl:
        evidence.append("piezometry_observation")

    # Positive evidence: pumping test with positive flow
    has_positive_flow = any(t.flow_l_s is not None and t.flow_l_s > 0 for t in pumping_tests)
    if has_positive_flow:
        evidence.append("pumping_test_positive_flow")

    # Negative evidence: pumping test with flow == 0 (dry well)
    has_dry_test = any(
        t.flow_l_s is not None and t.flow_l_s == 0 and t.drawdown_m is not None
        for t in pumping_tests
    )
    if has_dry_test:
        evidence.append("pumping_test_zero_flow")

    # Hydrometry: positive flow measurement
    has_hydrometry_flow = any(
        h.get("Valor") is not None
        and clean_number(h.get("Valor")) is not None
        and (clean_number(h.get("Valor")) or 0) > 0
        for h in hydrometry
    )
    if has_hydrometry_flow:
        evidence.append("hydrometry_flow")

    # Decision logic
    if has_positive_flow or has_hydrometry_flow:
        return WaterPresenceDetermination(
            status=WaterPresenceStatus.PRESENT,
            evidence_basis=evidence,
        )
    if has_dry_test and not has_positive_flow and not has_wl:
        return WaterPresenceDetermination(
            status=WaterPresenceStatus.ABSENT,
            evidence_basis=evidence,
        )
    # Piezometry alone indicates water level measured → PRESENT but weak
    if has_wl and not has_dry_test:
        return WaterPresenceDetermination(
            status=WaterPresenceStatus.PRESENT,
            evidence_basis=evidence,
        )
    return WaterPresenceDetermination(
        status=WaterPresenceStatus.UNKNOWN,
        evidence_basis=evidence,
    )


# ── 6. Target readiness (step 9) ────────────────────────────────────

TARGETS = [
    "TARGET_WATER_PRESENCE",
    "TARGET_DEPTH_TO_WATER",
    "TARGET_HYDRAULIC_HEAD_ELEVATION",
    "TARGET_AQUIFER_INTERVAL",
    "TARGET_PRODUCTIVITY",
    "TARGET_LITHOLOGY",
]


def assess_target_readiness(
    target: str,
    water_levels: list[WaterLevelObservation],
    lithology: list[LithologyInterval],
    pumping_tests: list[PumpingTestRecord],
    water_presence: WaterPresenceDetermination,
    temporal_summary: TemporalSummary,
    documentary_confidence: float,
) -> TargetReadinessAssessment:
    """Compute documentary readiness for a single target.

    Candidate Priority Score MUST NOT influence these classes.
    """
    reasons: list[str] = []
    missing: list[str] = []
    evidence_quality = 0.0
    evidence_available = False

    n_wl = len(water_levels)
    has_wl_depth = any(o.depth_to_water_m is not None for o in water_levels)
    has_wl_head = any(o.hydraulic_head_elevation_m is not None for o in water_levels)
    has_lithology = len(lithology) > 0
    has_aquifer_flag = any(i.aquifer_flagged_lithology_interval for i in lithology)
    has_positive_flow = any(t.flow_l_s is not None and t.flow_l_s > 0 for t in pumping_tests)

    temporal_relevance = temporal_summary.temporal_relevance_status

    if target == "TARGET_WATER_PRESENCE":
        if water_presence.status != WaterPresenceStatus.UNKNOWN:
            evidence_available = True
            evidence_quality = 0.8 if len(water_presence.evidence_basis) >= 2 else 0.5
        else:
            evidence_available = False
            missing.append("hydrogeological observation")
        temporal_relevance = temporal_summary.temporal_relevance_status if n_wl > 0 else None

    elif target == "TARGET_DEPTH_TO_WATER":
        if has_wl_depth:
            evidence_available = True
            evidence_quality = 1.0 if n_wl >= 2 else 0.6
        else:
            evidence_available = False
            missing.append("Nivel (depth to water)")

    elif target == "TARGET_HYDRAULIC_HEAD_ELEVATION":
        if has_wl_head:
            evidence_available = True
            evidence_quality = 1.0 if n_wl >= 2 else 0.6
        else:
            evidence_available = False
            missing.append("Valor (hydraulic head elevation)")

    elif target == "TARGET_AQUIFER_INTERVAL":
        if has_aquifer_flag:
            evidence_available = True
            evidence_quality = 0.7
        elif has_lithology:
            evidence_available = True
            evidence_quality = 0.3
            missing.append("Acuifero flag on intervals")
        else:
            evidence_available = False
            missing.append("lithology intervals")

    elif target == "TARGET_PRODUCTIVITY":
        if has_positive_flow:
            evidence_available = True
            evidence_quality = 1.0
        else:
            evidence_available = False
            missing.append("pumping test flow")

    elif target == "TARGET_LITHOLOGY":
        if has_lithology:
            evidence_available = True
            # Higher quality with age + depth
            complete = all(
                i.top_depth_m is not None and i.bottom_depth_m is not None for i in lithology
            )
            has_age = any(i.geological_age is not None for i in lithology)
            evidence_quality = 1.0 if (complete and has_age) else 0.6
        else:
            evidence_available = False
            missing.append("lithology log")

    # Deterministic readiness rules
    if not evidence_available:
        readiness = TargetReadiness.NOT_READY
    elif evidence_quality >= 0.8 and documentary_confidence >= 0.5:
        readiness = TargetReadiness.READY
    else:
        readiness = TargetReadiness.PARTIAL

    reasons.append(f"evidence_quality={evidence_quality:.2f}")
    reasons.append(f"documentary_confidence={documentary_confidence:.2f}")

    return TargetReadinessAssessment(
        target=target,
        readiness=readiness,
        evidence_available=evidence_available,
        evidence_quality=round(evidence_quality, 3),
        documentary_confidence=round(documentary_confidence, 3),
        temporal_relevance=temporal_relevance,
        missing_critical_fields=missing,
        reasons=reasons,
    )


def compute_documentary_confidence(
    water_levels: list[WaterLevelObservation],
    lithology: list[LithologyInterval],
    pumping_tests: list[PumpingTestRecord],
    chemistry: list[dict[str, Any]],
    hydrometry: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    technical: list[dict[str, Any]],
    source_changed: bool,
) -> float:
    """Compute documentary confidence (0-1) from available structured evidence."""
    families = 0
    if lithology:
        families += 1
    if water_levels:
        families += 1
    if pumping_tests:
        families += 1
    if chemistry:
        families += 1
    if hydrometry:
        families += 1

    confidence = min(1.0, families / 3.0)  # ≥3 families = full confidence

    if source_changed:
        confidence *= 0.8  # penalty for unstable source

    # Small bonus for documents/technical (structured but rarely present)
    if documents or technical:
        confidence = min(1.0, confidence + 0.1)

    return round(confidence, 3)


def build_normalized_record(
    igme_id: str,
    domain: str,
    candidate_tier: str | None,
    candidate_score: float | None,
    detail_url: str | None,
    raw: dict[str, Any],
) -> NormalizedGroundTruthRecord:
    """Build a NormalizedGroundTruthRecord from raw API responses."""
    water_levels = parse_water_level_observations(raw.get("piezometry", []), igme_id)
    lithology = parse_lithology_intervals(raw.get("lithology", []), igme_id)
    pumping_tests = parse_pumping_tests(raw.get("pumping_tests", []), igme_id)
    chemistry = parse_chemistry(raw.get("chemistry", []), igme_id)
    hydrometry = parse_hydrometry(raw.get("hydrometry", []), igme_id)

    temporal_summary = compute_temporal_summary(water_levels)
    lithology_validation = validate_lithology_intervals(
        lithology,
        total_depth_m=None,  # total depth not reliably passed here
    )

    source_changed = bool(raw.get("_source_changed", False))

    documentary_confidence = compute_documentary_confidence(
        water_levels,
        lithology,
        pumping_tests,
        chemistry,
        hydrometry,
        raw.get("documents", []),
        raw.get("technical", []),
        source_changed,
    )

    water_presence = determine_water_presence(water_levels, pumping_tests, hydrometry)

    target_readiness: dict[str, TargetReadinessAssessment] = {}
    for target in TARGETS:
        target_readiness[target] = assess_target_readiness(
            target,
            water_levels,
            lithology,
            pumping_tests,
            water_presence,
            temporal_summary,
            documentary_confidence,
        )

    # Documentary notes
    documentary_notes: list[str] = []
    if source_changed:
        documentary_notes.append("source_response_changed")

    # Provenance
    provenance = {
        "igme_id": igme_id,
        "piezometry_sha256": raw.get("piezometry_sha256", ""),
        "lithology_sha256": raw.get("lithology_sha256", ""),
        "pumping_tests_sha256": raw.get("pumping_tests_sha256", ""),
        "chemistry_sha256": raw.get("chemistry_sha256", ""),
        "hydrometry_sha256": raw.get("hydrometry_sha256", ""),
        "technical_sha256": raw.get("technical_sha256", ""),
        "documents_sha256": raw.get("documents_sha256", ""),
    }

    return NormalizedGroundTruthRecord(
        igme_id=igme_id,
        domain=domain,
        candidate_tier=candidate_tier,
        candidate_priority_score=candidate_score,
        detail_url=detail_url,
        geological_evidence=lithology,
        water_level_evidence=water_levels,
        hydraulic_evidence=pumping_tests,
        water_quality_evidence=chemistry,
        hydrometry_evidence=hydrometry,
        documentary_confidence=documentary_confidence,
        documentary_notes=documentary_notes,
        temporal_summary=temporal_summary,
        lithology_validation=lithology_validation,
        water_presence=water_presence,
        target_readiness=target_readiness,
        provenance=provenance,
        source_response_changed=source_changed,
        changed_endpoints=raw.get("_changed_endpoints", []),
    )
