"""Seismic deterministic pipeline.

SEISMIC_RAW -> SEISMIC_CALIBRATED -> SEISMIC_QC -> SEISMIC_FILTERED_VIEW
        -> SEISMIC_FIRST_ARRIVAL_PICKS -> SEISMIC_TRAVEL_TIMES
        -> SEISMIC_VELOCITY_MODEL -> SEISMIC_FEATURES

FILTERED_VIEW never replaces RAW. Every filter configuration is versioned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aquascan.physical.common import (
    ProcessingStage,
    compute_artifact_hash,
    make_artifact_ref,
)
from aquascan.physical.models import (
    SeismicFeatures,
    SeismicObservation,
    SeismicVelocityModel,
)
from aquascan.physical.seismic.backends import (
    InterceptTime1DSeismicBackend,
    SeismicInversionBackend,
)

PROCESSING_VERSION = "0.1.0"


class SeismicPipeline:
    """Runs the deterministic seismic processing chain with provenance."""

    def __init__(self, backend: SeismicInversionBackend | None = None) -> None:
        self.backend = backend or InterceptTime1DSeismicBackend()

    # ── Stage 1: RAW ─────────────────────────────────────────────────
    def stage_raw(self, observation: SeismicObservation) -> dict[str, Any]:
        payload = observation.model_dump(mode="json")
        raw_hash = compute_artifact_hash(payload)
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_RAW,
            payload,
            processing_version=PROCESSING_VERSION,
        )
        # RAW artifact id == RAW content hash for a consistent lineage chain.
        ref.artifact_id = raw_hash
        return {"artifact": ref.to_dict(), "raw_hash": raw_hash, "data": payload}

    # ── Stage 2: Calibrated ──────────────────────────────────────────
    def stage_calibrated(
        self,
        observation: SeismicObservation,
        raw_hash: str,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> dict[str, Any]:
        """Linear calibration of waveforms (public-domain AutoCal-style)."""
        shots: list[dict[str, object]] = []
        for s in observation.shots:
            sd = s.model_dump(mode="json")
            waves = sd.get("raw_waveforms") or []
            sd["calibrated_waveforms"] = [[gain * v + offset for v in trace] for trace in waves]
            sd["calibration"] = {"gain": gain, "offset": offset}
            shots.append(sd)
        payload = {
            "survey_id": observation.survey_id,
            "blinded_site_id": observation.blinded_site_id,
            "shots": shots,
        }
        param_hash = compute_artifact_hash({"gain": gain, "offset": offset})
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_CALIBRATED,
            payload,
            parent_artifact=raw_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=param_hash,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 3: QC ──────────────────────────────────────────────────
    def stage_qc(
        self,
        calibrated: dict[str, Any],
        calibrated_hash: str,
        *,
        min_snr: float = 3.0,
        max_dead_trace_pct: float = 10.0,
    ) -> dict[str, Any]:
        """Deterministic QC: dead/noisy channels, SNR, trigger validity."""
        shots_out: list[dict[str, object]] = []
        for s in calibrated["shots"]:
            waves = np.asarray(s.get("calibrated_waveforms") or s.get("raw_waveforms") or [])
            trace_flags: list[list[str]] = []
            snrs: list[float] = []
            if waves.ndim == 2 and waves.shape[0] > 0:
                for tr in waves:
                    flags: list[str] = []
                    rms = float(np.sqrt(np.mean(tr**2))) if len(tr) else 0.0
                    peak = float(np.max(np.abs(tr))) if len(tr) else 0.0
                    if rms == 0.0 or peak == 0.0:
                        flags.append("dead_trace")
                    if peak > 0 and rms > 0:
                        snr = peak / (rms + 1e-12)
                        snrs.append(float(snr))
                        if snr < min_snr:
                            flags.append("low_snr")
                    trace_flags.append(flags)
            n_dead = sum(1 for f in trace_flags if "dead_trace" in f)
            dead_pct = 100.0 * n_dead / len(trace_flags) if trace_flags else 100.0
            shot_flags: list[str] = []
            if dead_pct > max_dead_trace_pct:
                shot_flags.append("excessive_dead_traces")
            if s.get("trigger_timestamp") is None:
                shot_flags.append("missing_trigger")
            s["trace_qc_flags"] = trace_flags
            s["mean_snr"] = float(np.mean(snrs)) if snrs else None
            s["shot_qc_flags"] = shot_flags
            s["shot_qc_pass"] = len(shot_flags) == 0
            shots_out.append(s)
        payload = {
            "survey_id": calibrated["survey_id"],
            "blinded_site_id": calibrated["blinded_site_id"],
            "n_shots": len(shots_out),
            "shots": shots_out,
        }
        param_hash = compute_artifact_hash(
            {"min_snr": min_snr, "max_dead_trace_pct": max_dead_trace_pct}
        )
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_QC,
            payload,
            parent_artifact=calibrated_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=param_hash,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 4: Filtered view (derived only) ────────────────────────
    def stage_filtered(
        self,
        qc: dict[str, Any],
        qc_hash: str,
        *,
        lowpass_hz: float | None = 100.0,
        filter_order: int = 2,
    ) -> dict[str, Any]:
        """Deterministic low-pass filtering on CALIBRATED copies only.

        RAW is never replaced. The filter config is versioned.
        """
        from scipy.signal import butter, sosfilt

        shots_out: list[dict[str, object]] = []
        for s in qc["shots"]:
            waves = np.asarray(s.get("calibrated_waveforms") or s.get("raw_waveforms") or [])
            filtered: list[list[float]] = []
            if waves.ndim == 2 and waves.shape[0] > 0 and lowpass_hz is not None:
                fs = float(s.get("sample_rate_hz") or 1000.0)
                nyq = fs / 2.0
                if 0 < lowpass_hz < nyq:
                    sos = butter(filter_order, lowpass_hz / nyq, btype="low", output="sos")
                    for tr in waves:
                        y = sosfilt(sos, tr)
                        filtered.append([float(v) for v in y])
                else:
                    filtered = [[float(v) for v in tr] for tr in waves]
            else:
                filtered = [[float(v) for v in tr] for tr in waves]
            s["filtered_waveforms"] = filtered
            s["filter_config"] = {"lowpass_hz": lowpass_hz, "filter_order": filter_order}
            shots_out.append(s)
        payload = {
            "survey_id": qc["survey_id"],
            "blinded_site_id": qc["blinded_site_id"],
            "shots": shots_out,
            "note": "Derived view only; RAW waveforms remain immutable.",
        }
        param_hash = compute_artifact_hash({"lowpass_hz": lowpass_hz, "filter_order": filter_order})
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_FILTERED_VIEW,
            payload,
            parent_artifact=qc_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=param_hash,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 5: First-arrival picks ─────────────────────────────────
    def stage_picks(
        self,
        filtered: dict[str, Any],
        filtered_hash: str,
        *,
        threshold_factor: float = 3.0,
        method: str = "automatic",
    ) -> dict[str, Any]:
        """Deterministic automatic first-break picking with uncertainty.

        A pick's method and review status are preserved. Existing manual
        picks are never silently overwritten (review_status tracked).
        """
        picks: list[dict[str, object]] = []
        for s in filtered["shots"]:
            waves = np.asarray(s.get("filtered_waveforms") or s.get("raw_waveforms") or [])
            shot_id = str(s.get("shot_id", "unknown"))
            shot_x = float(s.get("shot_x", 0.0))
            receiver_ids = s.get("receiver_ids") or []
            receiver_x = s.get("receiver_x") or []
            if waves.ndim == 2 and waves.shape[0] > 0:
                noise_win = min(50, max(10, waves.shape[1] // 10))
                for ri, tr in enumerate(waves):
                    noise_rms = float(np.sqrt(np.mean(tr[:noise_win] ** 2)))
                    threshold = threshold_factor * noise_rms
                    above = np.where(np.abs(tr) > threshold)[0]
                    if len(above):
                        pick_idx = int(above[0])
                        pick_time = pick_idx / float(s.get("sample_rate_hz") or 1000.0)
                        receiver_id = str(receiver_ids[ri]) if ri < len(receiver_ids) else f"R{ri}"
                        rx = float(receiver_x[ri]) if ri < len(receiver_x) else 0.0
                        picks.append(
                            {
                                "shot_id": shot_id,
                                "receiver_id": receiver_id,
                                "shot_x": shot_x,
                                "receiver_x": rx,
                                "pick_time_s": pick_time,
                                "pick_uncertainty_s": round(
                                    1.0 / float(s.get("sample_rate_hz") or 1000.0), 6
                                ),
                                "method": method,
                                "review_status": "pending",
                            }
                        )
        payload = {
            "survey_id": filtered["survey_id"],
            "blinded_site_id": filtered["blinded_site_id"],
            "picks": picks,
        }
        param_hash = compute_artifact_hash({"threshold_factor": threshold_factor, "method": method})
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_FIRST_ARRIVAL_PICKS,
            payload,
            parent_artifact=filtered_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=param_hash,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 6: Travel times ────────────────────────────────────────
    def stage_travel_times(
        self,
        picks: dict[str, Any],
        picks_hash: str,
    ) -> dict[str, Any]:
        """Convert picks into source-receiver travel times with offsets.

        Geometry (shot_x / receiver_x) is propagated from the shots through
        the picks stage, so offsets are real survey offsets.
        """
        rows: list[dict[str, object]] = []
        for p in picks["picks"]:
            sx = float(p.get("shot_x", 0.0))
            rx = float(p.get("receiver_x", 0.0))
            rows.append(
                {
                    "shot_id": p["shot_id"],
                    "receiver_id": p["receiver_id"],
                    "offset_m": abs(rx - sx),
                    "travel_time_s": p["pick_time_s"],
                    "pick_uncertainty_s": p.get("pick_uncertainty_s"),
                }
            )
        payload = {
            "survey_id": picks["survey_id"],
            "blinded_site_id": picks["blinded_site_id"],
            "travel_times": rows,
            "geometry_note": "Offsets derived from shot/receiver geometry.",
        }
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_TRAVEL_TIMES,
            payload,
            parent_artifact=picks_hash,
            processing_version=PROCESSING_VERSION,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 7: Velocity model ──────────────────────────────────────
    def stage_velocity(
        self,
        travel_times: dict[str, Any],
        travel_times_hash: str,
    ) -> dict[str, Any]:
        """Run the configured seismic backend on travel times."""
        rows = travel_times["travel_times"]
        if not rows:
            raise ValueError("No travel times available for velocity inversion.")
        offsets = np.array([r["offset_m"] for r in rows], dtype=float)
        times = np.array([r["travel_time_s"] for r in rows], dtype=float)
        metadata = {
            "survey_id": travel_times["survey_id"],
            "blinded_site_id": travel_times["blinded_site_id"],
            "parent_artifact": travel_times_hash,
        }
        model = self.backend.invert(offsets, times, metadata=metadata)
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_VELOCITY_MODEL,
            model.model_dump(mode="json"),
            parent_artifact=travel_times_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=compute_artifact_hash({"backend": self.backend.name}),
        )
        return {"artifact": ref.to_dict(), "data": model}

    # ── Stage 8: Features ────────────────────────────────────────────
    def stage_features(
        self,
        model: SeismicVelocityModel,
        model_hash: str,
    ) -> dict[str, Any]:
        """Target-independent deterministic seismic features."""
        vp = np.asarray(model.vp, dtype=float)
        cov = np.asarray(model.ray_coverage, dtype=float) if model.ray_coverage else None

        features = SeismicFeatures(
            survey_id=model.survey_id,
            blinded_site_id=model.blinded_site_id,
            vp_mean_m_s=float(np.mean(vp)),
            vp_median_m_s=float(np.median(vp)),
            vp_std_m_s=float(np.std(vp)),
            vertical_gradient_max=float(np.max(np.abs(np.diff(vp)))) if len(vp) > 1 else None,
            lateral_gradient_max=0.0,
            velocity_contrast_max=(float(np.max(np.abs(np.diff(vp)))) if len(vp) > 1 else None),
            coverage_mean=float(np.mean(cov)) if cov is not None else None,
            signal_quality_mean=None,
            quality_score=None,
            parent_artifact=model_hash,
            processing_provenance={
                "processing_step": "seismic_feature_extraction",
                "processing_version": PROCESSING_VERSION,
            },
        )
        ref = make_artifact_ref(
            ProcessingStage.SEISMIC_FEATURES,
            features.model_dump(mode="json"),
            parent_artifact=model_hash,
            processing_version=PROCESSING_VERSION,
        )
        return {"artifact": ref.to_dict(), "data": features}

    # ── Run the full chain ───────────────────────────────────────────
    def run(
        self,
        observation: SeismicObservation,
        *,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> dict[str, Any]:
        chain: dict[str, Any] = {}
        raw = self.stage_raw(observation)
        chain["raw"] = raw
        cal = self.stage_calibrated(observation, raw["raw_hash"], gain=gain, offset=offset)
        chain["calibrated"] = cal
        qc = self.stage_qc(cal["data"], cal["artifact"]["artifact_id"])
        chain["qc"] = qc
        filt = self.stage_filtered(qc["data"], qc["artifact"]["artifact_id"])
        chain["filtered"] = filt
        picks = self.stage_picks(filt["data"], filt["artifact"]["artifact_id"])
        chain["picks"] = picks
        tt = self.stage_travel_times(picks["data"], picks["artifact"]["artifact_id"])
        chain["travel_times"] = tt
        vel = self.stage_velocity(tt["data"], tt["artifact"]["artifact_id"])
        chain["velocity"] = vel
        feats = self.stage_features(vel["data"], vel["artifact"]["artifact_id"])
        chain["features"] = feats
        return chain
