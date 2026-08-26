"""ERT deterministic pipeline (Gate 10, section 4).

ERT_RAW -> ERT_CALIBRATED -> ERT_QC -> ERT_APPARENT_RESISTIVITY
        -> ERT_INVERSION_INPUT -> ERT_INVERTED_MODEL -> ERT_FEATURES

Every artifact references its parent hash, processing version, parameter
hash, and timestamp. RAW is never modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aquascan.physical.common import (
    ProcessingStage,
    compute_artifact_hash,
    make_artifact_ref,
)
from aquascan.physical.ert.backends import (
    ERTInversionBackend,
    available_ert_backends,
)
from aquascan.physical.models import (
    ERTFeatures,
    ERTInvertedSection,
    ERTObservation,
)

PROCESSING_VERSION = "0.1.0"


class ERTPipeline:
    """Runs the deterministic ERT processing chain with full provenance."""

    def __init__(self, backend: ERTInversionBackend | None = None) -> None:
        self.backend = backend or available_ert_backends()[0]

    # ── Stage 1: RAW (immutable) ─────────────────────────────────────
    def stage_raw(self, observation: ERTObservation) -> dict[str, Any]:
        """Record the RAW observation and its immutable hash."""
        raw_payload = observation.model_dump(mode="json")
        raw_hash = compute_artifact_hash(raw_payload)
        ref = make_artifact_ref(
            ProcessingStage.ERT_RAW,
            raw_payload,
            processing_version=PROCESSING_VERSION,
        )
        # The RAW artifact id IS the RAW content hash, so lineage chains
        # consistently from the immutable original.
        ref.artifact_id = raw_hash
        return {"artifact": ref.to_dict(), "raw_hash": raw_hash, "data": raw_payload}

    # ── Stage 2: Calibrated ──────────────────────────────────────────
    def stage_calibrated(
        self,
        observation: ERTObservation,
        raw_hash: str,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> dict[str, Any]:
        """Apply deterministic linear calibration to current/voltage.

        gain/offset are public-domain AutoCal-style coefficients supplied
        by the calibration artifact (Gate 7 AutoCal protocol). RAW values
        are preserved; calibrated copies are produced alongside.
        """
        quads: list[dict[str, object]] = []
        for q in observation.quadrupoles:
            qd = q.model_dump(mode="json")
            if q.injected_current_a is not None:
                qd["calibrated_current_a"] = gain * q.injected_current_a + offset
            if q.measured_voltage_v is not None:
                qd["calibrated_voltage_v"] = gain * q.measured_voltage_v + offset
            quads.append(qd)
        payload = {
            "survey_id": observation.survey_id,
            "blinded_site_id": observation.blinded_site_id,
            "calibration": {"gain": gain, "offset": offset, "method": "linear_public_domain"},
            "quadrupoles": quads,
        }
        param_hash = compute_artifact_hash({"gain": gain, "offset": offset})
        ref = make_artifact_ref(
            ProcessingStage.ERT_CALIBRATED,
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
        max_contact_resistance_ohm: float = 2000.0,
        max_reciprocity_error_pct: float = 5.0,
        min_current_a: float = 0.0,
    ) -> dict[str, Any]:
        """Deterministic QC flagging (public-domain thresholds)."""
        quads: list[dict[str, object]] = []
        for q in calibrated["quadrupoles"]:
            flags: list[str] = []
            if q.get("contact_resistance_ohm") is not None:
                if q["contact_resistance_ohm"] > max_contact_resistance_ohm:
                    flags.append("high_contact_resistance")
            if q.get("injected_current_a") is not None:
                if q["injected_current_a"] <= min_current_a:
                    flags.append("zero_or_negative_current")
            if q.get("measured_voltage_v") is None and q.get("apparent_resistance_ohm") is None:
                flags.append("missing_measurement")
            q["qc_flags"] = flags
            q["qc_pass"] = len(flags) == 0
            quads.append(q)
        n_total = len(quads)
        n_pass = sum(1 for q in quads if q["qc_pass"])
        payload = {
            "survey_id": calibrated["survey_id"],
            "blinded_site_id": calibrated["blinded_site_id"],
            "n_measurements": n_total,
            "n_pass": n_pass,
            "quality_score": round(n_pass / n_total, 4) if n_total else 0.0,
            "quadrupoles": quads,
        }
        param_hash = compute_artifact_hash(
            {
                "max_contact_resistance_ohm": max_contact_resistance_ohm,
                "max_reciprocity_error_pct": max_reciprocity_error_pct,
                "min_current_a": min_current_a,
            }
        )
        ref = make_artifact_ref(
            ProcessingStage.ERT_QC,
            payload,
            parent_artifact=calibrated_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=param_hash,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 4: Apparent resistivity ────────────────────────────────
    def stage_apparent_resistivity(
        self,
        qc: dict[str, Any],
        qc_hash: str,
    ) -> dict[str, Any]:
        """Compute apparent resistance and apparent resistivity per quadrupole.

        Uses the public-domain general 4-electrode geometric factor
        (existing aquascan.ert.basic_processing is reused via direct formula
        here to keep the pipeline self-contained and deterministic).
        """
        quads: list[dict[str, object]] = []
        for q in qc["quadrupoles"]:
            a, b, m, n = (
                float(q["electrode_a"]),
                float(q["electrode_b"]),
                float(q["electrode_m"]),
                float(q["electrode_n"]),
            )
            am = abs(a - m)
            an = abs(a - n)
            bm = abs(b - m)
            bn = abs(b - n)
            eps = 1e-12
            r = q.get("apparent_resistance_ohm")
            if r is None and q.get("calibrated_voltage_v") is not None:
                i = q.get("calibrated_current_a")
                if i is not None and i != 0:
                    r = float(q["calibrated_voltage_v"]) / float(i)
            if r is None and q.get("measured_voltage_v") is not None:
                i = q.get("injected_current_a")
                if i is not None and i != 0:
                    r = float(q["measured_voltage_v"]) / float(i)
            if r is not None and min(am, an, bm, bn) > eps:
                k = (2.0 * np.pi) / (1.0 / am - 1.0 / an - 1.0 / bm + 1.0 / bn)
                q["geometric_factor"] = float(k)
                q["apparent_resistivity_ohm_m"] = float(k) * float(r)
            q["resistance_used_ohm"] = r
            quads.append(q)
        payload = {
            "survey_id": qc["survey_id"],
            "blinded_site_id": qc["blinded_site_id"],
            "n_measurements": len(quads),
            "quadrupoles": quads,
        }
        ref = make_artifact_ref(
            ProcessingStage.ERT_APPARENT_RESISTIVITY,
            payload,
            parent_artifact=qc_hash,
            processing_version=PROCESSING_VERSION,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 5: Inversion input ─────────────────────────────────────
    def stage_inversion_input(
        self,
        apparent: dict[str, Any],
        apparent_hash: str,
    ) -> dict[str, Any]:
        """Prepare the standardized inversion input arrays."""
        x: list[float] = []
        z: list[float] = []
        rho: list[float] = []
        unc: list[float] = []
        for q in apparent["quadrupoles"]:
            rho_a = q.get("apparent_resistivity_ohm_m")
            if rho_a is None or q.get("qc_pass") is False:
                continue
            mid = (float(q["electrode_a"]) + float(q["electrode_b"])) / 2.0
            # pseudo-depth = AB/6 (standard public-domain convention)
            pseudo_z = abs(float(q["electrode_a"]) - float(q["electrode_b"])) / 6.0
            x.append(mid)
            z.append(pseudo_z)
            rho.append(float(rho_a))
            unc.append(0.1 * float(rho_a))
        payload = {
            "survey_id": apparent["survey_id"],
            "blinded_site_id": apparent["blinded_site_id"],
            "x": x,
            "z": z,
            "apparent_resistivity": rho,
            "uncertainty": unc,
            "n_points": len(x),
        }
        ref = make_artifact_ref(
            ProcessingStage.ERT_INVERSION_INPUT,
            payload,
            parent_artifact=apparent_hash,
            processing_version=PROCESSING_VERSION,
        )
        return {"artifact": ref.to_dict(), "data": payload}

    # ── Stage 6: Inverted model ──────────────────────────────────────
    def stage_inverted(
        self,
        inv_input: dict[str, Any],
        inv_input_hash: str,
    ) -> dict[str, Any]:
        """Run the configured backend and return the standardized section."""
        x = np.asarray(inv_input["x"], dtype=float)
        z = np.asarray(inv_input["z"], dtype=float)
        rho = np.asarray(inv_input["apparent_resistivity"], dtype=float)
        unc = np.asarray(inv_input.get("uncertainty") or [], dtype=float)
        if len(x) == 0:
            raise ValueError("Empty inversion input: no QC-passing measurements.")
        mesh_metadata = {
            "survey_id": inv_input["survey_id"],
            "blinded_site_id": inv_input["blinded_site_id"],
            "parent_artifact": inv_input_hash,
        }
        section = self.backend.invert(
            x,
            z,
            rho,
            uncertainty=unc if len(unc) else None,
            mesh_metadata=mesh_metadata,
        )
        section.parent_artifact = inv_input_hash
        section.processing_provenance = {
            "processing_step": f"ert_inversion[{self.backend.name}]",
            "processing_version": PROCESSING_VERSION,
            "parent_artifact": inv_input_hash,
        }
        ref = make_artifact_ref(
            ProcessingStage.ERT_INVERTED_MODEL,
            section.model_dump(mode="json"),
            parent_artifact=inv_input_hash,
            processing_version=PROCESSING_VERSION,
            parameter_hash=compute_artifact_hash({"backend": self.backend.name}),
        )
        return {"artifact": ref.to_dict(), "data": section}

    # ── Stage 7: Features ────────────────────────────────────────────
    def stage_features(
        self,
        section: ERTInvertedSection,
        section_hash: str,
        *,
        n_depth_bins: int = 10,
    ) -> dict[str, Any]:
        """Target-independent deterministic feature extraction."""
        rho = np.asarray(section.resistivity, dtype=float)
        z = np.asarray(section.z, dtype=float)
        x = np.asarray(section.x, dtype=float)

        depth_bin_edges = np.linspace(0.0, float(np.max(z)) + 1e-9, n_depth_bins + 1)
        depth_bin_means: list[float] = []
        for i in range(n_depth_bins):
            mask = (z >= depth_bin_edges[i]) & (z < depth_bin_edges[i + 1])
            depth_bin_means.append(float(np.mean(rho[mask])) if mask.sum() else float("nan"))

        # vertical gradient: d(rho)/dz along depth-sorted profile
        order = np.argsort(z)
        dz = np.diff(z[order])
        dlog = np.diff(np.log(rho[order] + 1e-9))
        with np.errstate(divide="ignore", invalid="ignore"):
            vert_grad = np.where(dz > 0, np.abs(dlog / np.maximum(dz, 1e-9)), np.nan)
        vertical_gradient_max = float(np.nanmax(vert_grad)) if len(vert_grad) else None

        # horizontal gradient: max |d log rho / dx| per depth band
        horiz_grads: list[float] = []
        for i in range(n_depth_bins):
            mask = (z >= depth_bin_edges[i]) & (z < depth_bin_edges[i + 1])
            xs = x[mask]
            rs = rho[mask]
            if len(xs) >= 2:
                o = np.argsort(xs)
                dx = np.diff(xs[o])
                dr = np.diff(np.log(rs[o] + 1e-9))
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = np.abs(np.where(dx > 0, dr / np.maximum(dx, 1e-9), np.nan))
                if len(g):
                    horiz_grads.append(float(np.nanmax(g)))
        horizontal_gradient_max = float(np.max(horiz_grads)) if horiz_grads else None

        # layer boundaries via max vertical gradient
        n_boundaries = (
            int(np.sum(vert_grad > np.nanpercentile(vert_grad, 90))) if len(vert_grad) else 0
        )

        sens = np.asarray(section.sensitivity, dtype=float) if section.sensitivity else None
        sens_w = None
        if sens is not None and np.nansum(sens) > 0:
            sens_w = float(np.average(rho, weights=sens))
        unc_arr = np.asarray(section.uncertainty, dtype=float) if section.uncertainty else None
        mean_unc = float(np.nanmean(unc_arr)) if unc_arr is not None and len(unc_arr) else None

        features = ERTFeatures(
            survey_id=section.survey_id,
            blinded_site_id=section.blinded_site_id,
            resistivity_mean_ohm_m=float(np.mean(rho)),
            resistivity_median_ohm_m=float(np.median(rho)),
            resistivity_std_ohm_m=float(np.std(rho)),
            resistivity_p10_ohm_m=float(np.percentile(rho, 10)),
            resistivity_p90_ohm_m=float(np.percentile(rho, 90)),
            vertical_gradient_max=vertical_gradient_max,
            horizontal_gradient_max=horizontal_gradient_max,
            n_layer_boundaries=n_boundaries,
            depth_bin_means=depth_bin_means,
            depth_bin_edges_m=list(depth_bin_edges),
            sensitivity_weighted_mean=sens_w,
            mean_model_uncertainty=mean_unc,
            support_fraction=float(np.mean(sens > 0)) if sens is not None else None,
            parent_artifact=section_hash,
            processing_provenance={
                "processing_step": "ert_feature_extraction",
                "processing_version": PROCESSING_VERSION,
            },
        )
        ref = make_artifact_ref(
            ProcessingStage.ERT_FEATURES,
            features.model_dump(mode="json"),
            parent_artifact=section_hash,
            processing_version=PROCESSING_VERSION,
        )
        return {"artifact": ref.to_dict(), "data": features}

    # ── Run the full chain ───────────────────────────────────────────
    def run(
        self,
        observation: ERTObservation,
        *,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> dict[str, Any]:
        """Execute the full deterministic ERT chain with provenance."""
        chain: dict[str, Any] = {}
        raw = self.stage_raw(observation)
        chain["raw"] = raw
        cal = self.stage_calibrated(observation, raw["raw_hash"], gain=gain, offset=offset)
        chain["calibrated"] = cal
        qc = self.stage_qc(cal["data"], cal["artifact"]["artifact_id"])
        chain["qc"] = qc
        app = self.stage_apparent_resistivity(qc["data"], qc["artifact"]["artifact_id"])
        chain["apparent"] = app
        inv_in = self.stage_inversion_input(app["data"], app["artifact"]["artifact_id"])
        chain["inversion_input"] = inv_in
        inv = self.stage_inverted(inv_in["data"], inv_in["artifact"]["artifact_id"])
        chain["inverted"] = inv
        feats = self.stage_features(inv["data"], inv["artifact"]["artifact_id"])
        chain["features"] = feats
        return chain
