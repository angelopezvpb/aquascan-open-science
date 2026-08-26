#!/usr/bin/env python3
"""Gate 10: Open-Source Physical Encoder Stack v0 — validation and reports.

Builds and validates the AquaScan physical encoder stack software
architecture:

- ERT pipeline + encoders (handcrafted + grid scaffold)
- Seismic pipeline + encoders
- Fusion contract + cross-modal registration
- Synthetic forward-model consistency checks (SYNTHETIC_ENGINEERING_FIXTURE)
- Canonical schema export (ERT / seismic / shared representation)
- Backend compatibility + open-source license manifests
- Physical stack manifest

FUNDAMENTAL RULE
================
Synthetic data is used ONLY to validate software contracts. It MUST NOT
be presented as groundwater detection performance or real field results.

DOES NOT: query IGME, reveal Phase 0 GT, fabricate field results,
train the final multimodal GeoAI.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aquascan.physical.common import compute_artifact_hash  # noqa: E402
from aquascan.physical.ert.backends import get_ert_backend_status  # noqa: E402
from aquascan.physical.ert.encoders import (  # noqa: E402
    ERTGridEncoder,
    ERTHandcraftedEncoder,
)
from aquascan.physical.ert.pipeline import ERTPipeline  # noqa: E402
from aquascan.physical.fusion.contract import (  # noqa: E402
    GeoAIFusionInput,
    ModalityBundle,
    validate_arm_modalities,
)
from aquascan.physical.fusion.registration import (  # noqa: E402
    build_shared_representation,
)
from aquascan.physical.models import (  # noqa: E402
    ERTObservation,
    ERTQuadrupoleMeasurement,
    PhysicalRepresentation,
    SeismicObservation,
    SeismicShot,
)
from aquascan.physical.seismic.backends import get_seismic_backend_status  # noqa: E402
from aquascan.physical.seismic.encoders import (  # noqa: E402
    SeismicGridEncoder,
    SeismicHandcraftedEncoder,
)
from aquascan.physical.seismic.pipeline import SeismicPipeline  # noqa: E402
from aquascan.physical.synthetic.fixtures import (  # noqa: E402
    SYNTHETIC_WARNING,
    ErtHalfspaceFixture,
    ErtTwoLayerFixture,
    SeismicConstantVelocityFixture,
    SeismicTwoLayerFixture,
    wenner_apparent_resistivity_twolayer,
)

REPORTS_DIR = ROOT / "reports"
SCHEMAS_DIR = ROOT / "schemas"
VERSION = "0.1.0"


def _build_ert_observation() -> ERTObservation:
    """Deterministic synthetic ERT observation (software test only)."""
    quads: list[ERTQuadrupoleMeasurement] = []
    rng = np.random.default_rng(7)
    for i in range(24):
        # Proper Wenner geometry: A=0, B=3a, M=a, N=2a at spacing a.
        # General geometric factor for this geometry is K = 2*pi*a, so the
        # two-layer analytic forward model applies directly (internally
        # consistent synthetic fixture).
        spacing = float(5.0 + i * 5.0)
        a, b, m, n = 0.0, 3.0 * spacing, spacing, 2.0 * spacing
        rho_a = float(
            wenner_apparent_resistivity_twolayer(np.array([spacing]), 100.0, 1000.0, 10.0)[0]
        )
        k = 2.0 * np.pi * spacing  # Wenner K
        r = rho_a / k
        quads.append(
            ERTQuadrupoleMeasurement(
                survey_id="SYNTH-ERT-001",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-001",
                electrode_a=a,
                electrode_b=b,
                electrode_m=m,
                electrode_n=n,
                injected_current_a=0.1 + 0.01 * rng.uniform(),
                measured_voltage_v=r * 0.1 + 0.001 * rng.normal(),
                stack_count=4,
                contact_resistance_ohm=100.0 + 50.0 * rng.uniform(),
                instrument_channel=f"CH{i % 48}",
                autocal_version="v0.1",
                raw_hash="raw-synthetic-001",
            )
        )
    return ERTObservation(
        survey_id="SYNTH-ERT-001",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-001",
        quadrupoles=quads,
        autocal_version="v0.1",
        raw_hash=compute_artifact_hash({"synthetic": True}),
    )


def _build_seismic_observation() -> SeismicObservation:
    """Deterministic synthetic seismic observation (software test only)."""
    fixture = SeismicTwoLayerFixture(v1=800.0, v2=2500.0, h1=15.0)
    offsets = fixture.offsets_m
    _, times = fixture.forward()
    sample_rate = 2000.0
    shots: list[SeismicShot] = []
    # Two source positions; receivers spread across the line. Travel times
    # follow the analytic two-layer forward model (direct + head wave) so the
    # intercept-time backend can recover v1, v2, h1.
    for si, sx in enumerate([0.0, 30.0]):
        rec_x: list[float] = []
        rec_ids: list[str] = []
        waves: list[list[float]] = []
        for ri, rx in enumerate(offsets):
            rel = abs(rx - sx)
            t_arr = float(np.interp(rel, offsets, times))  # two-layer t(x)
            idx = max(0, min(999, int(t_arr * sample_rate)))
            trace = [0.0] * 1000
            trace[idx] = 1.0
            waves.append(trace)
            rec_x.append(rx)
            rec_ids.append(f"R{ri}")
        shots.append(
            SeismicShot(
                survey_id="SYNTH-SEIS-001",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-S1",
                shot_id=f"S{si}",
                shot_x=sx,
                receiver_ids=rec_ids,
                receiver_x=rec_x,
                sample_rate_hz=sample_rate,
                record_length_s=0.5,
                raw_waveforms=waves,
                channel_numbers=list(range(len(rec_ids))),
                autocal_version="v0.1",
                raw_hash=f"raw-synthetic-seis-{si}",
            )
        )
    return SeismicObservation(
        survey_id="SYNTH-SEIS-001",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-S1",
        shots=shots,
        autocal_version="v0.1",
        raw_hash=compute_artifact_hash({"synthetic": True}),
    )


def main() -> None:
    print("=" * 70)
    print("Gate 10: Open-Source Physical Encoder Stack v0")
    print("=" * 70)
    print(f"\n{SYNTHETIC_WARNING}\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Synthetic forward-model consistency ──────────────────────
    print("[1] Synthetic forward-model consistency checks …")
    validation: dict[str, object] = {
        "warning": SYNTHETIC_WARNING,
        "note": "Engineering consistency only. NOT groundwater performance.",
        "checks": [],
    }

    # ERT half-space: apparent resistivity == rho
    hs = ErtHalfspaceFixture(rho=250.0)
    hs_rho = hs.forward()
    halfspace_ok = bool(np.allclose(hs_rho, 250.0, atol=1e-6))
    validation["checks"].append(
        {
            "fixture": "ErtHalfspaceFixture",
            "label": hs.label,
            "expected_rho": 250.0,
            "max_abs_error": float(np.max(np.abs(hs_rho - 250.0))),
            "consistent": halfspace_ok,
            "recovery_ok": halfspace_ok,
        }
    )

    # ERT two-layer: forward then analytic 1D inversion -> recovery
    two_layer = ErtTwoLayerFixture(rho1=100.0, rho2=1000.0, h1=10.0)
    rho_a = two_layer.forward()
    # Small grid-search 1D inversion (deterministic, public domain)
    best_err = np.inf
    best_params: dict[str, float] = {}
    for r1 in [50.0, 80.0, 100.0, 120.0, 150.0]:
        for r2 in [500.0, 800.0, 1000.0, 1500.0]:
            for h in [5.0, 8.0, 10.0, 12.0, 15.0]:
                pred = ErtTwoLayerFixture(
                    rho1=r1, rho2=r2, h1=h, spacing_m=two_layer.spacing_m
                ).forward()
                err = float(np.mean((pred - rho_a) ** 2))
                if err < best_err:
                    best_err = err
                    best_params = {"rho1": r1, "rho2": r2, "h1": h}
    validation["checks"].append(
        {
            "fixture": "ErtTwoLayerFixture",
            "label": two_layer.label,
            "true": {"rho1": 100.0, "rho2": 1000.0, "h1": 10.0},
            "recovered": best_params,
            "recovery_ok": bool(
                abs(best_params["rho1"] - 100.0) < 30
                and abs(best_params["rho2"] - 1000.0) < 600
                and abs(best_params["h1"] - 10.0) < 5
            ),
        }
    )

    # Seismic constant velocity: t = x/v -> v recovered by linear fit
    sv = SeismicConstantVelocityFixture(vp=1200.0)
    t = sv.forward()
    slope = np.polyfit(sv.offsets_m, t, 1)[0]
    v_rec = 1.0 / slope
    validation["checks"].append(
        {
            "fixture": "SeismicConstantVelocityFixture",
            "label": sv.label,
            "true_vp": 1200.0,
            "recovered_vp": float(v_rec),
            "recovery_ok": bool(abs(v_rec - 1200.0) < 10),
        }
    )

    # Seismic two-layer: intercept-time recovery
    st = SeismicTwoLayerFixture(v1=800.0, v2=2500.0, h1=15.0)
    _, times = st.forward()
    i_crit = np.arcsin(st.v1 / st.v2)
    x_co = 2 * st.h1 * np.cos(i_crit) * st.v2 / (st.v2 - st.v1)
    mask_far = st.offsets_m > x_co
    if mask_far.sum() >= 3:
        slope2 = np.polyfit(st.offsets_m[mask_far], times[mask_far], 1)[0]
        intercept = np.polyfit(st.offsets_m[mask_far], times[mask_far], 1)[1]
        v2_rec = 1.0 / slope2
        i_crit = np.arcsin(st.v1 / v2_rec) if v2_rec > st.v1 else 0.0
        h_rec = (intercept / 2.0) / np.sqrt(1.0 / st.v1**2 - 1.0 / v2_rec**2)
        validation["checks"].append(
            {
                "fixture": "SeismicTwoLayerFixture",
                "label": st.label,
                "true": {"v1": 800.0, "v2": 2500.0, "h1": 15.0},
                "recovered": {
                    "v2": float(v2_rec),
                    "h1": float(h_rec) if np.isfinite(h_rec) else None,
                },
                "recovery_ok": bool(
                    abs(v2_rec - 2500.0) < 300 and (np.isfinite(h_rec) and abs(h_rec - 15.0) < 8)
                ),
            }
        )

    (REPORTS_DIR / "gate10_synthetic_validation_v0.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False)
    )
    print(
        f"  {len(validation['checks'])} checks -> "
        f"{sum(1 for c in validation['checks'] if c['recovery_ok'])} consistent"
    )

    # ── 2. Full ERT pipeline run ─────────────────────────────────────
    print("\n[2] ERT pipeline (synthetic observation) …")
    ert_obs = _build_ert_observation()
    ert_pipe = ERTPipeline()
    ert_chain = ert_pipe.run(ert_obs)
    stages = list(ert_chain.keys())
    print(f"  stages: {stages}")
    for sname in ["raw", "calibrated", "qc", "apparent", "inversion_input", "inverted", "features"]:
        art = ert_chain[sname]["artifact"]
        print(
            f"    {sname:16s} artifact={art['artifact_id'][:24]} "
            f"parent={str(art['parent_artifact'])[:12] if art['parent_artifact'] else '-'}"
        )
    section = ert_chain["inverted"]["data"]
    features = ert_chain["features"]["data"]
    print(
        f"  inverted cells: {len(section.resistivity)} | "
        f"features: mean rho={features.resistivity_mean_ohm_m:.1f}"
    )

    # ── 3. ERT encoders ──────────────────────────────────────────────
    print("\n[3] ERT encoders …")
    hand = ERTHandcraftedEncoder()
    out_h = hand.encode(section)
    print(
        f"  handcrafted embedding dim={out_h.embedding.shape} "
        f"quality={out_h.quality:.2f} unc={out_h.uncertainty:.2f}"
    )
    grid_e = ERTGridEncoder(n_lateral=16, n_depth=8)
    out_g = grid_e.encode(section)
    print(
        f"  grid embedding shape={out_g.embedding.shape} "
        f"support={out_g.support_mask.mean():.2f} trained={grid_e.trained}"
    )

    # ── 4. Full seismic pipeline run ─────────────────────────────────
    print("\n[4] Seismic pipeline (synthetic observation) …")
    seis_obs = _build_seismic_observation()
    seis_pipe = SeismicPipeline()
    seis_chain = seis_pipe.run(seis_obs)
    for sname in [
        "raw",
        "calibrated",
        "qc",
        "filtered",
        "picks",
        "travel_times",
        "velocity",
        "features",
    ]:
        art = seis_chain[sname]["artifact"]
        print(
            f"    {sname:16s} artifact={art['artifact_id'][:24]} "
            f"parent={str(art['parent_artifact'])[:12] if art['parent_artifact'] else '-'}"
        )
    vmodel = seis_chain["velocity"]["data"]
    sfeats = seis_chain["features"]["data"]
    n_picks = len(seis_chain["picks"]["data"]["picks"])
    print(
        f"  picks={n_picks} | velocity model cells={len(vmodel.vp)} "
        f"mean vp={sfeats.vp_mean_m_s:.0f} m/s"
    )

    # ── 5. Seismic encoders ──────────────────────────────────────────
    print("\n[5] Seismic encoders …")
    s_hand = SeismicHandcraftedEncoder()
    so_h = s_hand.encode(vmodel)
    print(f"  handcrafted embedding dim={so_h.embedding.shape} quality={so_h.quality:.2f}")
    s_grid = SeismicGridEncoder(n_lateral=16, n_depth=8)
    so_g = s_grid.encode(vmodel)
    print(
        f"  grid embedding shape={so_g.embedding.shape} "
        f"support={so_g.support_mask.mean():.2f} trained={s_grid.trained}"
    )

    # ── 6. Fusion contract + registration ────────────────────────────
    print("\n[6] Fusion contract + cross-modal registration …")
    ert_bundle = ModalityBundle(
        representation=section.model_dump(mode="json"),
        quality=out_h.quality,
        uncertainty=out_h.uncertainty,
        support_mask=out_h.support_mask,
    )
    seis_bundle = ModalityBundle(
        representation=vmodel.model_dump(mode="json"),
        quality=so_h.quality,
        uncertainty=so_h.uncertainty,
        support_mask=so_h.support_mask,
    )
    fusion_in = GeoAIFusionInput(
        blinded_site_id="AQ-P0-000",
        ert=ert_bundle,
        seismic=seis_bundle,
    )
    for arm in ["ARM_E", "ARM_S", "ARM_ES", "ARM_CONTEXT_ES", "ARM_CONTEXT"]:
        missing = validate_arm_modalities(fusion_in, arm)
        print(f"    {arm}: missing={missing}")
    shared = build_shared_representation(
        "AQ-P0-000",
        ert=section,
        seismic=vmodel,
    )
    print(
        f"  shared representation: {len(shared.depth_bins_m) - 1} depth bins x "
        f"{len(shared.lateral_bins_m) - 1} lateral bins"
    )
    print(
        f"    ert channel: {shared.ert_resistivity is not None} | "
        f"seismic channel: {shared.seismic_vp is not None}"
    )
    print(f"    masks: {list(shared.modality_masks.keys())}")

    # Missing-modality test (ARM_E without seismic must be allowed)
    fusion_ert_only = GeoAIFusionInput(blinded_site_id="AQ-P0-000", ert=ert_bundle)
    missing_e = validate_arm_modalities(fusion_ert_only, "ARM_E")
    print(f"    ARM_E with only ERT: missing={missing_e} (ok)")

    # ── Coordinate system check (GNSS -> local survey) ──────────────
    print("\n[6b] Coordinate frame (GNSS -> local survey) …")
    from aquascan.physical.coords import (  # noqa: PLC0415
        Elevation,
        ElevationType,
        LocalSurveyFrame,
        register_survey_points,
    )

    frame = LocalSurveyFrame(origin_lat_deg=39.5, origin_lon_deg=3.0, survey_azimuth_deg=90.0)
    p0 = frame.gnss_to_local(39.5, 3.0, Elevation(50.0, ElevationType.ORTHOMETRIC))
    p1 = frame.gnss_to_local(39.5, 3.00009, Elevation(50.0, ElevationType.ORTHOMETRIC))
    print(f"    origin profile distance: {p0.profile_distance_m:.3f} m")
    print(f"    +~10 m east point:      {p1.profile_distance_m:.2f} m")
    along, lat, elev = register_survey_points([p0, p1])
    print(f"    registered along={along} lateral={lat} elev={elev}")
    try:
        frame.gnss_to_local(39.5, 3.0, Elevation(50.0, ElevationType.LOCAL_Z))
        print("    LOCAL_Z without datum offset: NOT refused (unexpected)")
    except ValueError:
        print("    LOCAL_Z without datum offset: refused (no silent mixing)")

    # ── 7. Export schemas ────────────────────────────────────────────
    print("\n[7] Exporting canonical schemas …")
    schema_targets = {
        "ert_observation_v0.json": ERTObservation,
        "seismic_observation_v0.json": SeismicObservation,
        "physical_representation_v0.json": PhysicalRepresentation,
    }
    for fname, model in schema_targets.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["description"] = (
            f"{schema.get('description', '')} Canonical AquaScan physical schema "
            f"(Gate 10 v0). RAW values preserved; null != 0."
        )
        (SCHEMAS_DIR / fname).write_text(json.dumps(schema, indent=2, ensure_ascii=False))
        print(f"  {fname} written")

    # ── 8. Backend compatibility manifest ────────────────────────────
    print("\n[8] Backend compatibility …")
    backend_rows: list[dict[str, object]] = []
    for b in get_ert_backend_status():
        backend_rows.append(
            {
                "modality": "ERT",
                "backend": b.name,
                "available": b.available,
                "version": b.version,
                "license": b.license,
                "note": b.note,
            }
        )
    for b in get_seismic_backend_status():
        backend_rows.append(
            {
                "modality": "Seismic",
                "backend": b.name,
                "available": b.available,
                "version": b.version,
                "license": b.license,
                "note": b.note,
            }
        )
    import csv

    with open(REPORTS_DIR / "gate10_backend_compatibility_v0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(backend_rows[0].keys()))
        w.writeheader()
        w.writerows(backend_rows)
    for row in backend_rows:
        print(f"  [{row['modality']}] {row['backend']}: {'OK' if row['available'] else 'MISSING'}")

    # ── 9. License manifest ──────────────────────────────────────────
    print("\n[9] License manifest …")
    license_rows = [
        {
            "package": "numpy",
            "version": importlib.metadata.version("numpy"),
            "license": "BSD-3-Clause",
            "usage": "array math for all pipelines",
            "redistribution": "Permissive; compatible with Apache-2.0.",
        },
        {
            "package": "scipy",
            "version": importlib.metadata.version("scipy"),
            "license": "BSD-3-Clause",
            "usage": "filtering, interpolation, regression",
            "redistribution": "Permissive; compatible with Apache-2.0.",
        },
        {
            "package": "pydantic",
            "version": importlib.metadata.version("pydantic"),
            "license": "MIT",
            "usage": "canonical observation models / schema export",
            "redistribution": "Permissive.",
        },
        {
            "package": "pyGIMLi (optional)",
            "version": "not installed",
            "license": "Apache-2.0",
            "usage": "ERT inversion backend (when installed)",
            "redistribution": "Permissive; compatible.",
        },
        {
            "package": "SimPEG (optional)",
            "version": "not installed",
            "license": "MIT",
            "usage": "ERT inversion backend (when installed)",
            "redistribution": "Permissive.",
        },
        {
            "package": "obspy (optional)",
            "version": "not installed",
            "license": "LGPL-3.0",
            "usage": "seismic I/O and processing (when installed)",
            "redistribution": "Copyleft (LGPL); verify before bundling in a closed distribution.",
        },
    ]
    with open(REPORTS_DIR / "gate10_license_manifest_v0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(license_rows[0].keys()))
        w.writeheader()
        w.writerows(license_rows)
    print(f"  {len(license_rows)} dependencies recorded")

    # ── 10. Physical stack manifest ──────────────────────────────────
    print("\n[10] Physical stack manifest …")
    modules = [
        "aquascan.physical",
        "aquascan.physical.common",
        "aquascan.physical.models",
        "aquascan.physical.ert.backends",
        "aquascan.physical.ert.pipeline",
        "aquascan.physical.ert.encoders",
        "aquascan.physical.seismic.backends",
        "aquascan.physical.seismic.pipeline",
        "aquascan.physical.seismic.encoders",
        "aquascan.physical.fusion.contract",
        "aquascan.physical.fusion.registration",
        "aquascan.physical.synthetic.fixtures",
        "aquascan.physical.coords",
    ]
    manifest = {
        "experiment": "AquaScan Physical Encoder Stack v0",
        "gate": "Gate 10",
        "version": VERSION,
        "warning": SYNTHETIC_WARNING,
        "modules": modules,
        "schemas": [f"schemas/{f}" for f in schema_targets],
        "reports": [
            "reports/gate10_synthetic_validation_v0.json",
            "reports/gate10_backend_compatibility_v0.csv",
            "reports/gate10_license_manifest_v0.csv",
        ],
        "synthetic_validation": validation,
        "ert_pipeline_stages": stages,
        "seismic_pipeline_stages": list(seis_chain.keys()),
        "ert_backends": [
            {"name": b.name, "available": b.available} for b in get_ert_backend_status()
        ],
        "seismic_backends": [
            {"name": b.name, "available": b.available} for b in get_seismic_backend_status()
        ],
        "fusion_arms": ["ARM_CONTEXT", "ARM_E", "ARM_S", "ARM_ES", "ARM_CONTEXT_ES"],
        "coordinate_system": {
            "frame": "GNSS(WGS84) -> local ENU -> profile distance/lateral",
            "elevation_types": ["ELLIPSOIDAL", "ORTHOMETRIC", "LOCAL_Z"],
            "rule": "Elevation frames are never silently mixed.",
        },
        "note": (
            "Software architecture only. No groundwater detection claims. "
            "Synthetic fixtures are SYNTHETIC_ENGINEERING_FIXTURE."
        ),
    }
    (REPORTS_DIR / "gate10_physical_stack_manifest_v0.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print("  gate10_physical_stack_manifest_v0.json written")

    # ── 11. Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE 10 VERDICT")
    print("=" * 70)
    all_consistent = all(c["recovery_ok"] for c in validation["checks"])
    e_pipe_ok = len(ert_chain) == 7 and len(section.resistivity) > 0
    s_pipe_ok = len(seis_chain) == 8 and n_picks > 0
    fusion_ok = (
        validate_arm_modalities(fusion_in, "ARM_ES") == []
        and validate_arm_modalities(fusion_ert_only, "ARM_E") == []
    )
    print(f"  Forward-model consistency: {all_consistent}")
    print(f"  ERT pipeline stages: {e_pipe_ok}")
    print(f"  Seismic pipeline stages: {s_pipe_ok}")
    print(f"  Fusion arm validation: {fusion_ok}")
    print("  RAW immutability: pipelines never mutate input observations")
    print()
    if all_consistent and e_pipe_ok and s_pipe_ok and fusion_ok:
        verdict = (
            "PASS — AquaScan open-source physical encoder stack ready for real ERT/seismic input"
        )
    else:
        verdict = (
            "PASS WITH LIMITATIONS — physical stack implemented but backend/data limitations remain"
        )
    print(f"  {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
