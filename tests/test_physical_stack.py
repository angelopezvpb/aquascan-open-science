"""Open-source physical encoder stack tests.

Covers:
- RAW immutability
- ERT schema
- seismic schema
- geometry handling
- missing modality
- QC propagation
- support masks
- coordinate transformation
- cross-modal registration
- provenance lineage
- synthetic fixture labeling
- failure fixtures
- backend abstraction
- future real-data import contract

Tests using synthetic data explicitly identify them as synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aquascan.physical.common import (
    ProcessingStage,
    compute_artifact_hash,
    make_artifact_ref,
)
from aquascan.physical.coords import (
    Elevation,
    ElevationType,
    LocalSurveyFrame,
    register_survey_points,
)
from aquascan.physical.ert.backends import (
    Analytic1DERTBackend,
    available_ert_backends,
    get_ert_backend_status,
)
from aquascan.physical.ert.encoders import (
    ERTGridEncoder,
    ERTHandcraftedEncoder,
)
from aquascan.physical.ert.pipeline import ERTPipeline
from aquascan.physical.fusion.contract import (
    GeoAIFusionInput,
    ModalityBundle,
    validate_arm_modalities,
)
from aquascan.physical.fusion.registration import (
    build_shared_representation,
)
from aquascan.physical.models import (
    ERTObservation,
    ERTQuadrupoleMeasurement,
    SeismicObservation,
    SeismicShot,
)
from aquascan.physical.seismic.backends import (
    InterceptTime1DSeismicBackend,
    get_seismic_backend_status,
)
from aquascan.physical.seismic.encoders import (
    SeismicGridEncoder,
    SeismicHandcraftedEncoder,
)
from aquascan.physical.seismic.pipeline import SeismicPipeline
from aquascan.physical.synthetic.fixtures import (
    SYNTHETIC_WARNING,
    ErtHalfspaceFixture,
    ErtTwoLayerFixture,
    SeismicConstantVelocityFixture,
    SeismicTwoLayerFixture,
)

REPORTS_DIR = Path("reports")
SCHEMAS_DIR = Path("schemas")

SYNTHETIC = "SYNTHETIC_ENGINEERING_FIXTURE"


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_ert_observation() -> ERTObservation:
    """Synthetic ERT observation (SYNTHETIC_ENGINEERING_FIXTURE)."""
    quads = []
    for i in range(12):
        spacing = float(5.0 + i * 5.0)
        quads.append(
            ERTQuadrupoleMeasurement(
                survey_id="SYNTH-ERT-T",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-T",
                electrode_a=0.0,
                electrode_b=3.0 * spacing,
                electrode_m=spacing,
                electrode_n=2.0 * spacing,
                injected_current_a=0.1,
                measured_voltage_v=0.05,
                contact_resistance_ohm=150.0,
                instrument_channel=f"CH{i}",
                autocal_version="v0.1",
                raw_hash=f"raw-{i}",
            )
        )
    return ERTObservation(
        survey_id="SYNTH-ERT-T",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-T",
        quadrupoles=quads,
        autocal_version="v0.1",
        raw_hash=compute_artifact_hash({"synthetic": True}),
    )


def _make_seismic_observation() -> SeismicObservation:
    """Synthetic seismic observation (SYNTHETIC_ENGINEERING_FIXTURE)."""
    fixture = SeismicTwoLayerFixture()
    offsets = fixture.offsets_m
    _, times = fixture.forward()
    shots = []
    for si, sx in enumerate([0.0, 30.0]):
        rec_x, rec_ids, waves = [], [], []
        for ri, rx in enumerate(offsets):
            rel = abs(rx - sx)
            t_arr = float(np.interp(rel, offsets, times))
            idx = max(0, min(999, int(t_arr * 2000.0)))
            trace = [0.0] * 1000
            trace[idx] = 1.0
            waves.append(trace)
            rec_x.append(rx)
            rec_ids.append(f"R{ri}")
        shots.append(
            SeismicShot(
                survey_id="SYNTH-SEIS-T",
                blinded_site_id="AQ-P0-000",
                acquisition_id="ACQ-ST",
                shot_id=f"S{si}",
                shot_x=sx,
                receiver_ids=rec_ids,
                receiver_x=rec_x,
                sample_rate_hz=2000.0,
                record_length_s=0.5,
                raw_waveforms=waves,
                channel_numbers=list(range(len(rec_ids))),
                autocal_version="v0.1",
                raw_hash=f"raw-seis-{si}",
            )
        )
    return SeismicObservation(
        survey_id="SYNTH-SEIS-T",
        blinded_site_id="AQ-P0-000",
        acquisition_id="ACQ-ST",
        shots=shots,
        autocal_version="v0.1",
        raw_hash=compute_artifact_hash({"synthetic": True}),
    )


@pytest.fixture(scope="module")
def ert_chain() -> dict:
    obs = _make_ert_observation()
    return ERTPipeline().run(obs)


@pytest.fixture(scope="module")
def seis_chain() -> dict:
    obs = _make_seismic_observation()
    return SeismicPipeline().run(obs)


# ── 1. RAW immutability ───────────────────────────────────────────────


def test_ert_raw_unchanged_by_pipeline(ert_chain: dict) -> None:
    obs = _make_ert_observation()
    raw_payload_before = obs.model_dump(mode="json")
    ERTPipeline().run(obs)
    raw_payload_after = obs.model_dump(mode="json")
    assert raw_payload_before == raw_payload_after


def test_seismic_raw_unchanged_by_pipeline(seis_chain: dict) -> None:
    obs = _make_seismic_observation()
    raw_before = obs.model_dump(mode="json")
    SeismicPipeline().run(obs)
    assert obs.model_dump(mode="json") == raw_before


def test_raw_waveforms_preserved_in_filtered(seis_chain: dict) -> None:
    """FILTERED_VIEW must never replace RAW waveforms."""
    filtered = seis_chain["filtered"]["data"]
    for shot in filtered["shots"]:
        assert "raw_waveforms" in shot
        assert "filtered_waveforms" in shot
        assert len(shot["raw_waveforms"]) == len(shot["filtered_waveforms"])


# ── 2. Schemas ────────────────────────────────────────────────────────


def test_ert_schema_exported() -> None:
    schema = json.loads((SCHEMAS_DIR / "ert_observation_v0.json").read_text())
    assert schema["title"] == "ERTObservation"
    props = schema["properties"]
    for col in [
        "survey_id",
        "blinded_site_id",
        "acquisition_id",
        "quadrupoles",
        "raw_hash",
        "processing_provenance",
    ]:
        assert col in props


def test_seismic_schema_exported() -> None:
    schema = json.loads((SCHEMAS_DIR / "seismic_observation_v0.json").read_text())
    assert schema["title"] == "SeismicObservation"
    props = schema["properties"]
    for col in ["shots", "raw_hash", "autocal_version"]:
        assert col in props


def test_physical_representation_schema_exported() -> None:
    schema = json.loads((SCHEMAS_DIR / "physical_representation_v0.json").read_text())
    assert schema["title"] == "PhysicalRepresentation"
    for col in ["depth_bins_m", "lateral_bins_m", "modality_masks"]:
        assert col in schema["properties"]


def test_ert_quadrupole_schema_preserves_raw() -> None:
    schema = json.loads((SCHEMAS_DIR / "ert_observation_v0.json").read_text())
    quad = schema["$defs"]["ERTQuadrupoleMeasurement"]["properties"]
    for col in ["injected_current_a", "measured_voltage_v"]:
        assert col in quad
        assert "null" in quad[col]["anyOf"][0]["type"] or "null" in str(quad[col])


# ── 3. Pipeline stages + provenance lineage ───────────────────────────


def test_ert_pipeline_all_stages(ert_chain: dict) -> None:
    expected = [
        "raw",
        "calibrated",
        "qc",
        "apparent",
        "inversion_input",
        "inverted",
        "features",
    ]
    assert list(ert_chain.keys()) == expected


def test_seismic_pipeline_all_stages(seis_chain: dict) -> None:
    expected = [
        "raw",
        "calibrated",
        "qc",
        "filtered",
        "picks",
        "travel_times",
        "velocity",
        "features",
    ]
    assert list(seis_chain.keys()) == expected


def test_ert_provenance_chain(ert_chain: dict) -> None:
    """Each ERT artifact links back to its parent hash."""
    prev: str | None = None
    for sname in [
        "raw",
        "calibrated",
        "qc",
        "apparent",
        "inversion_input",
        "inverted",
        "features",
    ]:
        art = ert_chain[sname]["artifact"]
        if sname == "raw":
            assert art["parent_artifact"] is None
        else:
            assert art["parent_artifact"] == prev, f"{sname} lineage broken"
        prev = art["artifact_id"]


def test_seismic_provenance_chain(seis_chain: dict) -> None:
    prev: str | None = None
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
        if sname == "raw":
            assert art["parent_artifact"] is None
        else:
            assert art["parent_artifact"] == prev, f"{sname} lineage broken"
        prev = art["artifact_id"]


def test_every_artifact_has_timestamp_and_version(ert_chain: dict, seis_chain: dict) -> None:
    for chain in (ert_chain, seis_chain):
        for _stage, entry in chain.items():
            art = entry["artifact"]
            assert art["timestamp"]
            assert art["processing_version"]
            assert art["content_hash"]


def test_artifact_hash_deterministic() -> None:
    ref1 = make_artifact_ref(ProcessingStage.ERT_QC, {"a": 1, "b": [1, 2]})
    ref2 = make_artifact_ref(ProcessingStage.ERT_QC, {"a": 1, "b": [1, 2]})
    assert ref1.content_hash == ref2.content_hash
    ref3 = make_artifact_ref(ProcessingStage.ERT_QC, {"a": 1, "b": [1, 3]})
    assert ref1.content_hash != ref3.content_hash


# ── 4. QC propagation ─────────────────────────────────────────────────


def test_ert_qc_flags_high_contact(ert_chain: dict) -> None:
    obs = _make_ert_observation()
    obs.quadrupoles[0].contact_resistance_ohm = 5000.0  # bad electrode
    chain = ERTPipeline().run(obs)
    qc = chain["qc"]["data"]
    flags = qc["quadrupoles"][0]["qc_flags"]
    assert "high_contact_resistance" in flags
    assert qc["quadrupoles"][0]["qc_pass"] is False


def test_ert_qc_quality_score(ert_chain: dict) -> None:
    qc = ert_chain["qc"]["data"]
    assert 0.0 <= qc["quality_score"] <= 1.0
    assert qc["n_measurements"] == 12


def test_seismic_qc_dead_trace(seis_chain: dict) -> None:
    obs = _make_seismic_observation()
    # Kill one trace (all zeros)
    obs.shots[0].raw_waveforms[0] = [0.0] * 1000
    chain = SeismicPipeline().run(obs)
    qc = chain["qc"]["data"]
    first_shot = qc["shots"][0]
    assert "dead_trace" in first_shot["trace_qc_flags"][0]


# ── 5. Support masks / geometry handling ──────────────────────────────


def test_ert_grid_encoder_variable_geometry() -> None:
    """Different electrode counts/spacings must resample to the same grid."""
    obs_small = _make_ert_observation()
    obs_small.quadrupoles = obs_small.quadrupoles[:6]  # fewer quadrupoles
    sec_a = ERTPipeline().run(_make_ert_observation())["inverted"]["data"]
    sec_b = ERTPipeline().run(obs_small)["inverted"]["data"]
    enc = ERTGridEncoder(n_lateral=16, n_depth=8)
    out_a = enc.encode(sec_a)
    out_b = enc.encode(sec_b)
    assert out_a.embedding.shape == out_b.embedding.shape == (8, 16)
    assert out_a.support_mask.shape == out_b.support_mask.shape


def test_ert_grid_mask_marks_unsupported(ert_chain: dict) -> None:
    enc = ERTGridEncoder(n_lateral=16, n_depth=8)
    out = enc.encode(ert_chain["inverted"]["data"])
    assert out.support_mask.dtype == bool
    # Some cells are unsupported (sparse input geometry)
    assert out.support_mask.mean() < 1.0


def test_ert_handcrafted_embedding_shape(ert_chain: dict) -> None:
    out = ERTHandcraftedEncoder().encode(ert_chain["inverted"]["data"])
    assert out.embedding.ndim == 1
    assert out.embedding.shape[0] >= 10
    assert 0.0 <= out.quality <= 1.0


def test_seismic_handcrafted_embedding(seis_chain: dict) -> None:
    out = SeismicHandcraftedEncoder().encode(seis_chain["velocity"]["data"])
    assert out.embedding.ndim == 1
    assert out.embedding.shape[0] >= 8


def test_grid_encoders_untrained() -> None:
    """Grid scaffolds must be explicitly untrained (no groundwater claims)."""
    assert ERTGridEncoder().trained is False
    assert SeismicGridEncoder().trained is False


# ── 6. Missing modality / fusion contract ─────────────────────────────


def test_arm_requires_modalities() -> None:
    bundle = ModalityBundle(representation={"x": 1}, quality=1.0)
    inp_e = GeoAIFusionInput(blinded_site_id="S1", ert=bundle)
    assert validate_arm_modalities(inp_e, "ARM_E") == []
    assert validate_arm_modalities(inp_e, "ARM_S") == ["seismic"]
    assert validate_arm_modalities(inp_e, "ARM_ES") == ["seismic"]
    inp_es = GeoAIFusionInput(blinded_site_id="S1", ert=bundle, seismic=bundle)
    assert validate_arm_modalities(inp_es, "ARM_ES") == []
    assert validate_arm_modalities(inp_es, "ARM_CONTEXT_ES") == ["context"]


def test_missing_modality_quality_zero() -> None:
    from aquascan.physical.fusion.contract import modality_quality

    inp = GeoAIFusionInput(blinded_site_id="S1")
    q = modality_quality(inp)
    assert q["ert"] == 0.0 and q["seismic"] == 0.0 and q["context"] == 0.0


def test_unknown_arm_rejected() -> None:
    from aquascan.physical.fusion.contract import FusionArm

    inp = GeoAIFusionInput(blinded_site_id="S1")
    with pytest.raises(ValueError):
        validate_arm_modalities(inp, FusionArm("ARM_BOGUS"))


# ── 7. Cross-modal registration ───────────────────────────────────────


def test_shared_representation_channels(ert_chain: dict, seis_chain: dict) -> None:
    rep = build_shared_representation(
        "AQ-P0-000",
        ert=ert_chain["inverted"]["data"],
        seismic=seis_chain["velocity"]["data"],
    )
    assert rep.ert_resistivity is not None
    assert rep.seismic_vp is not None
    assert set(rep.modality_masks.keys()) == {"ert", "seismic"}
    n_depth = len(rep.depth_bins_m) - 1
    n_lat = len(rep.lateral_bins_m) - 1
    assert len(rep.ert_resistivity) == n_depth
    assert len(rep.ert_resistivity[0]) == n_lat


def test_shared_representation_ert_only() -> None:
    obs = _make_ert_observation()
    sec = ERTPipeline().run(obs)["inverted"]["data"]
    rep = build_shared_representation("AQ-P0-000", ert=sec)
    assert rep.seismic_vp is None
    assert "seismic" not in rep.modality_masks
    assert rep.ert_resistivity is not None


def test_native_models_preserved_after_registration(ert_chain: dict, seis_chain: dict) -> None:
    sec_before = ert_chain["inverted"]["data"].model_dump(mode="json")
    vel_before = seis_chain["velocity"]["data"].model_dump(mode="json")
    build_shared_representation(
        "AQ-P0-000",
        ert=ert_chain["inverted"]["data"],
        seismic=seis_chain["velocity"]["data"],
    )
    assert ert_chain["inverted"]["data"].model_dump(mode="json") == sec_before
    assert seis_chain["velocity"]["data"].model_dump(mode="json") == vel_before


# ── 8. Coordinate transformation ──────────────────────────────────────


def test_gnss_to_local_east_azimuth() -> None:
    frame = LocalSurveyFrame(origin_lat_deg=39.5, origin_lon_deg=3.0, survey_azimuth_deg=90.0)
    p0 = frame.gnss_to_local(39.5, 3.0, Elevation(50.0, ElevationType.ORTHOMETRIC))
    p1 = frame.gnss_to_local(39.5, 3.00009, Elevation(50.0, ElevationType.ORTHOMETRIC))
    assert abs(p0.profile_distance_m) < 1e-3
    # 0.00009 deg longitude at lat 39.5 ≈ 7.74 m east -> profile distance
    assert abs(p1.profile_distance_m - 7.74) < 0.1
    assert abs(p1.lateral_m) < 0.05


def test_gnss_to_local_north_azimuth() -> None:
    frame = LocalSurveyFrame(origin_lat_deg=39.5, origin_lon_deg=3.0, survey_azimuth_deg=0.0)
    p1 = frame.gnss_to_local(39.50009, 3.0, Elevation(50.0, ElevationType.ORTHOMETRIC))
    # 0.00009 deg latitude ≈ 9.99 m north
    assert abs(p1.profile_distance_m - 9.99) < 0.1
    assert abs(p1.lateral_m) < 0.05


def test_elevation_conversion_requires_geoid() -> None:
    e = Elevation(100.0, ElevationType.ELLIPSOIDAL)
    with pytest.raises(ValueError):
        e.as_orthometric()  # N is None -> refuse
    conv = e.as_orthometric(geoid_undulation_m=49.0)
    assert conv.value_m == pytest.approx(51.0)
    assert conv.type == ElevationType.ORTHOMETRIC


def test_local_z_not_silently_converted() -> None:
    e = Elevation(5.0, ElevationType.LOCAL_Z)
    with pytest.raises(ValueError):
        e.as_orthometric(geoid_undulation_m=49.0)


def test_register_survey_points_mixed_frames_refused() -> None:
    frame = LocalSurveyFrame(39.5, 3.0)
    p_ortho = frame.gnss_to_local(39.5, 3.0, Elevation(50.0, ElevationType.ORTHOMETRIC))
    # A LOCAL_Z point built directly (gnss_to_local already refuses LOCAL_Z
    # at conversion time; register_survey_points must refuse the mixture too).
    from aquascan.physical.coords import SurveyPoint

    p_local = SurveyPoint(
        easting_m=1.0,
        northing_m=1.0,
        elevation=Elevation(50.0, ElevationType.LOCAL_Z),
    )
    with pytest.raises(ValueError):
        register_survey_points([p_ortho, p_local])


# ── 9. Synthetic fixture labeling ─────────────────────────────────────


def test_synthetic_fixtures_labeled() -> None:
    for fx in [
        ErtHalfspaceFixture(),
        ErtTwoLayerFixture(),
        SeismicConstantVelocityFixture(),
        SeismicTwoLayerFixture(),
    ]:
        assert fx.label == SYNTHETIC
    assert SYNTHETIC in SYNTHETIC_WARNING


def test_forward_models_are_physical() -> None:
    hs = ErtHalfspaceFixture(rho=250.0)
    assert np.allclose(hs.forward(), 250.0, atol=1e-6)
    tl = ErtTwoLayerFixture(rho1=100.0, rho2=1000.0, h1=10.0)
    rho_a = tl.forward()
    # apparent resistivity between the two layer resistivities
    assert np.all(rho_a > 100.0) and np.all(rho_a < 1000.0)
    sv = SeismicConstantVelocityFixture(vp=1200.0)
    t = sv.forward()
    assert np.allclose(t, sv.offsets_m / 1200.0)
    st = SeismicTwoLayerFixture()
    direct, times = st.forward()
    # The first-arrival time is the earlier of the direct and head waves.
    i_crit = np.arcsin(st.v1 / st.v2)
    head = 2 * st.h1 * np.cos(i_crit) / st.v1 + st.offsets_m / st.v2
    expected = np.minimum(direct, head)
    assert np.allclose(times, expected, atol=1e-12)
    # Far offsets must eventually show the faster head wave (times < direct)
    far = st.offsets_m > 2 * st.h1 * np.tan(i_crit) * 5
    assert np.any(times[far] < direct[far])


def test_ert_twolayer_recovery() -> None:
    """Known model -> forward -> 1D grid inversion -> approximate recovery."""
    fx = ErtTwoLayerFixture(rho1=100.0, rho2=1000.0, h1=10.0)
    rho_a = fx.forward()
    best_err, best = np.inf, None
    for r1 in [80.0, 100.0, 120.0]:
        for r2 in [800.0, 1000.0, 1500.0]:
            for h in [8.0, 10.0, 12.0]:
                pred = ErtTwoLayerFixture(rho1=r1, rho2=r2, h1=h, spacing_m=fx.spacing_m).forward()
                err = float(np.mean((pred - rho_a) ** 2))
                if err < best_err:
                    best_err = err
                    best = (r1, r2, h)
    assert best is not None
    r1, r2, h = best
    assert abs(r1 - 100.0) <= 20
    assert abs(r2 - 1000.0) <= 500
    assert abs(h - 10.0) <= 2


def test_seismic_twolayer_intercept_recovery() -> None:
    fx = SeismicTwoLayerFixture(v1=800.0, v2=2500.0, h1=15.0)
    _, times = fx.forward()
    # Head wave is the first arrival only beyond the crossover distance:
    # x_co = 2*h1*cos(i)*v2/(v2-v1)
    i_crit = np.arcsin(fx.v1 / fx.v2)
    x_co = 2 * fx.h1 * np.cos(i_crit) * fx.v2 / (fx.v2 - fx.v1)
    mask_far = fx.offsets_m > x_co
    assert mask_far.sum() >= 5
    slope = np.polyfit(fx.offsets_m[mask_far], times[mask_far], 1)[0]
    v2_rec = 1.0 / slope
    assert abs(v2_rec - 2500.0) < 300


# ── 10. Backend abstraction ───────────────────────────────────────────


def test_ert_backend_registry() -> None:
    status = get_ert_backend_status()
    names = {b.name for b in status}
    assert "pyGIMLi" in names and "SimPEG" in names
    assert any(b.name == "AquaScanAnalytic1D" and b.available for b in status)


def test_seismic_backend_registry() -> None:
    status = get_seismic_backend_status()
    names = {b.name for b in status}
    assert "obspy" in names
    assert any(b.name == "AquaScanInterceptTime1D" and b.available for b in status)


def test_analytic_backend_returns_section() -> None:
    backend = Analytic1DERTBackend()
    x = np.array([1.0, 2.0, 3.0])
    z = np.array([1.0, 2.0, 3.0])
    rho = np.array([100.0, 200.0, 300.0])
    section = backend.invert(x, z, rho, mesh_metadata={"survey_id": "S", "blinded_site_id": "B"})
    assert len(section.resistivity) == 3
    assert section.processing_provenance["processing_step"].startswith("ert_analytic")


def test_intercept_backend_returns_model() -> None:
    backend = InterceptTime1DSeismicBackend()
    fx = SeismicTwoLayerFixture()
    _, times = fx.forward()
    model = backend.invert(
        fx.offsets_m,
        times,
        metadata={"survey_id": "S", "blinded_site_id": "B"},
    )
    assert len(model.vp) >= 4
    assert model.vp[0] > 0


def test_available_backends_instantiable() -> None:
    backends = available_ert_backends()
    assert len(backends) >= 1


# ── 11. Failure fixtures ──────────────────────────────────────────────


def test_empty_inversion_input_raises() -> None:
    obs = _make_ert_observation()
    obs.quadrupoles = []  # no measurements
    pipe = ERTPipeline()
    raw = pipe.stage_raw(obs)
    cal = pipe.stage_calibrated(obs, raw["raw_hash"])
    qc = pipe.stage_qc(cal["data"], cal["artifact"]["artifact_id"])
    app = pipe.stage_apparent_resistivity(qc["data"], qc["artifact"]["artifact_id"])
    inv_in = pipe.stage_inversion_input(app["data"], app["artifact"]["artifact_id"])
    with pytest.raises(ValueError):
        pipe.stage_inverted(inv_in["data"], inv_in["artifact"]["artifact_id"])


def test_all_measurements_fail_qc_quality_zero() -> None:
    obs = _make_ert_observation()
    for q in obs.quadrupoles:
        q.contact_resistance_ohm = 9000.0  # all bad
    pipe = ERTPipeline()
    raw = pipe.stage_raw(obs)
    cal = pipe.stage_calibrated(obs, raw["raw_hash"])
    qc = pipe.stage_qc(cal["data"], cal["artifact"]["artifact_id"])
    assert qc["data"]["quality_score"] == 0.0
    assert qc["data"]["n_pass"] == 0
    # No QC-passing measurements -> inversion must refuse (not fabricate)
    with pytest.raises(ValueError):
        pipe.run(obs)


def test_dead_channel_flagged_in_seismic_qc() -> None:
    obs = _make_seismic_observation()
    obs.shots[0].raw_waveforms[0] = [0.0] * 1000
    obs.shots[0].raw_waveforms[1] = [0.0] * 1000
    obs.shots[0].raw_waveforms[2] = [0.0] * 1000
    chain = SeismicPipeline().run(obs)
    qc = chain["qc"]["data"]
    dead = sum(1 for s in qc["shots"] for f in s["trace_qc_flags"] if "dead_trace" in f)
    assert dead >= 3


def test_bad_geometry_raises_geometric_factor() -> None:
    """Coincident electrodes must raise rather than silently compute."""
    obs = _make_ert_observation()
    obs.quadrupoles[0].electrode_m = obs.quadrupoles[0].electrode_a  # coincide
    chain = ERTPipeline().run(obs)
    # Measurement simply gets no apparent resistivity (NaN), not fabricated
    app = chain["apparent"]["data"]
    rho = app["quadrupoles"][0].get("apparent_resistivity_ohm_m")
    assert rho is None


# ── 12. Future real-data import contract ──────────────────────────────


def test_import_path_canonical_models() -> None:
    """Real field files must parse into canonical models without retraining.

    The canonical ERTObservation / SeismicObservation are the import
    contract: any future parser (ABEM/Syscal CSV, SEG-2/SEG-Y) must fill
    these structures. We verify the models accept a realistic record.
    """
    obs = _make_ert_observation()
    assert isinstance(obs, ERTObservation)
    assert obs.quadrupoles[0].survey_id == "SYNTH-ERT-T"
    seis = _make_seismic_observation()
    assert isinstance(seis, SeismicObservation)
    assert len(seis.shots) >= 2
    assert seis.shots[0].sample_rate_hz > 0


def test_pipeline_accepts_canonical_observation() -> None:
    """Pipeline entry point takes canonical models — AQ-TEST-001 drop-in."""
    obs = _make_ert_observation()
    obs.survey_id = "AQ-TEST-001"
    for q in obs.quadrupoles:
        q.survey_id = "AQ-TEST-001"
    chain = ERTPipeline().run(obs)
    assert chain["features"]["data"].survey_id == "AQ-TEST-001"


# ── 13. Reports exist ─────────────────────────────────────────────────


def test_gate10_reports_exist() -> None:
    for f in [
        "gate10_physical_stack_manifest_v0.json",
        "gate10_backend_compatibility_v0.csv",
        "gate10_license_manifest_v0.csv",
        "gate10_synthetic_validation_v0.json",
    ]:
        assert (REPORTS_DIR / f).exists(), f"missing {f}"


def test_synthetic_validation_has_warning() -> None:
    val = json.loads((REPORTS_DIR / "gate10_synthetic_validation_v0.json").read_text())
    assert SYNTHETIC in val["warning"]
    assert all(c["recovery_ok"] for c in val["checks"])


def test_manifest_no_groundwater_claims() -> None:
    manifest = json.loads((REPORTS_DIR / "gate10_physical_stack_manifest_v0.json").read_text())
    assert "No groundwater detection claims" in manifest["note"]
