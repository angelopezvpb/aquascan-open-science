"""Canonical data models for the physical encoder stack.

Every model preserves RAW measurements and carries full provenance.
null != 0 — missing values must never be silently converted to zeros.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


# ── ERT models ────────────────────────────────────────────────────────


class ERTQuadrupoleMeasurement(BaseModel):
    """One ERT quadrupole measurement (electrodes A B M N)."""

    survey_id: str
    blinded_site_id: str
    acquisition_id: str
    timestamp: datetime = Field(default_factory=_now)

    electrode_a: float = Field(description="Electrode A position along profile (m)")
    electrode_b: float = Field(description="Electrode B position along profile (m)")
    electrode_m: float = Field(description="Electrode M position along profile (m)")
    electrode_n: float = Field(description="Electrode N position along profile (m)")

    electrode_coordinates: list[dict[str, float]] = Field(
        default_factory=list,
        description="Optional electrode coordinates [{x_m, y_m, z_m}, ...] A,B,M,N",
    )
    electrode_elevations: list[float] | None = Field(
        default=None,
        description="Optional electrode elevations (m), same order A,B,M,N",
    )

    injected_current_a: float | None = Field(
        default=None, description="Injected current (A). Never overwrite RAW."
    )
    measured_voltage_v: float | None = Field(
        default=None, description="Measured voltage (V). Never overwrite RAW."
    )

    apparent_resistance_ohm: float | None = Field(
        default=None, description="Apparent resistance R = V/I (ohms)."
    )
    geometric_factor: float | None = Field(default=None, description="Geometric factor K (m).")
    apparent_resistivity_ohm_m: float | None = Field(
        default=None, description="Apparent resistivity rho_a = K * R (ohm·m)."
    )

    stack_count: int | None = Field(default=None, description="Number of stacks.")
    reciprocal_pair_id: str | None = Field(
        default=None, description="ID of the reciprocal measurement, if any."
    )

    contact_resistance_ohm: float | None = Field(default=None)
    instrument_channel: str | None = Field(default=None)
    qc_flags: list[str] = Field(default_factory=list)
    autocal_version: str | None = Field(default=None)
    raw_hash: str = Field(description="SHA-256 of the RAW record.")
    processing_provenance: dict[str, str] = Field(
        default_factory=dict,
        description="processing_step / processing_version / parent_artifact / timestamp.",
    )

    @field_validator("injected_current_a", "measured_voltage_v", mode="before")
    @classmethod
    def _not_zero_for_null(cls, v: object) -> object:
        return v


class ERTObservation(BaseModel):
    """A full ERT survey observation (list of quadrupoles + survey metadata)."""

    survey_id: str
    blinded_site_id: str
    acquisition_id: str
    timestamp: datetime = Field(default_factory=_now)
    quadrupoles: list[ERTQuadrupoleMeasurement] = Field(default_factory=list)
    autocal_version: str | None = Field(default=None)
    raw_hash: str = Field(description="SHA-256 of the RAW observation payload.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


class ERTInvertedSection(BaseModel):
    """Standardized inverted ERT section."""

    survey_id: str
    blinded_site_id: str
    x: list[float] = Field(description="Lateral positions (m).")
    z: list[float] = Field(description="Depth positions (m, positive downward).")
    resistivity: list[float] = Field(description="Inverted resistivity (ohm·m).")
    uncertainty: list[float] | None = Field(
        default=None, description="Model uncertainty per cell, where available."
    )
    sensitivity: list[float] | None = Field(
        default=None, description="Coverage/sensitivity per cell, where available."
    )
    mesh_metadata: dict[str, object] = Field(
        default_factory=dict, description="Mesh metadata preserved separately."
    )
    parent_artifact: str = Field(description="Hash of the inversion-input artifact.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


class ERTFeatures(BaseModel):
    """Target-independent deterministic ERT feature set.

    Does NOT encode 'water probability' at this stage.
    """

    survey_id: str
    blinded_site_id: str
    resistivity_mean_ohm_m: float | None = None
    resistivity_median_ohm_m: float | None = None
    resistivity_std_ohm_m: float | None = None
    resistivity_p10_ohm_m: float | None = None
    resistivity_p90_ohm_m: float | None = None
    vertical_gradient_max: float | None = None
    horizontal_gradient_max: float | None = None
    n_layer_boundaries: int | None = None
    layer_boundary_depths_m: list[float] = Field(default_factory=list)
    depth_bin_means: list[float] = Field(default_factory=list)
    depth_bin_edges_m: list[float] = Field(default_factory=list)
    anomaly_geometry: dict[str, float] = Field(default_factory=dict)
    sensitivity_weighted_mean: float | None = None
    mean_model_uncertainty: float | None = None
    support_fraction: float | None = None
    quality_score: float | None = Field(
        default=None, description="0-1 ERT QC quality (see aquascan.quality.ert)."
    )
    parent_artifact: str = Field(description="Hash of the inverted-model artifact.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


# ── Seismic models ────────────────────────────────────────────────────


class SeismicShot(BaseModel):
    """One seismic shot (all receiver traces for one source position)."""

    survey_id: str
    blinded_site_id: str
    acquisition_id: str
    shot_id: str
    shot_x: float
    shot_y: float | None = None
    shot_elevation_m: float | None = None
    trigger_timestamp: datetime | None = Field(default=None)
    receiver_ids: list[str] = Field(default_factory=list)
    receiver_x: list[float] = Field(default_factory=list)
    receiver_y: list[float] | None = None
    receiver_elevation_m: list[float] | None = None
    sample_rate_hz: float | None = None
    record_length_s: float | None = None
    raw_waveforms: list[list[float]] = Field(
        default_factory=list,
        description="RAW waveform per receiver (trace index aligned with receiver_ids).",
    )
    stack_index: int | None = Field(default=None)
    channel_numbers: list[int] | None = None
    geophone_metadata: dict[str, object] = Field(default_factory=dict)
    qc_flags: list[str] = Field(default_factory=list)
    autocal_version: str | None = Field(default=None)
    raw_hash: str = Field(description="SHA-256 of the RAW shot payload.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


class SeismicObservation(BaseModel):
    """A full seismic survey observation (list of shots)."""

    survey_id: str
    blinded_site_id: str
    acquisition_id: str
    timestamp: datetime = Field(default_factory=_now)
    shots: list[SeismicShot] = Field(default_factory=list)
    autocal_version: str | None = Field(default=None)
    raw_hash: str = Field(description="SHA-256 of the RAW observation payload.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


class SeismicPick(BaseModel):
    """A single first-arrival pick with provenance."""

    shot_id: str
    receiver_id: str
    pick_time_s: float
    pick_uncertainty_s: float | None = None
    method: Literal["manual", "automatic", "reviewed_automatic"] = "automatic"
    review_status: Literal["pending", "reviewed"] = "pending"
    parent_pick_artifact: str | None = Field(
        default=None, description="Never silently overwrite a previous pick."
    )


class SeismicVelocityModel(BaseModel):
    """Standardized P-wave velocity model from seismic inversion/tomography."""

    survey_id: str
    blinded_site_id: str
    x: list[float] = Field(description="Lateral positions (m).")
    z: list[float] = Field(description="Depth positions (m, positive downward).")
    vp: list[float] = Field(description="P-wave velocity (m/s).")
    uncertainty: list[float] | None = None
    ray_coverage: list[float] | None = None
    mesh_metadata: dict[str, object] = Field(default_factory=dict)
    parent_artifact: str = Field(description="Hash of the travel-time artifact.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


class SeismicFeatures(BaseModel):
    """Target-independent deterministic seismic feature set."""

    survey_id: str
    blinded_site_id: str
    vp_mean_m_s: float | None = None
    vp_median_m_s: float | None = None
    vp_std_m_s: float | None = None
    vertical_gradient_max: float | None = None
    lateral_gradient_max: float | None = None
    velocity_contrast_max: float | None = None
    travel_time_residual_rms_s: float | None = None
    first_arrival_uncertainty_mean_s: float | None = None
    depth_bin_means: list[float] = Field(default_factory=list)
    coverage_mean: float | None = None
    signal_quality_mean: float | None = None
    quality_score: float | None = Field(default=None, description="0-1 seismic QC quality.")
    parent_artifact: str = Field(description="Hash of the velocity-model artifact.")
    processing_provenance: dict[str, str] = Field(default_factory=dict)


# ── Shared physical representation (fusion-ready) ─────────────────────


class PhysicalRepresentation(BaseModel):
    """Fusion-ready shared physical grid.

    Channels (depth x lateral) with modality masks. No target labels.
    Interpolation is a derived representation; native models are preserved
    separately.
    """

    blinded_site_id: str
    depth_bins_m: list[float] = Field(description="Depth bin edges (m).")
    lateral_bins_m: list[float] = Field(description="Lateral bin edges (m).")
    ert_resistivity: list[list[float]] | None = Field(
        default=None, description="ERT resistivity channel (n_depth x n_lateral)."
    )
    ert_uncertainty: list[list[float]] | None = None
    ert_sensitivity: list[list[float]] | None = None
    seismic_vp: list[list[float]] | None = Field(
        default=None, description="P-wave velocity channel (n_depth x n_lateral)."
    )
    seismic_uncertainty: list[list[float]] | None = None
    seismic_coverage: list[list[float]] | None = None
    topography: list[float] | None = Field(
        default=None, description="Topography along lateral bins (m)."
    )
    modality_masks: dict[str, list[list[bool]]] = Field(
        default_factory=dict,
        description="Per-modality validity mask (True where the modality has support).",
    )
    parent_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Hash lineage: {'ert': ..., 'seismic': ...}.",
    )
    processing_provenance: dict[str, str] = Field(default_factory=dict)
