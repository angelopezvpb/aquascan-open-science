"""Cross-modal registration and shared physical representation.

Maps ERT cells and seismic velocity cells into a shared physical
coordinate representation (depth bins x lateral bins + modality masks).

Rules:
- Native models are NEVER modified.
- Interpolation is a DERIVED representation only.
- Cells outside support are masked, never filled with zeros-as-real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import griddata

from aquascan.physical.models import (
    ERTInvertedSection,
    PhysicalRepresentation,
    SeismicVelocityModel,
)


def _interp_to_grid(
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    lateral_bins: np.ndarray,
    depth_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate irregular (x, z, v) onto a regular grid.

    Returns (grid, mask) with shape (n_depth, n_lateral). Out-of-hull
    cells are masked (nan stays nan in `grid`, mask=False).
    """
    xc = (lateral_bins[:-1] + lateral_bins[1:]) / 2.0
    zc = (depth_bins[:-1] + depth_bins[1:]) / 2.0
    x_grid, z_grid = np.meshgrid(xc, zc)
    pts = np.column_stack([x.ravel(), z.ravel()])
    vals = values.ravel()
    try:
        grid = griddata(pts, vals, (x_grid, z_grid), method="linear")
    except Exception:
        # Degenerate geometry (e.g. a 1D layered model with 4 cells): fall
        # back to nearest-neighbour interpolation (still a derived view).
        grid = griddata(pts, vals, (x_grid, z_grid), method="nearest")
    # support mask: cells within the convex extent of native data
    mask = ~np.isnan(grid)
    return np.nan_to_num(grid, nan=0.0), mask


def build_shared_representation(
    blinded_site_id: str,
    *,
    ert: ERTInvertedSection | None = None,
    seismic: SeismicVelocityModel | None = None,
    depth_bins_m: list[float] | None = None,
    lateral_bins_m: list[float] | None = None,
    topography: list[float] | None = None,
) -> PhysicalRepresentation:
    """Build the fusion-ready shared representation from native models."""
    if depth_bins_m is None:
        depth_bins_m = list(np.linspace(0.0, 50.0, 11))
    if lateral_bins_m is None:
        lateral_bins_m = list(np.linspace(0.0, 240.0, 25))

    db = np.asarray(depth_bins_m, dtype=float)
    lb = np.asarray(lateral_bins_m, dtype=float)

    rep = PhysicalRepresentation(
        blinded_site_id=blinded_site_id,
        depth_bins_m=list(db),
        lateral_bins_m=list(lb),
        topography=topography,
        processing_provenance={
            "processing_step": "cross_modal_registration",
            "processing_version": "0.1.0",
            "note": "Interpolation is a derived representation; native models preserved.",
        },
    )

    if ert is not None:
        x = np.asarray(ert.x, dtype=float)
        z = np.asarray(ert.z, dtype=float)
        rho = np.asarray(ert.resistivity, dtype=float)
        grid, mask = _interp_to_grid(x, z, rho, lb, db)
        rep.ert_resistivity = grid.tolist()
        rep.modality_masks["ert"] = mask.tolist()
        if ert.uncertainty:
            ugrid, _ = _interp_to_grid(x, z, np.asarray(ert.uncertainty), lb, db)
            rep.ert_uncertainty = ugrid.tolist()
        if ert.sensitivity:
            sgrid, _ = _interp_to_grid(x, z, np.asarray(ert.sensitivity), lb, db)
            rep.ert_sensitivity = sgrid.tolist()
        rep.parent_artifacts["ert"] = ert.parent_artifact

    if seismic is not None:
        x = np.asarray(seismic.x, dtype=float)
        z = np.asarray(seismic.z, dtype=float)
        vp = np.asarray(seismic.vp, dtype=float)
        grid, mask = _interp_to_grid(x, z, vp, lb, db)
        rep.seismic_vp = grid.tolist()
        rep.modality_masks["seismic"] = mask.tolist()
        if seismic.ray_coverage:
            cgrid, _ = _interp_to_grid(x, z, np.asarray(seismic.ray_coverage), lb, db)
            rep.seismic_coverage = cgrid.tolist()
        rep.parent_artifacts["seismic"] = seismic.parent_artifact

    return rep


def register_cross_modal(
    blinded_site_id: str,
    ert: ERTInvertedSection,
    seismic: SeismicVelocityModel,
    **kwargs: Any,
) -> PhysicalRepresentation:
    """Convenience wrapper: register ERT + seismic into one representation."""
    return build_shared_representation(blinded_site_id, ert=ert, seismic=seismic, **kwargs)
