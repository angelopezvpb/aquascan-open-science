"""Seismic inversion backend abstraction.

Similar to the ERT backend abstraction. Optional open-source backends
(obspy-based workflows, pyGIMLi refraction) may be integrated later.

A deterministic public-domain intercept-time 1D refraction backend is
included for software-contract validation. P-wave velocity is never
treated as a direct water measurement.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from aquascan.physical.models import SeismicVelocityModel


class SeismicInversionBackend(ABC):
    """Interface every seismic inversion backend must implement."""

    name: str = "base"
    license: str = "unknown"

    @abstractmethod
    def invert(
        self,
        offsets_m: np.ndarray,
        travel_times_s: np.ndarray,
        *,
        uncertainty_s: np.ndarray | None = None,
        ray_coverage: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SeismicVelocityModel:
        """Invert travel times into a SeismicVelocityModel.

        Implementations must return the standardized representation and
        must never treat P-wave velocity as a direct water measurement.
        """
        raise NotImplementedError


@dataclass
class SeismicBackendStatus:
    """Availability status of an optional seismic backend."""

    name: str
    available: bool
    version: str | None
    license: str
    note: str


def _module_version(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def get_seismic_backend_status() -> list[SeismicBackendStatus]:
    return [
        SeismicBackendStatus(
            name="obspy",
            available=importlib.util.find_spec("obspy") is not None,
            version=_module_version("obspy"),
            license="LGPL-3.0",
            note="Open-source seismology toolbox (SEG-2/SEG-Y I/O, filtering).",
        ),
        SeismicBackendStatus(
            name="pyGIMLi_refraction",
            available=importlib.util.find_spec("pygimli") is not None,
            version=_module_version("pygimli"),
            license="Apache-2.0",
            note="Open-source refraction tomography backend when installed.",
        ),
        SeismicBackendStatus(
            name="AquaScanInterceptTime1D",
            available=True,
            version="0.1.0",
            license="Apache-2.0",
            note=(
                "Public-domain deterministic intercept-time 1D refraction "
                "backend for software-contract validation ONLY."
            ),
        ),
    ]


class InterceptTime1DSeismicBackend(SeismicInversionBackend):
    """Deterministic public-domain intercept-time refraction backend.

    Fits a two-segment travel-time curve (direct + refracted head wave)
    and builds a 1D velocity profile. Engineering validation only.
    """

    name = "AquaScanInterceptTime1D"
    license = "Apache-2.0"

    def invert(
        self,
        offsets_m: np.ndarray,
        travel_times_s: np.ndarray,
        *,
        uncertainty_s: np.ndarray | None = None,
        ray_coverage: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SeismicVelocityModel:
        x = np.asarray(offsets_m, dtype=float)
        t = np.asarray(travel_times_s, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(t))
        x, t = x[mask], t[mask]
        if len(x) < 8:
            raise ValueError("Insufficient travel-time picks for intercept-time fit.")

        order = np.argsort(x)
        x, t = x[order], t[order]
        n = len(x)

        # Brute-force two-segment breakpoint search: find the split point
        # minimizing combined SSE of two straight-line fits (direct wave +
        # refracted head wave). Deterministic and public domain.
        best_sse = np.inf
        best_k = -1
        for k in range(max(3, int(0.15 * n)), int(0.7 * n) + 1):
            if n - k < 3:
                continue
            try:
                p_near = np.polyfit(x[:k], t[:k], 1)
                p_far = np.polyfit(x[k:], t[k:], 1)
            except np.linalg.LinAlgError:
                continue
            if p_near[0] <= 0 or p_far[0] <= 0 or p_far[0] >= p_near[0]:
                continue
            sse = float(
                np.sum((np.polyval(p_near, x[:k]) - t[:k]) ** 2)
                + np.sum((np.polyval(p_far, x[k:]) - t[k:]) ** 2)
            )
            if sse < best_sse:
                best_sse = sse
                best_k = k
                best_near = p_near
                best_far = p_far
        if best_k < 0:
            raise ValueError("No valid two-segment travel-time fit found.")

        p_near, p_far = best_near, best_far
        v1 = float(1.0 / p_near[0])
        v2 = float(1.0 / p_far[0])
        if v1 <= 0 or v2 <= v1:
            raise ValueError("Non-physical velocities from intercept-time fit.")
        intercept = float(p_far[1])
        # h = (ti/2) / sqrt(1/v1^2 - 1/v2^2)
        h = (intercept / 2.0) / np.sqrt(1.0 / v1**2 - 1.0 / v2**2)

        survey_id = str(metadata.get("survey_id", "unknown")) if metadata else "unknown"
        blinded_site_id = str(metadata.get("blinded_site_id", "unknown")) if metadata else "unknown"

        # Build a two-layer velocity model (x/z uniform along profile)
        z_model = [0.0, h, h, 4 * h]
        x_model = [0.0, 0.0, float(np.max(x)), float(np.max(x))]
        vp = [v1, v1, v2, v2]
        return SeismicVelocityModel(
            survey_id=survey_id,
            blinded_site_id=blinded_site_id,
            x=x_model,
            z=z_model,
            vp=vp,
            uncertainty=list(np.full(4, 0.05 * v2)),
            ray_coverage=list(np.full(4, 1.0)),
            mesh_metadata={
                "method": "intercept_time_1d_breakpoint_search",
                "n_picks": n,
                "split_index": best_k,
                "intercept_s": intercept,
                "h_refractor_m": float(h),
            },
            parent_artifact=str(metadata.get("parent_artifact", "unknown"))
            if metadata
            else "unknown",
            processing_provenance={
                "processing_step": "seismic_intercept_time_1d",
                "processing_version": "0.1.0",
                "note": "Engineering validation backend; not a production tomography.",
            },
        )
