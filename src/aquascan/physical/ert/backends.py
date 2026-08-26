"""ERT inversion backend abstraction (Gate 10, section 5).

AquaScan must not be permanently bound to one ERT inversion library.
This module defines the `ERTInversionBackend` interface and a registry
of available backends (pyGIMLi, SimPEG) with their status.

A minimal deterministic public-domain analytic backend (1D layered slab)
is included so software contracts can be validated without optional
dependencies. It is an ENGINEERING tool, not a production inversion.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from aquascan.physical.models import ERTInvertedSection


class ERTInversionBackend(ABC):
    """Interface every ERT inversion backend must implement."""

    name: str = "base"
    license: str = "unknown"

    @abstractmethod
    def invert(
        self,
        x: np.ndarray,
        z: np.ndarray,
        resistivity: np.ndarray,
        *,
        uncertainty: np.ndarray | None = None,
        sensitivity: np.ndarray | None = None,
        mesh_metadata: dict[str, object] | None = None,
    ) -> ERTInvertedSection:
        """Invert apparent resistivity into an ERTInvertedSection.

        Implementations must return the standardized section representation
        and must never treat a low-resistivity zone as groundwater.
        """
        raise NotImplementedError


@dataclass
class BackendStatus:
    """Availability status of an optional backend."""

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


def get_ert_backend_status() -> list[BackendStatus]:
    """Registry of ERT inversion backends and their local availability."""
    return [
        BackendStatus(
            name="pyGIMLi",
            available=importlib.util.find_spec("pygimli") is not None,
            version=_module_version("pygimli"),
            license="Apache-2.0",
            note="Preferred open-source ERT inversion backend when installed.",
        ),
        BackendStatus(
            name="SimPEG",
            available=importlib.util.find_spec("SimPEG") is not None,
            version=_module_version("SimPEG"),
            license="MIT",
            note="Open-source geophysical simulation/inversion framework.",
        ),
        BackendStatus(
            name="AquaScanAnalytic1D",
            available=True,
            version="0.1.0",
            license="Apache-2.0",
            note=(
                "Public-domain deterministic 1D layered-slab analytic backend "
                "for software-contract validation ONLY."
            ),
        ),
    ]


def available_ert_backends() -> list[ERTInversionBackend]:
    """Instantiate all locally available backends."""
    out: list[ERTInversionBackend] = [Analytic1DERTBackend()]
    for status in get_ert_backend_status():
        if status.name == "pyGIMLi" and status.available:
            out.append(PyGIMLiERTBackend())
        elif status.name == "SimPEG" and status.available:
            out.append(SimPEGERTBackend())
    return out


class Analytic1DERTBackend(ERTInversionBackend):
    """Deterministic 1D layered-slab analytic ERT backend.

    Computes a pseudo-section by a depth-of-investigation weighted
    averaging of apparent resistivity. This is a PUBLIC-DOMAIN engineering
    approximation for software validation only. It is NOT a true 2D
    inversion and MUST NOT be used to interpret real field data.
    """

    name = "AquaScanAnalytic1D"
    license = "Apache-2.0"

    def invert(
        self,
        x: np.ndarray,
        z: np.ndarray,
        resistivity: np.ndarray,
        *,
        uncertainty: np.ndarray | None = None,
        sensitivity: np.ndarray | None = None,
        mesh_metadata: dict[str, object] | None = None,
    ) -> ERTInvertedSection:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        rho = np.asarray(resistivity, dtype=float)
        if x.shape != rho.shape or z.shape != rho.shape:
            raise ValueError("x, z and resistivity must share the same shape")

        # Pseudo-inversion: smooth along depth with an exponential kernel
        # to emulate finite-depth sensitivity, deterministic and invertible.
        z_flat = z.ravel()
        rho_flat = rho.ravel()
        order = np.argsort(z_flat)
        z_sorted = z_flat[order]
        rho_sorted = rho_flat[order]
        # Depth-smoothed resistivity (moving weighted average)
        smoothed = np.empty_like(rho_sorted)
        for i, zi in enumerate(z_sorted):
            w = np.exp(-np.abs(z_sorted - zi) / max(zi, 1.0))
            smoothed[i] = np.average(rho_sorted, weights=w)
        rho_out = np.empty_like(rho_flat)
        rho_out[order] = smoothed

        unc = (
            np.full_like(rho_flat, float("nan"))
            if uncertainty is None
            else np.asarray(uncertainty).ravel()
        )
        sens = (
            np.full_like(rho_flat, 1.0) if sensitivity is None else np.asarray(sensitivity).ravel()
        )

        return ERTInvertedSection(
            survey_id=(
                str(mesh_metadata.get("survey_id", "unknown")) if mesh_metadata else "unknown"
            ),
            blinded_site_id=(
                str(mesh_metadata.get("blinded_site_id", "unknown")) if mesh_metadata else "unknown"
            ),
            x=list(x.ravel()),
            z=list(z.ravel()),
            resistivity=list(rho_out),
            uncertainty=list(unc),
            sensitivity=list(sens),
            mesh_metadata=mesh_metadata or {},
            parent_artifact=str(mesh_metadata.get("parent_artifact", "unknown"))
            if mesh_metadata
            else "unknown",
            processing_provenance={
                "processing_step": "ert_analytic1d_inversion",
                "processing_version": "0.1.0",
                "note": "Engineering validation backend; not a production 2D inversion.",
            },
        )


class PyGIMLiERTBackend(ERTInversionBackend):
    """Adapter for pyGIMLi (loaded only when installed)."""

    name = "pyGIMLi"
    license = "Apache-2.0"

    def invert(self, x, z, resistivity, *, uncertainty=None, sensitivity=None, mesh_metadata=None):
        raise NotImplementedError(
            "pyGIMLi adapter declared but pyGIMLi is not installed in this environment."
        )


class SimPEGERTBackend(ERTInversionBackend):
    """Adapter for SimPEG (loaded only when installed)."""

    name = "SimPEG"
    license = "MIT"

    def invert(self, x, z, resistivity, *, uncertainty=None, sensitivity=None, mesh_metadata=None):
        raise NotImplementedError(
            "SimPEG adapter declared but SimPEG is not installed in this environment."
        )
