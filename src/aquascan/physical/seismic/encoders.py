"""Seismic encoders.

SEISMIC_HANDCRAFTED_ENCODER and SEISMIC_GRID_ENCODER (scaffold).
No claimed groundwater performance without real paired data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from aquascan.physical.models import SeismicVelocityModel


@dataclass
class SeismicEncoderOutput:
    """Standardized seismic encoder output."""

    embedding: np.ndarray
    quality: float
    uncertainty: float
    support_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


class SeismicEncoder(ABC):
    """Interface for all seismic encoders."""

    name: str = "base"

    @abstractmethod
    def encode(self, model: SeismicVelocityModel) -> SeismicEncoderOutput:
        raise NotImplementedError


class SeismicHandcraftedEncoder(SeismicEncoder):
    """Interpretable handcrafted seismic encoder."""

    name = "SEISMIC_HANDCRAFTED"

    def encode(self, model: SeismicVelocityModel) -> SeismicEncoderOutput:
        vp = np.asarray(model.vp, dtype=float)
        z = np.asarray(model.z, dtype=float)
        unc = np.asarray(model.uncertainty, dtype=float) if model.uncertainty else None
        cov = np.asarray(model.ray_coverage, dtype=float) if model.ray_coverage else None

        if len(vp) == 0:
            return SeismicEncoderOutput(
                embedding=np.zeros(10),
                quality=0.0,
                uncertainty=1.0,
                support_mask=np.zeros(0, dtype=bool),
                metadata={"error": "empty model"},
            )

        logvp = np.log(vp + 1e-9)
        embedding = np.array(
            [
                float(np.mean(logvp)),
                float(np.median(logvp)),
                float(np.std(logvp)),
                float(np.min(logvp)),
                float(np.max(logvp)),
                float(np.mean(np.abs(np.diff(logvp)))) if len(logvp) > 1 else 0.0,
                float(np.mean(cov)) if cov is not None else 0.0,
                float(np.mean(unc)) if unc is not None else 1.0,
                float(np.max(z)),
                float(len(vp)),
            ]
        )
        embedding = np.nan_to_num(embedding, nan=0.0)

        quality = float(np.mean(cov > 0)) if cov is not None else 0.5
        uncertainty = float(np.mean(unc)) if unc is not None else 1.0
        support_mask = np.ones(len(vp), dtype=bool) if cov is None else cov > 0

        return SeismicEncoderOutput(
            embedding=embedding,
            quality=quality,
            uncertainty=uncertainty,
            support_mask=support_mask,
            metadata={"n_cells": int(len(vp))},
        )


class SeismicGridEncoder(SeismicEncoder):
    """Waveform/section grid scaffold for a future neural encoder.

    NOT TRAINED. Validates resampling and masking only.
    """

    name = "SEISMIC_GRID"
    trained = False

    def __init__(self, n_lateral: int = 32, n_depth: int = 16) -> None:
        self.n_lateral = n_lateral
        self.n_depth = n_depth

    def encode(self, model: SeismicVelocityModel) -> SeismicEncoderOutput:
        x = np.asarray(model.x, dtype=float)
        z = np.asarray(model.z, dtype=float)
        vp = np.asarray(model.vp, dtype=float)
        xmax = float(np.max(x)) if len(x) else 1.0
        zmax = float(np.max(z)) if len(z) else 1.0
        grid = np.full((self.n_depth, self.n_lateral), np.nan)
        mask = np.zeros((self.n_depth, self.n_lateral), dtype=bool)
        xbins = np.linspace(0.0, xmax, self.n_lateral + 1)
        zbins = np.linspace(0.0, zmax, self.n_depth + 1)
        for xi, zi, vi in zip(x, z, vp, strict=False):
            if not (0 <= xi <= xmax and 0 <= zi <= zmax):
                continue
            ix = int(np.clip(np.searchsorted(xbins, xi, side="right") - 1, 0, self.n_lateral - 1))
            iz = int(np.clip(np.searchsorted(zbins, zi, side="right") - 1, 0, self.n_depth - 1))
            if not mask[iz, ix] or np.isnan(grid[iz, ix]):
                grid[iz, ix] = vi
                mask[iz, ix] = True
        return SeismicEncoderOutput(
            embedding=np.nan_to_num(grid, nan=0.0),
            quality=float(np.mean(mask)),
            uncertainty=1.0 - float(np.mean(mask)),
            support_mask=mask,
            metadata={
                "trained": False,
                "note": "Untrained scaffold. No groundwater claims.",
                "n_lateral": self.n_lateral,
                "n_depth": self.n_depth,
            },
        )
