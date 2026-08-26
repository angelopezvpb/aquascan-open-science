"""ERT encoders.

Two implementations:
- ERTHandcraftedEncoder: scientifically interpretable summary features.
- ERTGridEncoder: resampling scaffold for a future neural representation.
  Structurally tested ONLY; never claimed as trained for groundwater
  detection without real paired field GT.

Variable geometry (electrode count, spacing, profile length, mesh) is
handled by resampling onto a canonical grid with explicit support masks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from aquascan.physical.models import ERTInvertedSection


@dataclass
class ERTEncoderOutput:
    """Standardized encoder output."""

    embedding: np.ndarray
    quality: float
    uncertainty: float
    support_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


class ERTEncoder(ABC):
    """Interface for all ERT encoders."""

    name: str = "base"

    @abstractmethod
    def encode(self, section: ERTInvertedSection) -> ERTEncoderOutput:
        raise NotImplementedError


def _to_grid(
    section: ERTInvertedSection,
    n_lateral: int,
    n_depth: int,
    xmax: float | None = None,
    zmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a native irregular section onto a canonical grid.

    Returns (values, mask) with shape (n_depth, n_lateral). Out-of-support
    cells are masked, never filled with zeros-as-real-measurements.
    """
    x = np.asarray(section.x, dtype=float)
    z = np.asarray(section.z, dtype=float)
    v = np.asarray(section.resistivity, dtype=float)
    if xmax is None:
        xmax = float(np.max(x)) if len(x) else 1.0
    if zmax is None:
        zmax = float(np.max(z)) if len(z) else 1.0
    grid = np.full((n_depth, n_lateral), np.nan)
    mask = np.zeros((n_depth, n_lateral), dtype=bool)
    xbins = np.linspace(0.0, xmax, n_lateral + 1)
    zbins = np.linspace(0.0, zmax, n_depth + 1)
    for xi, zi, vi in zip(x, z, v, strict=False):
        if not (0 <= xi <= xmax and 0 <= zi <= zmax):
            continue
        ix = int(np.clip(np.searchsorted(xbins, xi, side="right") - 1, 0, n_lateral - 1))
        iz = int(np.clip(np.searchsorted(zbins, zi, side="right") - 1, 0, n_depth - 1))
        # nearest-neighbour fill (deterministic)
        if not mask[iz, ix] or np.isnan(grid[iz, ix]):
            grid[iz, ix] = vi
            mask[iz, ix] = True
    return grid, mask


class ERTHandcraftedEncoder(ERTEncoder):
    """Interpretable handcrafted ERT encoder."""

    name = "ERT_HANDCRAFTED"

    def encode(self, section: ERTInvertedSection) -> ERTEncoderOutput:
        rho = np.asarray(section.resistivity, dtype=float)
        z = np.asarray(section.z, dtype=float)
        sens = np.asarray(section.sensitivity, dtype=float) if section.sensitivity else None
        unc = np.asarray(section.uncertainty, dtype=float) if section.uncertainty else None

        if len(rho) == 0:
            return ERTEncoderOutput(
                embedding=np.zeros(14),
                quality=0.0,
                uncertainty=1.0,
                support_mask=np.zeros(0, dtype=bool),
                metadata={"error": "empty section"},
            )

        # log-resistivity distribution (robust to dynamic range)
        logrho = np.log(rho + 1e-9)
        depth_bin_means: list[float] = []
        n_bins = 6
        edges = np.linspace(0.0, float(np.max(z)) + 1e-9, n_bins + 1)
        for i in range(n_bins):
            m = (z >= edges[i]) & (z < edges[i + 1])
            depth_bin_means.append(float(np.mean(logrho[m])) if m.sum() else float("nan"))

        # vertical gradient profile
        order = np.argsort(z)
        dz = np.diff(z[order])
        dlog = np.diff(logrho[order])
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.abs(np.where(dz > 0, dlog / np.maximum(dz, 1e-9), np.nan))
        grad = grad[~np.isnan(grad)]

        embedding = np.array(
            [
                float(np.mean(logrho)),
                float(np.median(logrho)),
                float(np.std(logrho)),
                float(np.percentile(logrho, 10)),
                float(np.percentile(logrho, 90)),
                float(np.min(logrho)),
                float(np.max(logrho)),
                float(np.mean(depth_bin_means)) if depth_bin_means else float("nan"),
                float(np.nanmax(grad)) if len(grad) else float("nan"),
                float(np.mean(grad)) if len(grad) else float("nan"),
                float(np.mean(rho > np.median(rho))),
                float(np.mean(sens)) if sens is not None else float("nan"),
                float(np.mean(unc)) if unc is not None else float("nan"),
                float(len(rho)),
            ]
        )
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        quality = float(np.mean(sens > 0)) if sens is not None else 0.5
        uncertainty = float(np.mean(unc)) if unc is not None else 1.0
        support_mask = np.ones(len(rho), dtype=bool) if sens is None else sens > 0

        return ERTEncoderOutput(
            embedding=embedding,
            quality=quality,
            uncertainty=uncertainty,
            support_mask=support_mask,
            metadata={"feature_source": "ERTFeatures-derived", "n_cells": int(len(rho))},
        )


class ERTGridEncoder(ERTEncoder):
    """Resampling scaffold for a future neural/convolational ERT encoder.

    NOT TRAINED. This scaffold validates geometry handling, resampling and
    support masking only. It MUST NOT be claimed as a groundwater detector.
    """

    name = "ERT_GRID"
    trained = False

    def __init__(self, n_lateral: int = 32, n_depth: int = 16) -> None:
        self.n_lateral = n_lateral
        self.n_depth = n_depth

    def encode(self, section: ERTInvertedSection) -> ERTEncoderOutput:
        grid, mask = _to_grid(section, self.n_lateral, self.n_depth)
        # NaN -> explicit missing sentinel; support mask keeps NaN cells
        # distinguishable from real (near-)zero measurements.
        embedding = np.nan_to_num(grid, nan=0.0)
        quality = float(np.mean(mask))
        uncertainty = 1.0 - float(np.mean(mask))
        return ERTEncoderOutput(
            embedding=embedding,
            quality=quality,
            uncertainty=uncertainty,
            support_mask=mask,
            metadata={
                "trained": False,
                "note": "Untrained scaffold. No groundwater claims.",
                "n_lateral": self.n_lateral,
                "n_depth": self.n_depth,
            },
        )
