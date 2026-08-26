"""Synthetic engineering fixtures with analytic forward models.

Fixtures are labeled SYNTHETIC_ENGINEERING_FIXTURE. They may be used to:

- test parsers, validate dimensions, test preprocessing,
- validate inversion interfaces, test encoders,
- test uncertainty propagation and multimodal software contracts.

They MUST NOT be used to claim groundwater detection or any real field
performance. Forward models are public-domain analytic physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SYNTHETIC_WARNING = (
    "SYNTHETIC_ENGINEERING_FIXTURE — software test only. "
    "NOT Ground Truth for AquaScan groundwater performance."
)


# ── ERT forward models ────────────────────────────────────────────────


def wenner_apparent_resistivity_twolayer(
    a: np.ndarray,
    rho1: float,
    rho2: float,
    h1: float,
    n_terms: int = 60,
) -> np.ndarray:
    """Analytic two-layer Wenner apparent resistivity (public domain).

    rho_a(a) = rho1 * (1 + 4 * sum_n [ k^n / sqrt(1+(2nh/a)^2)
                                           - k^n / sqrt(4+(2nh/a)^2) ])

    k = (rho2 - rho1) / (rho2 + rho1)  (reflection coefficient)
    a = Wenner spacing (m), h1 = thickness of layer 1 (m).
    """
    k = (rho2 - rho1) / (rho2 + rho1)
    out = np.full_like(a, rho1, dtype=float)
    for n in range(1, n_terms + 1):
        kn = k**n
        term = kn / np.sqrt(1.0 + (2.0 * n * h1 / a) ** 2) - kn / np.sqrt(
            4.0 + (2.0 * n * h1 / a) ** 2
        )
        out += 4.0 * rho1 * term
    return out


@dataclass
class ErtHalfspaceFixture:
    """Homogeneous half-space fixture: apparent resistivity == rho."""

    rho: float = 100.0
    spacing_m: np.ndarray = field(default_factory=lambda: np.arange(5.0, 50.0, 5.0))
    label: str = "SYNTHETIC_ENGINEERING_FIXTURE"

    def forward(self) -> np.ndarray:
        return np.full_like(self.spacing_m, self.rho, dtype=float)


@dataclass
class ErtTwoLayerFixture:
    """Two-layer Wenner fixture with analytic forward model."""

    rho1: float = 100.0
    rho2: float = 1000.0
    h1: float = 10.0
    spacing_m: np.ndarray = field(default_factory=lambda: np.linspace(5.0, 100.0, 12))
    label: str = "SYNTHETIC_ENGINEERING_FIXTURE"

    def forward(self) -> np.ndarray:
        return wenner_apparent_resistivity_twolayer(self.spacing_m, self.rho1, self.rho2, self.h1)


# ── Seismic forward models ────────────────────────────────────────────


@dataclass
class SeismicConstantVelocityFixture:
    """Constant-velocity direct-wave fixture: t = offset / v."""

    vp: float = 1000.0
    offsets_m: np.ndarray = field(default_factory=lambda: np.arange(5.0, 100.0, 5.0))
    label: str = "SYNTHETIC_ENGINEERING_FIXTURE"

    def forward(self) -> np.ndarray:
        return self.offsets_m / self.vp


@dataclass
class SeismicTwoLayerFixture:
    """Two-layer refraction fixture (direct wave + head wave)."""

    v1: float = 800.0
    v2: float = 2500.0
    h1: float = 15.0
    offsets_m: np.ndarray = field(default_factory=lambda: np.linspace(5.0, 200.0, 20))
    label: str = "SYNTHETIC_ENGINEERING_FIXTURE"

    def forward(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (direct_times, first_arrival_times) for all offsets.

        Physically correct first-arrival times: the refracted head wave is
        the first arrival only beyond the crossover distance (where the
        head-wave travel time drops below the direct wave). Between the
        critical and crossover distances the direct wave still arrives
        first, so times = min(direct, head).
        """
        i_crit = np.arcsin(self.v1 / self.v2)
        direct = self.offsets_m / self.v1
        head = 2.0 * self.h1 * np.cos(i_crit) / self.v1 + self.offsets_m / self.v2
        times = np.minimum(direct, head)
        return direct, times
