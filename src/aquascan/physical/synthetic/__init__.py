"""Synthetic engineering fixtures.

All fixtures are labeled SYNTHETIC_ENGINEERING_FIXTURE. They are for
software tests only, NOT Ground Truth for AquaScan groundwater performance.
"""

from aquascan.physical.synthetic.fixtures import (
    SYNTHETIC_WARNING,
    ErtHalfspaceFixture,
    ErtTwoLayerFixture,
    SeismicConstantVelocityFixture,
    SeismicTwoLayerFixture,
)

__all__ = [
    "SYNTHETIC_WARNING",
    "ErtHalfspaceFixture",
    "ErtTwoLayerFixture",
    "SeismicConstantVelocityFixture",
    "SeismicTwoLayerFixture",
]
