"""Synthetic data generation for testing and development.

WARNING: Synthetic data tests code correctness but CANNOT validate
real-world performance. Never present synthetic-only results as
evidence of field capability.
"""

from aquascan.simulation.synthetic import (
    generate_multilayer_profile,
    generate_synthetic_campaign,
    generate_synthetic_dataset,
)

__all__ = [
    "generate_synthetic_campaign",
    "generate_synthetic_dataset",
    "generate_multilayer_profile",
]
