"""Fusion contract and cross-modal registration."""

from aquascan.physical.fusion.contract import (
    FusionArm,
    FusionResult,
    GeoAIFusionInput,
    validate_arm_modalities,
)
from aquascan.physical.fusion.registration import (
    build_shared_representation,
    register_cross_modal,
)

__all__ = [
    "GeoAIFusionInput",
    "FusionArm",
    "FusionResult",
    "validate_arm_modalities",
    "build_shared_representation",
    "register_cross_modal",
]
