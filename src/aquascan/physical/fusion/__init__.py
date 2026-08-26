"""Fusion contract and cross-modal registration (Gate 10, sections 17-21)."""

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
