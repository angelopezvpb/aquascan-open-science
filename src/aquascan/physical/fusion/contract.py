"""Multimodal fusion contract.

GeoAIFusionInput supports optional modalities:

    GeoAIFusionInput:
        context: ContextRepresentation | None
        ert: ERTRepresentation | None
        seismic: SeismicRepresentation | None

plus quality / support / uncertainty / provenance for each modality.

The contract supports missing modalities (ARM_E without context, etc.).
No early fusion by accident: each modality is kept separable so the
Phase 0 ablation (E / S / ES / CONTEXT / CONTEXT+ES) can be run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FusionArm(StrEnum):
    """Supported analysis arms."""

    ARM_CONTEXT = "ARM_CONTEXT"
    ARM_E = "ARM_E"
    ARM_S = "ARM_S"
    ARM_ES = "ARM_ES"
    ARM_CONTEXT_ES = "ARM_CONTEXT_ES"


# Required modalities per arm. Arms are strict: an arm must not silently
# receive information from modalities it is supposed to be blind to.
ARM_REQUIREMENTS: dict[FusionArm, set[str]] = {
    FusionArm.ARM_CONTEXT: {"context"},
    FusionArm.ARM_E: {"ert"},
    FusionArm.ARM_S: {"seismic"},
    FusionArm.ARM_ES: {"ert", "seismic"},
    FusionArm.ARM_CONTEXT_ES: {"context", "ert", "seismic"},
}


@dataclass
class ModalityBundle:
    """One modality's representation + quality + support + provenance."""

    representation: dict[str, Any]
    quality: float = 0.0
    uncertainty: float = 1.0
    support_mask: Any = None
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class GeoAIFusionInput:
    """Input to the future GeoAIFusionModel. Modalities may be missing."""

    blinded_site_id: str
    context: ModalityBundle | None = None
    ert: ModalityBundle | None = None
    seismic: ModalityBundle | None = None


@dataclass
class FusionResult:
    """Standardized fusion output."""

    blinded_site_id: str
    arm: FusionArm
    prediction: dict[str, Any]
    uncertainty: dict[str, float]
    confidence_gate: str = "RED"  # GREEN / YELLOW / RED (abstain)
    provenance: dict[str, str] = field(default_factory=dict)


def validate_arm_modalities(
    inp: GeoAIFusionInput,
    arm: FusionArm,
) -> list[str]:
    """Validate that the input provides exactly the required modalities.

    Returns a list of missing-required-modality names. An empty list means
    the arm can run. Extra modalities are NOT an error here (the caller
    decides how to fuse), but missing required ones are.
    """
    if arm not in ARM_REQUIREMENTS:
        raise ValueError(f"Unknown fusion arm: {arm}")
    required = ARM_REQUIREMENTS[arm]
    provided: set[str] = set()
    if inp.context is not None:
        provided.add("context")
    if inp.ert is not None:
        provided.add("ert")
    if inp.seismic is not None:
        provided.add("seismic")
    return sorted(required - provided)


def modality_quality(inp: GeoAIFusionInput) -> dict[str, float]:
    """Explicit quality per modality (missing -> 0.0, never fake)."""
    return {
        "context": inp.context.quality if inp.context else 0.0,
        "ert": inp.ert.quality if inp.ert else 0.0,
        "seismic": inp.seismic.quality if inp.seismic else 0.0,
    }
