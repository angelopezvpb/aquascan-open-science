"""AquaScan Physical Encoder Stack v0.

Software architecture for ERT + P-wave seismic physical modalities.

FUNDAMENTAL RULE
================
Synthetic or public geophysical data may be used ONLY to test parsers,
validate dimensions, test preprocessing, validate inversion interfaces,
test encoders, test uncertainty propagation, and test multimodal software
contracts. It MUST NOT be used to claim groundwater detection or any real
field performance.

DOES NOT: query IGME, reveal Phase 0 GT, fabricate field results,
train the final multimodal GeoAI.
"""

from aquascan.physical.common import (
    ArtifactRef,
    ProcessingStage,
    compute_artifact_hash,
    make_artifact_ref,
)
from aquascan.physical.coords import (
    Elevation,
    ElevationType,
    LocalSurveyFrame,
    SurveyPoint,
    register_survey_points,
)
from aquascan.physical.ert.backends import ERTInversionBackend, get_ert_backend_status
from aquascan.physical.ert.encoders import ERTEncoder, ERTGridEncoder, ERTHandcraftedEncoder
from aquascan.physical.ert.pipeline import ERTPipeline
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
from aquascan.physical.models import (
    ERTFeatures,
    ERTInvertedSection,
    ERTObservation,
    ERTQuadrupoleMeasurement,
    PhysicalRepresentation,
    SeismicFeatures,
    SeismicObservation,
    SeismicShot,
    SeismicVelocityModel,
)
from aquascan.physical.seismic.backends import SeismicInversionBackend
from aquascan.physical.seismic.encoders import (
    SeismicEncoder,
    SeismicGridEncoder,
    SeismicHandcraftedEncoder,
)
from aquascan.physical.seismic.pipeline import SeismicPipeline
from aquascan.physical.synthetic.fixtures import (
    SYNTHETIC_WARNING,
    ErtHalfspaceFixture,
    ErtTwoLayerFixture,
    SeismicConstantVelocityFixture,
    SeismicTwoLayerFixture,
)

__all__ = [
    # common
    "ArtifactRef",
    "ProcessingStage",
    "compute_artifact_hash",
    "make_artifact_ref",
    # coords
    "Elevation",
    "ElevationType",
    "LocalSurveyFrame",
    "SurveyPoint",
    "register_survey_points",
    # models
    "ERTObservation",
    "ERTQuadrupoleMeasurement",
    "ERTInvertedSection",
    "ERTFeatures",
    "SeismicObservation",
    "SeismicShot",
    "SeismicVelocityModel",
    "SeismicFeatures",
    "PhysicalRepresentation",
    # ERT
    "ERTInversionBackend",
    "get_ert_backend_status",
    "ERTPipeline",
    "ERTEncoder",
    "ERTHandcraftedEncoder",
    "ERTGridEncoder",
    # Seismic
    "SeismicInversionBackend",
    "SeismicPipeline",
    "SeismicEncoder",
    "SeismicHandcraftedEncoder",
    "SeismicGridEncoder",
    # Fusion
    "GeoAIFusionInput",
    "FusionArm",
    "FusionResult",
    "validate_arm_modalities",
    "build_shared_representation",
    "register_cross_modal",
    # Synthetic fixtures
    "SYNTHETIC_WARNING",
    "ErtHalfspaceFixture",
    "ErtTwoLayerFixture",
    "SeismicConstantVelocityFixture",
    "SeismicTwoLayerFixture",
]
