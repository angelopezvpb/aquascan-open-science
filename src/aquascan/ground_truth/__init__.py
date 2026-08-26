"""Ground Truth scoring and quality assessment."""

from aquascan.ground_truth.quality import GroundTruthResult, score_ground_truth
from aquascan.ground_truth.scoring import (
    CandidatePriorityResult,
    compute_candidate_priority_score,
)

__all__ = [
    "GroundTruthResult",
    "score_ground_truth",
    "CandidatePriorityResult",
    "compute_candidate_priority_score",
]
