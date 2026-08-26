"""Evaluation metrics, calibration, abstention, and geographic splits."""

from aquascan.evaluation.abstention import (
    abstention_curve,
    compute_risk_coverage,
    selective_prediction_metrics,
)
from aquascan.evaluation.calibration import (
    brier_score,
    calibration_curve,
    expected_calibration_error,
)
from aquascan.evaluation.geographic_split import (
    GeographicHoldoutSplit,
    SpatialDistanceSplit,
    check_geographic_leakage,
    split_by_geology_domain,
    split_by_well_id,
)
from aquascan.evaluation.metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_depth_error,
)

__all__ = [
    # Metrics
    "compute_all_metrics",
    "compute_classification_metrics",
    "compute_depth_error",
    # Calibration
    "brier_score",
    "calibration_curve",
    "expected_calibration_error",
    # Abstention
    "abstention_curve",
    "compute_risk_coverage",
    "selective_prediction_metrics",
    # Geographic split
    "GeographicHoldoutSplit",
    "SpatialDistanceSplit",
    "check_geographic_leakage",
    "split_by_geology_domain",
    "split_by_well_id",
]
