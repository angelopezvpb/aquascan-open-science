"""Leakage-safe modeling utilities for AquaScan."""

from aquascan.modeling.national_depth import (
    FEATURES,
    FOLD_GROUP_COLUMN,
    TARGET_COLUMN,
    NationalDepthData,
    build_national_depth_frame,
    build_ood_policy,
    conformal_quantile,
    make_model,
    make_preprocessor,
    proximity_groups,
    regression_metrics,
    run_domain_holdout,
    run_grouped_cv,
    train_final_model,
)

__all__ = [
    "FEATURES",
    "FOLD_GROUP_COLUMN",
    "TARGET_COLUMN",
    "NationalDepthData",
    "build_national_depth_frame",
    "build_ood_policy",
    "conformal_quantile",
    "make_model",
    "make_preprocessor",
    "proximity_groups",
    "regression_metrics",
    "run_domain_holdout",
    "run_grouped_cv",
    "train_final_model",
]
