"""Evaluation metrics for groundwater target prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_score: Predicted scores/probabilities for class 1.

    Returns:
        Dictionary of metric names to values.
    """
    metrics: dict[str, float] = {}

    # Basic metrics
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(np.mean(y_true == y_pred))

    # Class balance
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = n_pos
    metrics["n_negative"] = n_neg
    metrics["positive_ratio"] = n_pos / len(y_true) if len(y_true) > 0 else float("nan")

    # Ranking metrics (require scores)
    if y_score is not None:
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    return metrics


def compute_depth_error(
    y_true_depth: np.ndarray,
    y_pred_depth: np.ndarray,
) -> dict[str, float]:
    """Compute depth prediction error metrics.

    Args:
        y_true_depth: True depths (meters).
        y_pred_depth: Predicted depths (meters).

    Returns:
        Dict with MAE, RMSE, MAPE (%), and bias.
    """
    mask = ~(np.isnan(y_true_depth) | np.isnan(y_pred_depth))
    true = y_true_depth[mask]
    pred = y_pred_depth[mask]

    if len(true) == 0:
        return {
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "mape_pct": float("nan"),
            "bias_m": float("nan"),
        }

    errors = pred - true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(abs_errors / np.maximum(np.abs(true), 1e-6)) * 100)
    bias = float(np.mean(errors))

    return {"mae_m": mae, "rmse_m": rmse, "mape_pct": mape, "bias_m": bias}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    y_true_depth: np.ndarray | None = None,
    y_pred_depth: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all available metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Predicted scores.
        y_true_depth: True depth values.
        y_pred_depth: Predicted depth values.

    Returns:
        Combined metrics dictionary.
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    if y_true_depth is not None and y_pred_depth is not None:
        depth_metrics = compute_depth_error(y_true_depth, y_pred_depth)
        metrics.update(depth_metrics)

    return metrics
