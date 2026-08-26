"""Selective prediction / abstention evaluation.

AquaScan must be able to say "NO CONCLUYENTE" when evidence is insufficient.
This module provides infrastructure for studying the trade-off between
coverage (fraction of predictions made) and error.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_risk_coverage(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute risk-coverage curve data.

    Sorts predictions by confidence, then computes error at each coverage
    level (fraction of highest-confidence predictions kept).

    Args:
        y_true: Ground truth binary labels (0/1).
        y_score: Predicted scores/probabilities.
        confidence: Confidence estimates for each prediction.

    Returns:
        Dict with 'coverage', 'error_rate', and 'selective_risk' arrays.
    """
    n = len(y_true)

    # Sort by confidence (descending)
    order = np.argsort(-confidence)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    coverages = np.linspace(0.1, 1.0, 100)
    errors = np.zeros(len(coverages))

    for i, cov in enumerate(coverages):
        n_keep = max(1, int(cov * n))
        y_pred = (y_score_sorted[:n_keep] >= 0.5).astype(int)
        errors[i] = np.mean(y_pred != y_true_sorted[:n_keep])

    return {
        "coverage": coverages,
        "error_rate": errors,
    }


def abstention_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    evidence_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute abstention analysis: coverage vs error as we vary evidence threshold.

    Higher evidence threshold → lower coverage but better accuracy.
    Lower evidence threshold → higher coverage but worse accuracy.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores.
        evidence_scores: Evidence quality scores per prediction.
        thresholds: Evidence thresholds to evaluate. Default is 10 evenly spaced.

    Returns:
        Dict with 'thresholds', 'coverage', 'error_rate', 'abstention_rate'.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 20)

    n = len(y_true)
    coverage = np.zeros(len(thresholds))
    error_rates = np.zeros(len(thresholds))
    abstention_rates = np.zeros(len(thresholds))

    for i, thresh in enumerate(thresholds):
        keep = evidence_scores >= thresh
        n_kept = np.sum(keep)

        if n_kept > 0:
            y_pred = (y_score[keep] >= 0.5).astype(int)
            error_rates[i] = np.mean(y_pred != y_true[keep])
            coverage[i] = n_kept / n
            abstention_rates[i] = 1.0 - coverage[i]
        else:
            error_rates[i] = 1.0
            coverage[i] = 0.0
            abstention_rates[i] = 1.0

    return {
        "thresholds": thresholds,
        "coverage": coverage,
        "error_rate": error_rates,
        "abstention_rate": abstention_rates,
    }


def selective_prediction_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    evidence_scores: np.ndarray,
    evidence_threshold: float = 0.4,
) -> dict[str, float]:
    """Compute metrics under selective prediction.

    Only predictions with evidence_score >= threshold are evaluated.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores.
        evidence_scores: Evidence scores.
        evidence_threshold: Minimum evidence to include.

    Returns:
        Metrics dict with coverage and error rate.
    """
    keep = evidence_scores >= evidence_threshold
    n_total = len(y_true)
    n_kept = int(np.sum(keep))

    if n_kept == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "error_rate_on_kept": float("nan"),
            "n_kept": 0,
            "n_total": n_total,
        }

    y_pred = (y_score[keep] >= 0.5).astype(int)
    error_on_kept = float(np.mean(y_pred != y_true[keep]))

    return {
        "coverage": round(n_kept / n_total, 4),
        "abstention_rate": round(1.0 - n_kept / n_total, 4),
        "error_rate_on_kept": round(error_on_kept, 4),
        "n_kept": n_kept,
        "n_total": n_total,
    }
