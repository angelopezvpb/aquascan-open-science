"""Calibration evaluation for probabilistic predictions.

A well-calibrated model's predicted probabilities match observed frequencies.
E.g., among predictions with P=0.8, about 80% should be positive.
"""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score (mean squared error of probabilities).

    Brier = (1/N) * sum((p_i - y_i)^2)
    Range: [0, 1]. Lower is better. 0.25 is random guessing for binary.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_prob: Predicted probabilities for class 1.

    Returns:
        Brier score (float).
    """
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum(|B_m|/N * |acc(B_m) - conf(B_m)|)

    Args:
        y_true: Binary ground truth labels (0/1).
        y_prob: Predicted probabilities for class 1.
        n_bins: Number of bins.

    Returns:
        ECE value (float). Lower is better calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

    return float(ece)


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute calibration curve data.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dict with 'prob_true', 'prob_pred', 'bin_counts'.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            prob_true[i] = np.mean(y_true[in_bin])
            prob_pred[i] = np.mean(y_prob[in_bin])
        else:
            prob_true[i] = 0.0
            prob_pred[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
    }
