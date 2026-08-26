"""Baseline 0 — Simple rule-based / statistical baseline.

The simplest possible model: predicts based on mean target value in
training data, or applies simple threshold rules on geophysical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RuleBaseline:
    """Simple rule-based baseline.

    Implements a constant prediction (mean of training targets) as the
    simplest possible baseline. Any model that does worse than this
    should be questioned.

    Also supports a basic geophysical threshold rule:
    - Low resistivity + high Vp contrast suggests saturated zone
    - Returns a score rather than hard classification.
    """

    def __init__(self, strategy: str = "mean"):
        """
        Args:
            strategy: Baseline strategy:
                'mean' — predict the mean training target.
                'threshold' — use geophysical thresholds.
        """
        self.strategy = strategy
        self._mean: float = 0.0
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleBaseline:
        """Fit the baseline (just stores the mean for 'mean' strategy).

        Args:
            X: Feature DataFrame (unused for 'mean' strategy).
            y: Target values.

        Returns:
            Self.
        """
        self._mean = float(np.mean(y))
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted scores.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.strategy == "mean":
            return np.full(len(X), self._mean)

        # Threshold-based strategy
        scores = np.full(len(X), 0.5)  # default

        # Low resistivity suggests saturated zone
        if "ert_resistivity_mean" in X.columns:
            resistivity = X["ert_resistivity_mean"].values
            # Resistivity < 100 ohm·m often indicates water
            low_r = (resistivity < 100).astype(float)
            # Resistivity < 30 ohm·m strongly suggests water
            very_low_r = (resistivity < 30).astype(float)
            scores = 0.3 + 0.3 * low_r + 0.2 * very_low_r

        if "seismic_vp_mean" in X.columns:
            vp = X["seismic_vp_mean"].values
            # Vp 1500-2500 m/s in shallow layers suggests saturated
            vp_signal = ((vp > 1500) & (vp < 2500)).astype(float)
            scores = np.clip(scores + 0.15 * vp_signal, 0.0, 1.0)

        return np.clip(scores, 0.0, 1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return scores as pseudo-probabilities (N, 2).

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape (N, 2) with [P(negative), P(positive)].
        """
        scores = self.predict(X)
        return np.column_stack([1.0 - scores, scores])
