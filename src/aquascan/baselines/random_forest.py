"""Baseline 2 — Random Forest.

A non-linear baseline that can capture feature interactions without
the complexity of deep learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RandomForestBaseline:
    """Random Forest baseline.

    Pipeline: SimpleImputer (median) → StandardScaler → RandomForest

    Random Forest serves as a strong non-linear baseline. It handles
    heterogeneous features well and provides feature importance estimates.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth (None = unlimited).
            class_weight: Class weights.
            random_state: Random seed.
            n_jobs: Number of parallel jobs (-1 = all cores).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_kwargs) -> RandomForestBaseline:
        """Fit the random forest model.

        Args:
            X: Feature DataFrame.
            y: Binary target labels (0/1).
            **fit_kwargs: Additional kwargs passed to sklearn fit.

        Returns:
            Self.
        """
        self._pipeline.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted class labels.
        """
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape (N, 2) with [P(negative), P(positive)].
        """
        return self._pipeline.predict_proba(X)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probability as a water score.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of water scores (0-1).
        """
        return self.predict_proba(X)[:, 1]

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return feature importance values."""
        try:
            rf = self._pipeline.named_steps["classifier"]
            feature_names = getattr(rf, "feature_names_in_", None)
            if feature_names is None:
                return {}
            return dict(zip(feature_names, rf.feature_importances_, strict=False))
        except (AttributeError, KeyError):
            return {}
