"""Baseline 1 — Logistic Regression.

A simple, interpretable linear baseline for groundwater target prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticBaseline:
    """Logistic Regression baseline with standard preprocessing.

    Pipeline: SimpleImputer (median) → StandardScaler → LogisticRegression

    This is a strong, interpretable baseline. Any complex model should
    demonstrate meaningful improvement over this.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Args:
            C: Inverse regularization strength.
            class_weight: Class weights ('balanced' recommended for imbalanced data).
            max_iter: Maximum iterations for solver.
            random_state: Random seed.
        """
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self._pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        class_weight=class_weight,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_kwargs) -> LogisticBaseline:
        """Fit the logistic regression model.

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
    def coefficients(self) -> dict[str, float]:
        """Return feature coefficients for interpretability."""
        try:
            lr = self._pipeline.named_steps["classifier"]
            feature_names = self._pipeline.named_steps.get("feature_names_in_", None)
            if feature_names is None:
                return {}
            return dict(zip(feature_names, lr.coef_[0], strict=False))
        except (AttributeError, KeyError):
            return {}
