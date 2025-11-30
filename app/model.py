"""Model creation utilities for Support Vector Regression."""
from __future__ import annotations

from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

Kernel = Literal["linear", "poly", "rbf"]


def build_svr_pipeline(
    kernel: Kernel = "rbf",
    C: float = 1.0,
    epsilon: float = 0.1,
    degree: int = 3,
    gamma: str | float = "scale",
) -> Pipeline:
    """
    Create a scikit-learn ``Pipeline`` with scaling and an SVR estimator.

    Parameters
    ----------
    kernel:
        Kernel type to be used in the algorithm (`"linear"`, `"poly"`, `"rbf"`).
    C:
        Regularization strength. Larger values try to fit the training data more closely.
    epsilon:
        Width of the epsilon-insensitive tube around the regression line.
    degree:
        Degree of the polynomial kernel (ignored by linear/RBF).
    gamma:
        Kernel coefficient for RBF and polynomial kernels. ``"scale"`` works well by default.

    Returns
    -------
    Pipeline
        A ready-to-train pipeline that scales inputs then fits an SVR model.
    """

    svr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", svr),
    ])
    return pipeline


__all__ = ["build_svr_pipeline"]
