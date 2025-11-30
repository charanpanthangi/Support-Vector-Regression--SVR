"""Evaluation helpers for regression metrics."""
from __future__ import annotations

from math import sqrt
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute common regression metrics for SVR predictions.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted target values from the model.

    Returns
    -------
    dict
        Dictionary containing MSE, MAE, RMSE, and R^2.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


__all__ = ["regression_metrics"]
