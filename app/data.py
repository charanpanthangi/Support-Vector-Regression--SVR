"""Data loading utilities for the SVR template."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    The dataset predicts median house values using eight numerical features.

    Parameters
    ----------
    as_frame:
        Whether to return the data as pandas objects. Defaults to ``True`` for
        easy inspection and plotting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` and target vector ``y``.
    """

    dataset = fetch_california_housing(as_frame=as_frame)
    X = dataset.data
    y = dataset.target
    return X, y


__all__ = ["load_california_housing"]
