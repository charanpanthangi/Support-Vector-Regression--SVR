"""Preprocessing helpers for splitting and scaling data."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and targets into train and test sets.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector.
    test_size:
        Fraction of the data to reserve for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Tuple containing ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def create_scaler() -> StandardScaler:
    """Return a ``StandardScaler`` with default settings."""

    return StandardScaler()


__all__ = ["split_data", "create_scaler"]
