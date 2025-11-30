import pandas as pd

from app.data import load_california_housing


def test_load_california_housing_shapes():
    X, y = load_california_housing()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    # The California Housing dataset has 8 feature columns
    assert X.shape[1] == 8
