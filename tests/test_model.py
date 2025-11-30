import numpy as np

from app.data import load_california_housing
from app.model import build_svr_pipeline
from app.preprocess import split_data


def test_svr_pipeline_trains_and_predicts():
    X, y = load_california_housing()
    # Use a small subset for a fast test
    X_sample = X.head(200)
    y_sample = y.head(200)

    X_train, X_test, y_train, y_test = split_data(X_sample, y_sample, test_size=0.25, random_state=0)
    pipeline = build_svr_pipeline(kernel="rbf")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y_test)
