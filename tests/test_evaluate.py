import numpy as np

from app.evaluate import regression_metrics


def test_regression_metrics_outputs_expected_keys():
    y_true = np.array([3.0, 2.5, 4.0])
    y_pred = np.array([2.8, 2.4, 4.2])

    metrics = regression_metrics(y_true, y_pred)

    expected_keys = {"mse", "mae", "rmse", "r2"}
    assert set(metrics.keys()) == expected_keys
    # Basic sanity: errors should be small for these close predictions
    assert metrics["mse"] < 0.05
    assert metrics["mae"] < 0.3
