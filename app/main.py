"""Run the full SVR pipeline end-to-end."""
from __future__ import annotations

from pathlib import Path

from app.data import load_california_housing
from app.evaluate import regression_metrics
from app.model import build_svr_pipeline
from app.preprocess import split_data
from app.visualize import plot_predictions


def run_pipeline() -> None:
    """Load data, train an SVR model, evaluate, and visualize predictions."""

    print("Loading California Housing data...")
    X, y = load_california_housing()

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building SVR pipeline with RBF kernel (default)...")
    pipeline = build_svr_pipeline(kernel="rbf")

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)

    print("Evaluating performance...")
    metrics = regression_metrics(y_test, y_pred)
    for name, value in metrics.items():
        print(f"{name.upper():<6}: {value:.4f}")

    print("Saving visualization to artifacts/pred_vs_actual.svg...")
    output_path = Path("artifacts") / "pred_vs_actual.svg"
    plot_predictions(y_test, y_pred, save_path=output_path)
    print(f"Plot saved to {output_path.resolve()}")


if __name__ == "__main__":
    run_pipeline()
