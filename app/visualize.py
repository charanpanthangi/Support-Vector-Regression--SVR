"""Visualization helpers for SVR predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str | Path] = "pred_vs_actual.svg",
) -> Path:
    """
    Plot predicted vs. actual target values and save to disk.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values from the model.
    save_path:
        Location to save the SVG plot.

    Returns
    -------
    Path
        Path to the saved plot file.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    save_path = Path(save_path)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("SVR Predictions vs. Actual Values")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="svg")
    plt.close()
    return save_path


__all__ = ["plot_predictions"]
