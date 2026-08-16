"""Non-interactive plots for benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_confusion_matrix(
    confusion: np.ndarray,
    *,
    labels: Sequence[int],
    label_names: Mapping[int, str],
    title: str,
    output_path: str | Path,
) -> None:
    """Save an English-labelled confusion matrix without font assumptions."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tick_labels = [f"{label}: {label_names.get(int(label), label)}" for label in labels]
    figure, axis = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
