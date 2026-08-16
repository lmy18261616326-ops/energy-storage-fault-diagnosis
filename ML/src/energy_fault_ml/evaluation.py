"""Classification metrics used consistently across all candidate models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    label_names: Mapping[int, str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Return aggregate metrics, per-class metrics, and a confusion matrix."""

    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    label_list = [int(label) for label in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=label_list,
        zero_division=0,
    )
    confusion = confusion_matrix(truth, prediction, labels=label_list)
    healthy_mask = truth == 0
    healthy_false_alarm = (
        float(np.mean(prediction[healthy_mask] != 0))
        if np.any(healthy_mask)
        else float("nan")
    )

    aggregate = {
        "Accuracy": float(accuracy_score(truth, prediction)),
        "BalancedAccuracy": float(balanced_accuracy_score(truth, prediction)),
        "MacroPrecision": float(
            precision_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "MacroRecall": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
        "MacroF1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "HealthyFalseAlarmRate": healthy_false_alarm,
        "SampleCount": int(len(truth)),
    }
    per_class = pd.DataFrame(
        {
            "ClassID": label_list,
            "ClassName": [label_names.get(label, str(label)) for label in label_list],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )
    return aggregate, per_class, confusion
