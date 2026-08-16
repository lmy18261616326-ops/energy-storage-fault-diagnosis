"""Deployment wrapper for the physically observable event-classification scope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def align_probability(
    probability: np.ndarray, classes: np.ndarray, class_count: int = 5
) -> np.ndarray:
    aligned = np.zeros((len(probability), class_count), dtype=float)
    for column, label in enumerate(np.asarray(classes, dtype=int)):
        aligned[:, label] = probability[:, column]
    return aligned


def gate_switch_probability(
    probability: np.ndarray, mode_command: np.ndarray | pd.Series
) -> np.ndarray:
    """Transfer impossible switch-location mass to healthy/abstain."""

    gated = np.asarray(probability, dtype=float).copy()
    modes = np.asarray(mode_command, dtype=int)
    if gated.ndim != 2 or gated.shape[1] != 5 or len(gated) != len(modes):
        raise ValueError("Expected probability shape (n, 5) and one mode per row")
    invalid_s1 = modes != 1
    invalid_s2 = modes != 2
    gated[invalid_s1, 0] += gated[invalid_s1, 3]
    gated[invalid_s2, 0] += gated[invalid_s2, 4]
    gated[invalid_s1, 3] = 0.0
    gated[invalid_s2, 4] = 0.0
    row_sum = gated.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Invalid zero probability after mode gating")
    return gated / row_sum


@dataclass
class ActiveScopeFaultModel:
    """Primary classifier plus a nonlinear verifier and scope status output."""

    primary_model: object
    verifier_model: object
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...] = (
        "healthy",
        "vbus_sensor_bias",
        "inductor_current_sensor_bias",
        "switch_S1_fault",
        "switch_S2_fault",
    )

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_names).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing model features: {missing[:10]}")
        return frame.loc[:, self.feature_names]

    @staticmethod
    def _probability(model: object, values: pd.DataFrame) -> np.ndarray:
        raw = model.predict_proba(values)
        estimator = model.named_steps["model"]
        return align_probability(raw, estimator.classes_)

    def predict_proba(
        self, frame: pd.DataFrame, mode_command: np.ndarray | pd.Series
    ) -> np.ndarray:
        values = self._features(frame)
        probability = self._probability(self.primary_model, values)
        return gate_switch_probability(probability, mode_command)

    def predict_with_status(
        self, frame: pd.DataFrame, mode_command: np.ndarray | pd.Series
    ) -> pd.DataFrame:
        values = self._features(frame)
        modes = np.asarray(mode_command, dtype=int)
        primary_probability = gate_switch_probability(
            self._probability(self.primary_model, values), modes
        )
        verifier_probability = gate_switch_probability(
            self._probability(self.verifier_model, values), modes
        )
        primary = np.argmax(primary_probability, axis=1)
        verifier = np.argmax(verifier_probability, axis=1)
        scope = np.select(
            [modes == 1, modes == 2],
            ["sensor_and_S1_observable", "sensor_and_S2_observable"],
            default="sensor_only_wait_switch_excitation",
        )
        return pd.DataFrame(
            {
                "PredictedClassID": primary,
                "PredictedClassName": [self.label_names[label] for label in primary],
                "Confidence": primary_probability.max(axis=1),
                "VerifierClassID": verifier,
                "ModelsAgree": primary == verifier,
                "ObservabilityStatus": scope,
                "HighResistanceSupported": False,
            },
            index=frame.index,
        )
