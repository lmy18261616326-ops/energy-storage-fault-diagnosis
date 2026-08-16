"""Probability calibration and alarm-threshold wrapper for saved models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import ModelArtifact


def temperature_scale(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply multiclass temperature scaling to probability estimates."""

    logits = np.log(np.clip(probability, 1e-12, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


@dataclass
class CalibratedAlarmArtifact:
    """Final estimator with global temperature and fault-alarm threshold."""

    base_artifact: ModelArtifact
    temperature: float
    alarm_threshold: float

    @property
    def feature_columns(self) -> list[str]:
        return self.base_artifact.feature_columns

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.base_artifact.predict_proba(features)
        if raw is None:
            raise RuntimeError("Base estimator does not provide probabilities.")
        return temperature_scale(raw, self.temperature)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(features)
        fault_class = np.argmax(probability[:, 1:], axis=1) + 1
        alarm_probability = 1.0 - probability[:, 0]
        return np.where(
            alarm_probability >= self.alarm_threshold,
            fault_class,
            0,
        )
