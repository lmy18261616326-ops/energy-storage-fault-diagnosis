from __future__ import annotations

import numpy as np
import pandas as pd

from energy_fault_ml.calibration import (
    CalibratedAlarmArtifact,
    temperature_scale,
)


class _ProbabilityStub:
    name = "stub"
    feature_columns = ["x"]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(features["x"].tolist(), dtype=float)


def test_temperature_scale_is_normalized() -> None:
    probability = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    calibrated = temperature_scale(probability, 1.7)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    assert np.all(calibrated > 0)


def test_alarm_threshold_can_hold_low_confidence_fault_as_healthy() -> None:
    frame = pd.DataFrame(
        {
            "x": [
                [0.80, 0.10, 0.10],
                [0.20, 0.70, 0.10],
            ]
        }
    )
    artifact = CalibratedAlarmArtifact(
        base_artifact=_ProbabilityStub(),  # type: ignore[arg-type]
        temperature=1.0,
        alarm_threshold=0.50,
    )
    np.testing.assert_array_equal(artifact.predict(frame), [0, 1])
