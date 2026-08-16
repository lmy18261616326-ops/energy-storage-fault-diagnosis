from __future__ import annotations

import numpy as np

from energy_fault_ml.diagnostics import (
    build_prediction_frame,
    healthy_false_alarm_by_dimension,
    healthy_false_alarm_by_operating_point,
    healthy_false_alarm_destinations,
)
from tests.make_synthetic_dataset import make_synthetic_frame


def test_healthy_false_alarm_diagnostics_reconcile_counts() -> None:
    frame = make_synthetic_frame(group_count=3, windows_per_class=2)
    truth = frame["WindowFaultID"].to_numpy(dtype=int)
    prediction = truth.copy()
    healthy_indices = np.flatnonzero(truth == 0)
    prediction[healthy_indices[:2]] = 1
    prediction[healthy_indices[2:3]] = 3

    predictions = build_prediction_frame(
        frame,
        prediction,
        model_name="test_model",
        split_name="test",
        label_column="WindowFaultID",
        label_names={
            0: "healthy",
            1: "vbus_sensor_bias",
            2: "inductor_current_sensor_bias",
            3: "switch_S1_open",
            4: "switch_S2_open",
        },
    )
    by_operating_point = healthy_false_alarm_by_operating_point(predictions)
    by_dimension = healthy_false_alarm_by_dimension(predictions)
    destinations = healthy_false_alarm_destinations(predictions)

    assert int(by_operating_point["FalseAlarmCount"].sum()) == 3
    operating_point_rows = by_dimension.loc[
        by_dimension["Dimension"].eq("OperatingPointID")
    ]
    assert int(operating_point_rows["FalseAlarmCount"].sum()) == 3
    assert int(destinations["FalseAlarmCount"].sum()) == 3
    assert set(destinations["PredictedClassID"]) == {1, 3}
