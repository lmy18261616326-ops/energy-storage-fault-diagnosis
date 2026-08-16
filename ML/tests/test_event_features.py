from __future__ import annotations

import pandas as pd
import pytest

from energy_fault_ml.event_features import (
    add_physics_normalized_features,
    build_event_dataset,
    fault_name_to_fine_class,
    fault_name_to_class,
    fine_probability_to_coarse,
)


def test_fault_name_to_class_maps_switch_mechanisms_by_location() -> None:
    assert fault_name_to_class("healthy") == 0
    assert fault_name_to_class("vbus_sensor_bias") == 1
    assert fault_name_to_class("inductor_current_sensor_bias") == 2
    assert fault_name_to_class("switch_S1_high_resistance") == 3
    assert fault_name_to_class("switch_S2_partial_open") == 4
    with pytest.raises(ValueError):
        fault_name_to_class("unknown")


def test_fine_labels_collapse_to_coarse_probabilities() -> None:
    assert fault_name_to_fine_class("switch_S1_high_resistance") == 9
    assert fault_name_to_fine_class("switch_S2_high_resistance") == 10
    probability = fine_probability_to_coarse(
        probability=[[0.1, 0.1, 0.1, 0.2, 0.3, 0.05, 0.05, 0.02, 0.03, 0.02, 0.03]],
        fine_classes=list(range(11)),
    )
    assert probability.shape == (1, 5)
    assert probability[0].sum() == pytest.approx(1.0)
    assert probability[0, 3] == pytest.approx(0.29)
    assert probability[0, 4] == pytest.approx(0.41)


def test_physics_ratios_are_finite_with_zero_reference() -> None:
    frame = pd.DataFrame(
        {
            "VoltageErrorMean": [2.0],
            "VoltageErrorRMS": [3.0],
            "VbusRefMean": [0.0],
            "Vbus_measMean": [4.0],
            "CurrentErrorRMS": [5.0],
            "IrefMean": [0.0],
            "CurrentPairResidualRMS": [6.0],
        }
    )
    result = add_physics_normalized_features(frame)
    assert result.loc[0, "VoltageTrackingRelative"] == 2.0
    assert result.loc[0, "CurrentPairRMSRelative"] == 6.0


def test_event_dataset_uses_all_windows_and_produces_unique_runs() -> None:
    rows = []
    for run, fault in (("r1", "healthy"), ("r2", "switch_S1_full_open")):
        for window in range(3):
            rows.append(
                {
                    "RunID": run,
                    "OperatingPointID": "op1" if run == "r1" else "op2",
                    "WindowID": window + 1,
                    "WindowStart": 0.01 * window,
                    "FaultName": fault,
                    "ModeCommand": 1,
                    "SOCInit": 50,
                    "IrefLevel": 10,
                    "VbusRefSetting": 400,
                    "Rload": 200,
                    "Pload": 400,
                    "Rbat": 0.5,
                    "Cbus": 0.002,
                    "CbusESR": 0.001,
                    "Vbus_measMean": 400.0 - window,
                    "VbusRefMean": 400.0,
                    "VoltageErrorMean": float(window),
                }
            )
    event, features = build_event_dataset(pd.DataFrame(rows))
    assert len(event) == 2
    assert event["RunID"].is_unique
    assert set(event["WindowFaultID"]) == {0, 3}
    assert set(event["FineFaultID"]) == {0, 3}
    assert "VoltageErrorMean__delta" in features
    assert event.loc[event["RunID"].eq("r2"), "VoltageErrorMean__delta"].iloc[0] == 2.0
