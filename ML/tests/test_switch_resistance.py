from __future__ import annotations

import numpy as np

from energy_fault_ml.switch_resistance import SwitchResistanceSpecialist


def test_switch_resistance_predicts_active_fault_ids() -> None:
    model = SwitchResistanceSpecialist(threshold_ohm=0.0105)
    frame = {
        "ModeCommand": np.array([1, 1, 2, 2]),
        "S1_ron_estimateMedian": np.array([0.001, 0.05, 0.001, 0.001]),
        "S2_ron_estimateMedian": np.array([0.001, 0.001, 0.001, 0.05]),
    }
    np.testing.assert_array_equal(model.predict(frame), [0, 3, 0, 4])


def test_switch_resistance_marks_inactive_mode_unsupported() -> None:
    model = SwitchResistanceSpecialist()
    frame = {
        "ModeCommand": np.array([0, 3]),
        "S1_ron_estimateMedian": np.array([0.05, 0.05]),
        "S2_ron_estimateMedian": np.array([0.05, 0.05]),
    }
    np.testing.assert_array_equal(model.predict(frame), [0, 0])
    assert set(model.qualification_status(frame)) == {"unsupported_wait_excitation"}


def test_switch_resistance_rejects_misaligned_columns() -> None:
    model = SwitchResistanceSpecialist()
    frame = {
        "ModeCommand": np.array([1, 2]),
        "S1_ron_estimateMedian": np.array([0.001]),
        "S2_ron_estimateMedian": np.array([0.001, 0.001]),
    }
    try:
        model.predict(frame)
    except ValueError as error:
        assert "must align" in str(error)
    else:
        raise AssertionError("Expected misaligned feature columns to fail")
