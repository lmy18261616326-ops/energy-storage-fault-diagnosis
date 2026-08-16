from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_reference_hdf5_dataset import (  # noqa: E402
    CONTROLLER_RAW,
    DEVICE_RAW,
    EXCLUDED_TRUTH_CHANNELS,
    derived_channels,
    exact_interval,
    window_series_features,
)


def test_window_series_features_has_stable_shape_and_delta() -> None:
    values = np.arange(2000, dtype=float)
    result = window_series_features(
        {"signal": values},
        len(values),
        window_samples=1000,
        step_samples=500,
    )
    assert len(result) == 50
    assert result["signal__mean__delta"] == 1000.0
    assert np.isfinite(np.fromiter(result.values(), dtype=float)).all()


def test_device_features_derive_ron_without_truth_channels() -> None:
    samples = 20
    signals = {name: np.ones(samples, dtype=float) for name in CONTROLLER_RAW + DEVICE_RAW}
    signals["s1_device_voltage_V"][:] = 0.02
    signals["s1_device_current_A"][:] = 2.0
    signals["s2_device_voltage_V"][:] = 0.03
    signals["s2_device_current_A"][:] = 3.0
    result = derived_channels(signals, include_devices=True)
    assert np.allclose(result["s1_ron_estimate_ohm"], 0.01)
    assert np.allclose(result["s2_ron_estimate_ohm"], 0.01)
    assert not set(EXCLUDED_TRUTH_CHANNELS).intersection(result)


def test_exact_interval_handles_zero_and_perfect_counts() -> None:
    zero = exact_interval(0, 36)
    perfect = exact_interval(162, 162)
    assert zero[0] == 0.0 and 0.0 < zero[1] < 0.11
    assert 0.97 < perfect[0] < 1.0 and perfect[1] == 1.0
