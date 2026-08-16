from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_ieee_computer_evidence.py"
SPEC = importlib.util.spec_from_file_location("build_ieee_computer_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_interval_for_all_successes_has_finite_lower_bound() -> None:
    low, high = MODULE.clopper_pearson(16, 16)
    assert 0.79 < low < 0.80
    assert high == 1.0


def test_first_consecutive_true_returns_start_index() -> None:
    flags = np.array([False, True, False, True, True, True])
    assert MODULE.first_consecutive_true(flags, required=2) == 3
    assert MODULE.first_consecutive_true(flags, required=4) is None


def test_event_windows_recover_constant_resistance() -> None:
    time_s = np.arange(0, 0.050, 50e-6)
    current_a = np.full_like(time_s, 10.0)
    resistance = np.full_like(time_s, 0.012)
    event_time, event_resistance = MODULE.event_windows(
        time_s, resistance, current_a
    )
    assert len(event_time) > 0
    assert np.allclose(event_resistance, 0.012)


def test_quantize_is_identity_when_disabled() -> None:
    values = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(MODULE.quantize(values, 0), values)
