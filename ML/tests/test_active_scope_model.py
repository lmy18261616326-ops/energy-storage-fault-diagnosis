from __future__ import annotations

import numpy as np

from energy_fault_ml.active_scope_model import gate_switch_probability


def test_mode_gate_transfers_inactive_switch_mass_to_healthy() -> None:
    probability = np.array(
        [
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.35, 0.35],
        ]
    )
    gated = gate_switch_probability(probability, np.array([1, 2, 0]))
    np.testing.assert_allclose(gated.sum(axis=1), 1.0)
    assert gated[0, 4] == 0.0
    assert gated[1, 3] == 0.0
    assert gated[2, 3] == 0.0 and gated[2, 4] == 0.0
    assert gated[2, 0] == 0.8
