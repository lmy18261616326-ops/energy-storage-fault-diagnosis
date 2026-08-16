"""Inference helper for the v06 direct switch-resistance specialist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SwitchResistanceSpecialist:
    """Classify active S1/S2 high resistance from window-level V/I features.

    Fault IDs follow the project dataset: 0=healthy, 3=S1 high resistance,
    4=S2 high resistance.  Only Mode 1/S1 and Mode 2/S2 are qualified.
    """

    threshold_ohm: float = 0.0105
    s1_feature: str = "S1_ron_estimateMedian"
    s2_feature: str = "S2_ron_estimateMedian"
    mode_feature: str = "ModeCommand"
    required_model: str = "main_model_fd_v06_switchobservability"

    def active_resistance(self, frame: Mapping[str, object]) -> np.ndarray:
        mode = np.asarray(frame[self.mode_feature], dtype=float)
        s1 = np.asarray(frame[self.s1_feature], dtype=float)
        s2 = np.asarray(frame[self.s2_feature], dtype=float)
        if mode.shape != s1.shape or mode.shape != s2.shape:
            raise ValueError("ModeCommand and switch-resistance columns must align")
        return np.where(mode == 1, s1, np.where(mode == 2, s2, np.nan))

    def qualification_status(self, frame: Mapping[str, object]) -> np.ndarray:
        mode = np.asarray(frame[self.mode_feature], dtype=float)
        resistance = self.active_resistance(frame)
        supported = np.isin(mode, (1, 2)) & np.isfinite(resistance)
        return np.where(supported, "qualified_active_mode", "unsupported_wait_excitation")

    def predict_binary(self, frame: Mapping[str, object]) -> np.ndarray:
        resistance = self.active_resistance(frame)
        supported = self.qualification_status(frame) == "qualified_active_mode"
        return (supported & (resistance >= self.threshold_ohm)).astype(int)

    def predict(self, frame: Mapping[str, object]) -> np.ndarray:
        mode = np.asarray(frame[self.mode_feature], dtype=float)
        detected = self.predict_binary(frame).astype(bool)
        result = np.zeros(mode.shape, dtype=int)
        result[detected & (mode == 1)] = 3
        result[detected & (mode == 2)] = 4
        return result
