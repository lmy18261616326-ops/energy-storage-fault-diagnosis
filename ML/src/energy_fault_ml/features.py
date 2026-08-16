"""Online-feature selection with explicit leakage prevention."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


EXCLUDED_EXACT = {
    "WindowFaultID",
    "RunID",
    "OperatingPointID",
    "WindowID",
    "WindowStart",
    "WindowEnd",
    "RandomSeed",
    "SOCInit",
    "IrefLevel",
    "VbusRefSetting",
    "VbatInit",
    "Rload",
    "Pload",
    "Rbat",
    "Cbus",
    "CbusESR",
    "ScenarioFaultID",
    "ActiveFaultID",
    "ObservableFaultID",
    "FaultName",
    "FaultLocation",
    "FaultMagnitude",
    "FaultParameter1",
    "FaultParameter2",
    "FaultStartTime",
    "FaultEndTime",
    "FaultActiveRatio",
    "FaultObservableRatio",
    "TransitionRatio",
    "IsTransitionWindow",
    "IsTrainingEligible",
    "SampleWeight",
    # The MATLAB feature policy excludes this incomplete residual.
    "PowerResidualMean",
    "PowerResidualRMS",
}

TRUE_SIGNAL_PREFIXES = (
    "il_true",
    "ibat_true",
    "vbus_true",
    "vbat_true",
    "soc_true",
)

STATISTICAL_SIGNAL_PREFIXES = (
    "IL_meas",
    "Ibat_meas",
    "Vbus_meas",
    "Vbat_meas",
    "Iload_meas",
    "SOC_est",
    "Iref",
    "VbusRef",
)


def is_forbidden_feature(name: str) -> bool:
    """Return True for metadata, labels, truth signals, or validation-only data."""

    if name in EXCLUDED_EXACT:
        return True
    lowered = name.lower()
    if lowered.startswith("validation_"):
        return True
    if lowered.startswith(TRUE_SIGNAL_PREFIXES):
        return True
    if "fault" in lowered:
        return True
    return False


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_set: str = "physics_enhanced",
    include_mode_command: bool = True,
) -> list[str]:
    """Select numeric online features without inspecting labels or test statistics."""

    if feature_set not in {"statistical", "physics_enhanced"}:
        raise ValueError(
            "feature_set must be 'statistical' or 'physics_enhanced'."
        )

    selected: list[str] = []
    for name in frame.columns:
        if is_forbidden_feature(name):
            continue
        if name == "ModeCommand" and not include_mode_command:
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        if feature_set == "statistical":
            if name == "ModeCommand" and include_mode_command:
                selected.append(name)
            elif name.startswith(STATISTICAL_SIGNAL_PREFIXES):
                selected.append(name)
        else:
            selected.append(name)

    if not selected:
        raise ValueError(
            f"No usable numeric features remain for feature_set={feature_set!r}."
        )
    return selected


def remove_unusable_training_features(
    train: pd.DataFrame,
    columns: Sequence[str],
    *,
    remove_constant: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Remove all-missing and optionally constant columns using training data only."""

    kept: list[str] = []
    removed: dict[str, str] = {}
    for name in columns:
        numeric = pd.to_numeric(train[name], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            removed[name] = "all_missing_in_training"
        elif remove_constant and finite.nunique(dropna=True) <= 1:
            removed[name] = "constant_in_training"
        else:
            kept.append(name)
    if not kept:
        raise ValueError("All candidate features are missing or constant in training.")
    return kept, removed


def feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Create a numeric matrix while representing infinity as missing data."""

    matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)
