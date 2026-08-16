"""Leakage-aware event-level features for fault-diagnosis experiments.

The existing window classifier is useful for low-latency decisions, but its
overlapping 10 ms windows are not independent and weak persistent faults can be
hard to identify in a single window.  This module builds one row per simulation
run from *all* windows in chronological order.  It deliberately avoids
``IsTrainingEligible`` and fault timing fields so the aggregation can be
reproduced online without knowing when a fault starts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_EVENT_SIGNALS: tuple[str, ...] = (
    "IL_measMean",
    "IL_measStd",
    "IL_measRMS",
    "Ibat_measMean",
    "Ibat_measStd",
    "Ibat_measRMS",
    "Vbus_measMean",
    "Vbus_measStd",
    "Vbus_measRMS",
    "Vbat_measMean",
    "Vbat_measStd",
    "Iload_measMean",
    "Iload_measStd",
    "Iload_measRMS",
    "Iload_measDiffRMS",
    "SOC_estMean",
    "SOC_estSlope",
    "IrefMean",
    "VbusRefMean",
    "CurrentErrorMean",
    "CurrentErrorStd",
    "CurrentErrorRMS",
    "VoltageErrorMean",
    "VoltageErrorStd",
    "VoltageErrorRMS",
    "DutyRawMean",
    "DutyRawStd",
    "DutyAppliedMean",
    "DutyAppliedStd",
    "PIiOutMean",
    "PIiOutStd",
    "PIvOutMean",
    "PIvOutStd",
    "Psource_measMean",
    "Psource_measStd",
    "Pload_measMean",
    "Pload_measStd",
    "Pstored_measMean",
    "PowerBalanceResidualMean",
    "PowerBalanceResidualRMS",
    "CurrentPairResidualMean",
    "CurrentPairResidualStd",
    "CurrentPairResidualRMS",
    "CurrentPairResidualMAE",
    "BalancedPowerResidualMean",
    "BalancedPowerResidualStd",
    "BalancedPowerResidualRMS",
    "PbatMean",
    "PbatStd",
    "PbusMean",
    "PbusStd",
    "SatRatio",
    "EnableRatio",
    "S1GateDutyRatio",
    "S2GateDutyRatio",
    "DutyLimitResidualMean",
    "DutyLimitResidualRMS",
    "DutySatRatio",
)


RUN_METADATA: tuple[str, ...] = (
    "RunID",
    "OperatingPointID",
    "WindowID",
    "WindowStart",
    "FaultName",
    "ModeCommand",
    "SOCInit",
    "IrefLevel",
    "VbusRefSetting",
    "Rload",
    "Pload",
    "Rbat",
    "Cbus",
    "CbusESR",
)


CONTEXT_COLUMNS: tuple[str, ...] = (
    "ModeCommand",
    "SOCInit",
    "IrefLevel",
    "VbusRefSetting",
    "Rload",
    "Pload",
    "Rbat",
    "Cbus",
    "CbusESR",
)


def fault_name_to_class(name: str) -> int:
    """Map a simulation scenario name to the five-class research target."""

    lowered = str(name).lower()
    if lowered == "healthy":
        return 0
    if lowered == "vbus_sensor_bias":
        return 1
    if lowered == "inductor_current_sensor_bias":
        return 2
    if lowered.startswith("switch_s1_"):
        return 3
    if lowered.startswith("switch_s2_"):
        return 4
    raise ValueError(f"Unsupported FaultName for event classification: {name!r}")


def fault_mechanism(name: str) -> str:
    """Return a stable mechanism label for scenario-level diagnostics."""

    lowered = str(name).lower()
    if lowered == "healthy":
        return "healthy"
    if lowered == "vbus_sensor_bias":
        return "vbus_sensor_bias"
    if lowered == "inductor_current_sensor_bias":
        return "inductor_current_sensor_bias"
    for mechanism in (
        "full_open",
        "partial_open",
        "intermittent",
        "high_resistance",
    ):
        if mechanism in lowered:
            return mechanism
    return lowered


FINE_LABEL_NAMES: dict[int, str] = {
    0: "healthy",
    1: "vbus_sensor_bias",
    2: "inductor_current_sensor_bias",
    3: "switch_S1_full_open",
    4: "switch_S2_full_open",
    5: "switch_S1_partial_open",
    6: "switch_S2_partial_open",
    7: "switch_S1_intermittent",
    8: "switch_S2_intermittent",
    9: "switch_S1_high_resistance",
    10: "switch_S2_high_resistance",
}


def fault_name_to_fine_class(name: str) -> int:
    """Map every switch mechanism/location pair to a separate train label."""

    lowered = str(name).lower()
    lookup = {value.lower(): key for key, value in FINE_LABEL_NAMES.items()}
    if lowered not in lookup:
        raise ValueError(f"Unsupported FaultName for fine event labels: {name!r}")
    return lookup[lowered]


def fine_probability_to_coarse(
    probability: np.ndarray,
    fine_classes: Sequence[int],
) -> np.ndarray:
    """Collapse eleven fine-label probabilities into the five deployment classes."""

    values = np.asarray(probability, dtype=float)
    classes = np.asarray(fine_classes, dtype=int)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("Probability columns must match fine_classes.")
    result = np.zeros((len(values), 5), dtype=float)
    for column, fine_class in enumerate(classes):
        if fine_class in (0, 1, 2):
            coarse_class = fine_class
        elif fine_class in (3, 5, 7, 9):
            coarse_class = 3
        elif fine_class in (4, 6, 8, 10):
            coarse_class = 4
        else:
            raise ValueError(f"Unexpected fine class: {fine_class}")
        result[:, coarse_class] += values[:, column]
    return result


def required_event_columns(
    available: Iterable[str],
    signals: Sequence[str] = DEFAULT_EVENT_SIGNALS,
) -> list[str]:
    """Select required columns while tolerating optional signal absence."""

    available_set = set(available)
    missing_metadata = sorted(set(RUN_METADATA).difference(available_set))
    if missing_metadata:
        raise ValueError(f"Event dataset is missing metadata: {missing_metadata}")
    selected_signals = [name for name in signals if name in available_set]
    if not selected_signals:
        raise ValueError("None of the requested event signals are present.")
    return list(RUN_METADATA) + selected_signals


def add_physics_normalized_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add online-computable ratios that reduce operating-point dependence."""

    result = frame.copy()

    def safe_denominator(values: pd.Series, floor: float = 1.0) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").abs()
        return numeric.clip(lower=floor)

    if {"VoltageErrorMean", "VbusRefMean"}.issubset(result.columns):
        result["VoltageTrackingRelative"] = (
            pd.to_numeric(result["VoltageErrorMean"], errors="coerce")
            / safe_denominator(result["VbusRefMean"])
        )
    if {"VoltageErrorRMS", "VbusRefMean"}.issubset(result.columns):
        result["VoltageTrackingRMSRelative"] = (
            pd.to_numeric(result["VoltageErrorRMS"], errors="coerce")
            / safe_denominator(result["VbusRefMean"])
        )
    if {"Vbus_measMean", "VbusRefMean"}.issubset(result.columns):
        result["VbusMeasuredRelative"] = (
            pd.to_numeric(result["Vbus_measMean"], errors="coerce")
            / safe_denominator(result["VbusRefMean"])
        )
    if {"CurrentErrorRMS", "IrefMean"}.issubset(result.columns):
        result["CurrentTrackingRMSRelative"] = (
            pd.to_numeric(result["CurrentErrorRMS"], errors="coerce")
            / safe_denominator(result["IrefMean"])
        )
    if {"CurrentPairResidualRMS", "IrefMean"}.issubset(result.columns):
        result["CurrentPairRMSRelative"] = (
            pd.to_numeric(result["CurrentPairResidualRMS"], errors="coerce")
            / safe_denominator(result["IrefMean"])
        )
    if {
        "BalancedPowerResidualRMS",
        "Psource_measMean",
        "Pload_measMean",
    }.issubset(result.columns):
        power_scale = (
            pd.to_numeric(result["Psource_measMean"], errors="coerce").abs()
            + pd.to_numeric(result["Pload_measMean"], errors="coerce").abs()
        ).clip(lower=1.0)
        result["BalancedPowerResidualRelative"] = (
            pd.to_numeric(result["BalancedPowerResidualRMS"], errors="coerce")
            / power_scale
        )
    if {"DutyRawMean", "DutyAppliedMean"}.issubset(result.columns):
        result["DutyApplicationGap"] = (
            pd.to_numeric(result["DutyRawMean"], errors="coerce")
            - pd.to_numeric(result["DutyAppliedMean"], errors="coerce")
        )
    if {"S1GateDutyRatio", "S2GateDutyRatio"}.issubset(result.columns):
        result["GateDutyDifference"] = (
            pd.to_numeric(result["S1GateDutyRatio"], errors="coerce")
            - pd.to_numeric(result["S2GateDutyRatio"], errors="coerce")
        )
    return result


def _constant_per_run(frame: pd.DataFrame, column: str) -> pd.Series:
    counts = frame.groupby("RunID", sort=False)[column].nunique(dropna=False)
    invalid = counts[counts.gt(1)]
    if not invalid.empty:
        raise ValueError(
            f"{column} must be constant within each RunID; invalid="
            f"{invalid.index.astype(str).tolist()[:10]}"
        )
    return frame.groupby("RunID", sort=False)[column].first()


def build_event_dataset(
    windows: pd.DataFrame,
    *,
    include_context: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate chronological windows into one leakage-aware row per RunID."""

    required = {"RunID", "OperatingPointID", "WindowStart", "FaultName"}
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"Event aggregation is missing columns: {missing}")

    ordered = windows.sort_values(
        ["RunID", "WindowStart", "WindowID"], kind="stable"
    ).reset_index(drop=True)
    ordered = add_physics_normalized_features(ordered)

    metadata = pd.DataFrame(index=pd.Index(ordered["RunID"].unique(), name="RunID"))
    for column in ("OperatingPointID", "FaultName", *CONTEXT_COLUMNS):
        if column in ordered.columns:
            metadata[column] = _constant_per_run(ordered, column)
    metadata["WindowFaultID"] = metadata["FaultName"].map(fault_name_to_class)
    metadata["FineFaultID"] = metadata["FaultName"].map(fault_name_to_fine_class)
    metadata["FaultMechanism"] = metadata["FaultName"].map(fault_mechanism)

    excluded = set(RUN_METADATA) | {"WindowFaultID", "FaultMechanism"}
    signal_columns = [
        name
        for name in ordered.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(ordered[name])
    ]
    if not signal_columns:
        raise ValueError("No numeric signals remain for event aggregation.")

    grouped = ordered.groupby("RunID", sort=False)[signal_columns]
    basic = grouped.agg(["mean", "std", "median", "min", "max", "first", "last"])
    basic.columns = [f"{name}__{stat}" for name, stat in basic.columns]
    quantile_10 = grouped.quantile(0.10).add_suffix("__q10")
    quantile_90 = grouped.quantile(0.90).add_suffix("__q90")
    event = metadata.join(basic).join(quantile_10).join(quantile_90)
    delta = pd.DataFrame(
        {
            f"{name}__delta": basic[f"{name}__last"] - basic[f"{name}__first"]
            for name in signal_columns
        },
        index=basic.index,
    )
    event = event.join(delta)

    feature_columns = [
        name
        for name in event.columns
        if "__" in name and pd.api.types.is_numeric_dtype(event[name])
    ]
    if include_context:
        for name in CONTEXT_COLUMNS:
            if name in event.columns and pd.api.types.is_numeric_dtype(event[name]):
                feature_columns.append(name)

    # ``join`` above creates many internal pandas blocks.  Consolidate before
    # reset_index so large event tables do not emit fragmentation warnings.
    event = event.copy().reset_index()
    if event["RunID"].duplicated().any():
        raise ValueError("Event aggregation produced duplicate RunID rows.")
    return event, feature_columns
