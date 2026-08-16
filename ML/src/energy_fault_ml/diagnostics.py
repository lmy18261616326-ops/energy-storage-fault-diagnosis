"""Prediction-level diagnostics for healthy false alarms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA_COLUMNS = (
    "RunID",
    "OperatingPointID",
    "WindowID",
    "WindowStart",
    "WindowEnd",
    "ModeCommand",
    "SOCInit",
    "Pload",
)


def build_prediction_frame(
    source: pd.DataFrame,
    prediction: Sequence[int],
    *,
    model_name: str,
    split_name: str,
    label_column: str,
    label_names: Mapping[int, str],
    fold: int | None = None,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Attach predictions to non-feature metadata for auditable diagnostics."""

    predicted = np.asarray(prediction, dtype=int)
    if len(source) != len(predicted):
        raise ValueError(
            "Prediction length does not match source rows: "
            f"{len(predicted)} != {len(source)}."
        )

    columns = [name for name in metadata_columns if name in source.columns]
    result = source.loc[:, columns].reset_index(drop=True).copy()
    truth = pd.to_numeric(source[label_column], errors="raise").astype(int).to_numpy()
    result.insert(0, "Split", split_name)
    result.insert(0, "Model", model_name)
    if fold is not None:
        result.insert(0, "Fold", int(fold))
    result["TrueClassID"] = truth
    result["TrueClassName"] = [
        label_names.get(int(value), str(int(value))) for value in truth
    ]
    result["PredictedClassID"] = predicted
    result["PredictedClassName"] = [
        label_names.get(int(value), str(int(value))) for value in predicted
    ]
    result["IsHealthyFalseAlarm"] = (truth == 0) & (predicted != 0)
    return result


def healthy_false_alarm_by_operating_point(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize healthy-window false alarms for each operating point."""

    required = {
        "Model",
        "Split",
        "OperatingPointID",
        "TrueClassID",
        "IsHealthyFalseAlarm",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction diagnostics are missing columns: {missing}")

    healthy = predictions.loc[predictions["TrueClassID"].eq(0)].copy()
    group_columns = ["Model", "Split", "OperatingPointID"]
    if "Fold" in healthy.columns:
        group_columns.insert(2, "Fold")
    for dimension in ("ModeCommand", "SOCInit", "Pload"):
        if dimension in healthy.columns:
            group_columns.append(dimension)

    result = (
        healthy.groupby(group_columns, dropna=False, sort=True)
        .agg(
            HealthyWindowCount=("IsHealthyFalseAlarm", "size"),
            FalseAlarmCount=("IsHealthyFalseAlarm", "sum"),
            HealthyFalseAlarmRate=("IsHealthyFalseAlarm", "mean"),
            RunIDCount=("RunID", "nunique")
            if "RunID" in healthy.columns
            else ("IsHealthyFalseAlarm", "size"),
        )
        .reset_index()
    )
    return result.sort_values(
        ["Model", "HealthyFalseAlarmRate", "OperatingPointID"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def healthy_false_alarm_by_dimension(
    predictions: pd.DataFrame,
    *,
    dimensions: Sequence[str] = (
        "OperatingPointID",
        "ModeCommand",
        "SOCInit",
        "Pload",
    ),
) -> pd.DataFrame:
    """Summarize healthy false alarms across actionable operating dimensions."""

    required = {"Model", "Split", "TrueClassID", "IsHealthyFalseAlarm"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction diagnostics are missing columns: {missing}")

    healthy = predictions.loc[predictions["TrueClassID"].eq(0)].copy()
    rows: list[pd.DataFrame] = []
    for dimension in dimensions:
        if dimension not in healthy.columns:
            continue
        group_columns = ["Model", "Split", dimension]
        summary = (
            healthy.groupby(group_columns, dropna=False, sort=True)
            .agg(
                HealthyWindowCount=("IsHealthyFalseAlarm", "size"),
                FalseAlarmCount=("IsHealthyFalseAlarm", "sum"),
                HealthyFalseAlarmRate=("IsHealthyFalseAlarm", "mean"),
                RunIDCount=("RunID", "nunique")
                if "RunID" in healthy.columns
                else ("IsHealthyFalseAlarm", "size"),
            )
            .reset_index()
            .rename(columns={dimension: "Value"})
        )
        summary.insert(2, "Dimension", dimension)
        rows.append(summary)

    columns = [
        "Model",
        "Split",
        "Dimension",
        "Value",
        "HealthyWindowCount",
        "FalseAlarmCount",
        "HealthyFalseAlarmRate",
        "RunIDCount",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(rows, ignore_index=True)
        .loc[:, columns]
        .sort_values(
            ["Model", "Dimension", "HealthyFalseAlarmRate", "Value"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def healthy_false_alarm_destinations(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Show which fault classes receive misclassified healthy windows."""

    required = {
        "Model",
        "Split",
        "OperatingPointID",
        "TrueClassID",
        "PredictedClassID",
        "PredictedClassName",
        "IsHealthyFalseAlarm",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction diagnostics are missing columns: {missing}")

    healthy = predictions.loc[predictions["TrueClassID"].eq(0)].copy()
    false_alarms = healthy.loc[healthy["IsHealthyFalseAlarm"]].copy()
    columns = [
        "Model",
        "Split",
        "OperatingPointID",
        "PredictedClassID",
        "PredictedClassName",
        "FalseAlarmCount",
        "HealthyWindowCount",
        "ShareOfHealthyWindows",
        "ShareOfFalseAlarms",
    ]
    if false_alarms.empty:
        return pd.DataFrame(columns=columns)

    totals = (
        healthy.groupby(["Model", "Split", "OperatingPointID"], dropna=False)
        .size()
        .rename("HealthyWindowCount")
        .reset_index()
    )
    false_alarm_totals = (
        false_alarms.groupby(["Model", "Split", "OperatingPointID"], dropna=False)
        .size()
        .rename("TotalFalseAlarms")
        .reset_index()
    )
    result = (
        false_alarms.groupby(
            [
                "Model",
                "Split",
                "OperatingPointID",
                "PredictedClassID",
                "PredictedClassName",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("FalseAlarmCount")
        .reset_index()
        .merge(totals, on=["Model", "Split", "OperatingPointID"], how="left")
        .merge(
            false_alarm_totals,
            on=["Model", "Split", "OperatingPointID"],
            how="left",
        )
    )
    result["ShareOfHealthyWindows"] = (
        result["FalseAlarmCount"] / result["HealthyWindowCount"]
    )
    result["ShareOfFalseAlarms"] = (
        result["FalseAlarmCount"] / result["TotalFalseAlarms"]
    )
    return (
        result.loc[:, columns]
        .sort_values(
            ["Model", "OperatingPointID", "FalseAlarmCount"],
            ascending=[True, True, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def write_healthy_false_alarm_diagnostics(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> None:
    """Write the standard healthy false-alarm diagnostic tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    healthy_false_alarm_by_operating_point(predictions).to_csv(
        output / f"{prefix}healthy_false_alarm_by_operating_point.csv",
        index=False,
    )
    healthy_false_alarm_by_dimension(predictions).to_csv(
        output / f"{prefix}healthy_false_alarm_by_dimension.csv",
        index=False,
    )
    healthy_false_alarm_destinations(predictions).to_csv(
        output / f"{prefix}healthy_false_alarm_destinations.csv",
        index=False,
    )
