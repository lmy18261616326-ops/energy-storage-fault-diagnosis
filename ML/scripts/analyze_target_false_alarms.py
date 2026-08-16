#!/usr/bin/env python
"""Analyze target operating-point false alarms without leaking test statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output"
    / "combined"
    / "feature_dataset.csv"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT / "ML" / "results" / "group_cv_analysis" / "group_cv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "ML" / "results" / "target_false_alarm_analysis"
TARGET_OPERATING_POINTS = ("op_0002", "op_0006", "op_0009")
KEY_COLUMNS = ("RunID", "OperatingPointID", "WindowID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def robust_training_scale(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    center = numeric.median(axis=0)
    scale = numeric.quantile(0.75) - numeric.quantile(0.25)
    fallback = numeric.std(axis=0, ddof=0)
    scale = scale.mask(~np.isfinite(scale) | scale.le(1e-12), fallback)
    scale = scale.mask(~np.isfinite(scale) | scale.le(1e-12), 1.0)
    return center, scale


def nearest_pairs(
    healthy: pd.DataFrame,
    fault: pd.DataFrame,
    feature_columns: list[str],
    center: pd.Series,
    scale: pd.Series,
) -> pd.DataFrame:
    h = (
        healthy.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(center)
    )
    f = (
        fault.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(center)
    )
    hz = ((h - center) / scale).to_numpy(dtype=np.float32)
    fz = ((f - center) / scale).to_numpy(dtype=np.float32)

    rows: list[dict[str, object]] = []
    chunk_size = 64
    for start in range(0, len(hz), chunk_size):
        stop = min(start + chunk_size, len(hz))
        distance = np.mean(
            np.square(hz[start:stop, None, :] - fz[None, :, :]),
            axis=2,
        )
        nearest_index = np.argmin(distance, axis=1)
        for local_index, fault_index in enumerate(nearest_index):
            h_index = start + local_index
            rows.append(
                {
                    "HealthyRunID": healthy.iloc[h_index]["RunID"],
                    "HealthyWindowID": int(healthy.iloc[h_index]["WindowID"]),
                    "HealthyWindowStart": float(
                        healthy.iloc[h_index]["WindowStart"]
                    ),
                    "FaultRunID": fault.iloc[fault_index]["RunID"],
                    "FaultWindowID": int(fault.iloc[fault_index]["WindowID"]),
                    "FaultWindowStart": float(
                        fault.iloc[fault_index]["WindowStart"]
                    ),
                    "MeanSquaredRobustDistance": float(
                        distance[local_index, fault_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def similarity_table(
    healthy: pd.DataFrame,
    fault: pd.DataFrame,
    training: pd.DataFrame,
    destination: int,
    feature_columns: list[str],
    scale: pd.Series,
) -> pd.DataFrame:
    h = healthy.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    f = fault.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    mean_difference = (h.mean(axis=0) - f.mean(axis=0)).abs()
    median_difference = (h.median(axis=0) - f.median(axis=0)).abs()
    training_healthy = (
        training.loc[training["WindowFaultID"].eq(0), feature_columns]
        .apply(pd.to_numeric, errors="coerce")
    )
    training_fault = (
        training.loc[
            training["WindowFaultID"].eq(destination),
            feature_columns,
        ]
        .apply(pd.to_numeric, errors="coerce")
    )
    training_separation = (
        training_healthy.mean(axis=0) - training_fault.mean(axis=0)
    ).abs() / scale
    result = pd.DataFrame(
        {
            "Feature": feature_columns,
            "HealthyMean": h.mean(axis=0).to_numpy(),
            "FaultMean": f.mean(axis=0).to_numpy(),
            "HealthyMedian": h.median(axis=0).to_numpy(),
            "FaultMedian": f.median(axis=0).to_numpy(),
            "RobustMeanDifference": (mean_difference / scale).to_numpy(),
            "RobustMedianDifference": (median_difference / scale).to_numpy(),
            "TrainingClassSeparation": training_separation.to_numpy(),
        }
    )
    result["SimilarityScore"] = (
        result["RobustMeanDifference"] + result["RobustMedianDifference"]
    ) / 2
    # High values identify a feature that separates the two classes in the
    # training operating points but becomes nearly identical at this test point.
    result["ConfusionRiskScore"] = result["TrainingClassSeparation"] / (
        result["SimilarityScore"] + 0.05
    )
    return result.sort_values(
        ["SimilarityScore", "Feature"],
        kind="stable",
    ).reset_index(drop=True)


def save_similarity_plot(
    table: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    selected = table.loc[
        table["TrainingClassSeparation"].ge(0.10)
    ].nlargest(20, "ConfusionRiskScore")
    selected = selected.sort_values("ConfusionRiskScore", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        selected["Feature"],
        selected["ConfusionRiskScore"],
        color="#C55A11",
    )
    axis.set_xlabel("Confusion-risk score (larger = more suspicious)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.data)
    eligible = pd.to_numeric(
        features["IsTrainingEligible"], errors="coerce"
    ).fillna(0)
    features = features.loc[eligible.ne(0)].copy()
    predictions = pd.read_csv(args.results / "window_predictions.csv")
    assignments = pd.read_csv(args.results / "fold_group_assignments.csv")
    feature_columns = [
        line.strip()
        for line in (args.results.parent / "feature_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    prediction_targets = predictions.loc[
        predictions["OperatingPointID"].isin(TARGET_OPERATING_POINTS)
        & predictions["Model"].isin(["random_forest", "xgboost"])
        & predictions["TrueClassID"].eq(0)
    ].copy()
    summary = (
        prediction_targets.groupby(["Model", "OperatingPointID"], sort=True)
        .agg(
            HealthyWindows=("TrueClassID", "size"),
            FalseAlarms=("IsHealthyFalseAlarm", "sum"),
            HealthyRunIDs=("RunID", "nunique"),
        )
        .reset_index()
    )
    summary["FalseAlarmRate"] = (
        summary["FalseAlarms"] / summary["HealthyWindows"]
    )

    all_similarity: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    waveform_manifest: list[dict[str, object]] = []
    for (model_name, operating_point), part in prediction_targets.groupby(
        ["Model", "OperatingPointID"],
        sort=True,
    ):
        false_alarm_predictions = part.loc[
            part["IsHealthyFalseAlarm"].astype(bool)
        ].copy()
        if false_alarm_predictions.empty:
            continue
        destination = int(
            false_alarm_predictions["PredictedClassID"].mode().iloc[0]
        )
        fold = int(false_alarm_predictions["Fold"].iloc[0])
        train_groups = assignments.loc[
            assignments["Fold"].eq(fold) & assignments["Role"].eq("train"),
            "OperatingPointID",
        ].astype(str)
        training = features.loc[
            features["OperatingPointID"].astype(str).isin(train_groups)
        ]
        center, scale = robust_training_scale(training, feature_columns)

        false_alarm_keys = false_alarm_predictions.loc[:, KEY_COLUMNS]
        healthy = false_alarm_keys.merge(
            features,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        fault = features.loc[
            features["OperatingPointID"].eq(operating_point)
            & features["WindowFaultID"].eq(destination)
        ].copy()
        if fault.empty:
            continue

        similarity = similarity_table(
            healthy,
            fault,
            training,
            destination,
            feature_columns,
            scale,
        )
        similarity.insert(0, "DestinationClassID", destination)
        similarity.insert(0, "OperatingPointID", operating_point)
        similarity.insert(0, "Model", model_name)
        all_similarity.append(similarity)

        pairs = nearest_pairs(
            healthy,
            fault,
            feature_columns,
            center,
            scale,
        )
        pairs.insert(0, "DestinationClassID", destination)
        pairs.insert(0, "OperatingPointID", operating_point)
        pairs.insert(0, "Model", model_name)
        all_pairs.append(pairs)
        representative = pairs.sort_values(
            "MeanSquaredRobustDistance",
            kind="stable",
        ).iloc[0]
        waveform_manifest.append(
            {
                "Model": model_name,
                "OperatingPointID": operating_point,
                "DestinationClassID": destination,
                **representative.to_dict(),
            }
        )
        save_similarity_plot(
            similarity,
            args.output
            / f"{model_name}_{operating_point}_similar_features.png",
            f"{model_name}: {operating_point} healthy vs class {destination}",
        )

    summary.to_csv(args.output / "target_false_alarm_summary.csv", index=False)
    pd.concat(all_similarity, ignore_index=True).to_csv(
        args.output / "feature_similarity.csv",
        index=False,
    )
    pd.concat(all_pairs, ignore_index=True).to_csv(
        args.output / "nearest_window_pairs.csv",
        index=False,
    )
    pd.DataFrame(waveform_manifest).to_csv(
        args.output / "waveform_manifest.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
