#!/usr/bin/env python
"""Test whether historical target false alarms reproduce on new healthy runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from energy_fault_ml.data import constrained_group_kfolds, load_feature_dataset
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model
from run_feature_study import model_parameters, rank_features


OLD_DATA = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output"
    / "combined"
    / "feature_dataset.csv"
)
NEW_HEALTH = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "phase1_target_health"
    / "combined"
    / "feature_dataset.csv"
)
DEFAULT_OUTPUT = ML_ROOT / "results" / "new_healthy_reproduction"
TARGETS = ("op_0002", "op_0006", "op_0009")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", type=Path, default=OLD_DATA)
    parser.add_argument("--new-health", type=Path, default=NEW_HEALTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=240727)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = load_feature_dataset(args.old_data)
    new = pd.read_csv(args.new_health)
    eligible = pd.to_numeric(
        new["IsTrainingEligible"], errors="coerce"
    ).fillna(0)
    new = new.loc[eligible.ne(0)].copy()
    if set(new["WindowFaultID"].astype(int).unique()) != {0}:
        raise ValueError("The reproduction dataset must contain healthy only.")
    folds = constrained_group_kfolds(old, n_splits=6, random_seed=args.seed)
    candidates = select_feature_columns(
        old,
        feature_set="physics_enhanced",
        include_mode_command=True,
    )

    predictions: list[pd.DataFrame] = []
    for operating_point in TARGETS:
        fold = next(item for item in folds if operating_point in item.test_groups)
        usable, _ = remove_unusable_training_features(
            fold.train,
            candidates,
            remove_constant=True,
        )
        ranking = rank_features(
            fold.train,
            usable,
            args.seed + fold.fold,
        )
        variants = {
            "top_80": ranking["Feature"].head(80).tolist(),
            "all_features": usable,
        }
        target_new = new.loc[
            new["OperatingPointID"].eq(operating_point)
        ].copy()
        for variant, columns in variants.items():
            for model_name in ("random_forest", "xgboost"):
                result = train_model(
                    model_name,
                    feature_matrix(fold.train, columns),
                    fold.train["WindowFaultID"].to_numpy(dtype=int),
                    feature_matrix(fold.validation, columns),
                    fold.validation["WindowFaultID"].to_numpy(dtype=int),
                    feature_columns=columns,
                    model_parameters=model_parameters()[model_name],
                    random_seed=args.seed + fold.fold,
                    tune=False,
                    tuning_grid={},
                )
                prediction = result.artifact.predict(
                    feature_matrix(target_new, columns)
                )
                part = target_new.loc[
                    :,
                    [
                        "RunID",
                        "OperatingPointID",
                        "WindowID",
                        "WindowStart",
                        "RandomSeed",
                        "Rbat",
                        "Cbus",
                        "CbusESR",
                    ],
                ].copy()
                part.insert(0, "Model", model_name)
                part.insert(1, "Variant", variant)
                part["PredictedClassID"] = prediction
                part["IsFalseAlarm"] = prediction != 0
                predictions.append(part)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(args.output / "window_predictions.csv", index=False)
    summary = (
        prediction_frame.groupby(
            ["Model", "Variant", "OperatingPointID"],
            sort=True,
        )
        .agg(
            HealthyWindowCount=("IsFalseAlarm", "size"),
            FalseAlarmCount=("IsFalseAlarm", "sum"),
            FalseAlarmRate=("IsFalseAlarm", "mean"),
            RunIDCount=("RunID", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    by_run = (
        prediction_frame.groupby(
            [
                "Model",
                "Variant",
                "OperatingPointID",
                "RunID",
                "RandomSeed",
                "Rbat",
                "Cbus",
                "CbusESR",
            ],
            sort=True,
        )
        .agg(
            HealthyWindowCount=("IsFalseAlarm", "size"),
            FalseAlarmCount=("IsFalseAlarm", "sum"),
            FalseAlarmRate=("IsFalseAlarm", "mean"),
        )
        .reset_index()
    )
    by_run.to_csv(args.output / "by_run.csv", index=False)

    correlations: list[dict[str, object]] = []
    for keys, part in by_run.groupby(
        ["Model", "Variant", "OperatingPointID"],
        sort=True,
    ):
        for parameter in ("RandomSeed", "Rbat", "Cbus", "CbusESR"):
            correlations.append(
                {
                    "Model": keys[0],
                    "Variant": keys[1],
                    "OperatingPointID": keys[2],
                    "Parameter": parameter,
                    "SpearmanCorrelation": part[
                        [parameter, "FalseAlarmRate"]
                    ]
                    .corr(method="spearman")
                    .iloc[0, 1],
                }
            )
    pd.DataFrame(correlations).to_csv(
        args.output / "parameter_correlations.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
