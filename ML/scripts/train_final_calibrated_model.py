#!/usr/bin/env python
"""Train the locked final model and calibrate it from honest OOF probabilities."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from energy_fault_ml.calibration import CalibratedAlarmArtifact
from energy_fault_ml.data import load_feature_dataset
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model
from run_feature_study import model_parameters, rank_features
from run_nested_calibrated import fit_temperature, optimize_threshold


DEFAULT_DATA = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "combined_development"
    / "feature_dataset.csv"
)
DEFAULT_NESTED = ML_ROOT / "results" / "nested_calibrated_expanded"
DEFAULT_OUTPUT = ML_ROOT / "models" / "final_calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--nested-results", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=("random_forest", "xgboost"),
        default="xgboost",
    )
    parser.add_argument("--feature-count", type=int, default=80)
    parser.add_argument("--target-healthy-far", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=240727)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_feature_dataset(args.data)
    candidates = select_feature_columns(
        frame,
        feature_set="physics_enhanced",
        include_mode_command=True,
    )
    usable, removed = remove_unusable_training_features(
        frame,
        candidates,
        remove_constant=True,
    )
    ranking = rank_features(frame, usable, args.seed)
    columns = ranking["Feature"].head(args.feature_count).tolist()

    trials = pd.read_csv(args.nested_results / "tuning_trials.csv")
    parameter_score = (
        trials.groupby("Parameters", sort=True)["ValidationMacroF1"]
        .agg(["mean", "std", "min"])
        .reset_index()
        .sort_values(
            ["mean", "std", "min"],
            ascending=[False, True, False],
            kind="stable",
        )
    )
    selected_parameters = ast.literal_eval(
        str(parameter_score.iloc[0]["Parameters"])
    )
    fold_metrics = pd.read_csv(args.nested_results / "fold_metrics.csv")
    best_iterations = (
        fold_metrics[["Fold", "BestIteration"]]
        .drop_duplicates()
        ["BestIteration"]
        .dropna()
    )
    if args.model == "xgboost":
        selected_parameters.pop("early_stopping_rounds", None)
        if not best_iterations.empty:
            selected_parameters["n_estimators"] = int(
                np.median(best_iterations) + 1
            )
    baseline = model_parameters()[args.model]
    baseline.update(selected_parameters)
    baseline.pop("early_stopping_rounds", None)

    matrix = feature_matrix(frame, columns)
    labels = frame["WindowFaultID"].to_numpy(dtype=int)
    trained = train_model(
        args.model,
        matrix,
        labels,
        matrix,
        labels,
        feature_columns=columns,
        model_parameters=baseline,
        random_seed=args.seed,
        tune=False,
        tuning_grid={},
    )

    oof = pd.read_csv(args.nested_results / "window_predictions.csv")
    oof = oof.loc[oof["Variant"].eq("raw_argmax")].copy()
    probability_columns = [f"ProbabilityClass{index}" for index in range(5)]
    oof_probability = oof[probability_columns].to_numpy(dtype=float)
    oof_labels = oof["WindowFaultID"].to_numpy(dtype=int)
    temperature = fit_temperature(oof_probability, oof_labels)
    from energy_fault_ml.calibration import temperature_scale

    calibrated_oof = temperature_scale(oof_probability, temperature)
    threshold, threshold_trials = optimize_threshold(
        calibrated_oof,
        oof_labels,
        args.target_healthy_far,
    )
    artifact = CalibratedAlarmArtifact(
        base_artifact=trained.artifact,
        temperature=temperature,
        alarm_threshold=threshold,
    )
    joblib.dump(artifact, args.output / f"{args.model}.joblib")
    ranking.to_csv(args.output / "feature_ranking.csv", index=False)
    (args.output / "selected_features.txt").write_text(
        "\n".join(columns) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"Feature": feature, "Reason": reason}
            for feature, reason in sorted(removed.items())
        ]
    ).to_csv(args.output / "removed_features.csv", index=False)
    parameter_score.to_csv(
        args.output / "parameter_aggregation.csv",
        index=False,
    )
    threshold_trials.to_csv(
        args.output / "global_threshold_trials.csv",
        index=False,
    )
    metadata = {
        "model": args.model,
        "data": str(args.data.resolve()),
        "feature_count": len(columns),
        "selected_parameters": baseline,
        "temperature": temperature,
        "alarm_threshold": threshold,
        "target_healthy_false_alarm_rate": args.target_healthy_far,
        "calibration_source": "six-fold out-of-fold development probabilities",
        "blind_data_used": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
