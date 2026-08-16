#!/usr/bin/env python
"""Nested grouped tuning, temperature calibration, and alarm-threshold search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from energy_fault_ml.data import constrained_group_kfolds, load_feature_dataset
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model
from run_feature_study import LABEL_NAMES, LABELS, rank_features


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
DEFAULT_OUTPUT = ML_ROOT / "results" / "nested_calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
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


def model_setup(
    model_name: str,
) -> tuple[dict[str, object], dict[str, list[object]]]:
    if model_name == "random_forest":
        parameters: dict[str, object] = {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        grid: dict[str, list[object]] = {
            "max_depth": [None, 16],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.5],
        }
        return parameters, grid
    parameters = {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 40,
        "n_jobs": -1,
    }
    grid = {
        "learning_rate": [0.03, 0.07],
        "max_depth": [4, 6],
        "min_child_weight": [1.0, 3.0],
    }
    return parameters, grid


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-12, 1.0))
    return softmax(logits / temperature)


def fit_temperature(probability: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_temperature: log_loss(
            labels,
            temperature_scale(probability, np.exp(log_temperature)),
            labels=list(LABELS),
        ),
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def threshold_prediction(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    fault_probability = probability[:, 1:]
    fault_class = np.argmax(fault_probability, axis=1) + 1
    alarm_probability = 1.0 - probability[:, 0]
    return np.where(alarm_probability >= threshold, fault_class, 0)


def optimize_threshold(
    probability: np.ndarray,
    labels: np.ndarray,
    target_far: float,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 1.00, 192):
        prediction = threshold_prediction(probability, float(threshold))
        healthy = labels == 0
        far = float(np.mean(prediction[healthy] != 0))
        rows.append(
            {
                "Threshold": float(threshold),
                "MacroF1": float(
                    f1_score(
                        labels,
                        prediction,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "HealthyFalseAlarmRate": far,
            }
        )
    trials = pd.DataFrame(rows)
    feasible = trials.loc[
        trials["HealthyFalseAlarmRate"].le(target_far + 1e-12)
    ]
    candidates = feasible if not feasible.empty else trials
    selected = candidates.sort_values(
        ["MacroF1", "HealthyFalseAlarmRate", "Threshold"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    return float(selected["Threshold"]), trials


def expected_calibration_error(
    probability: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        accuracy = np.mean(prediction[mask] == labels[mask])
        error += np.mean(mask) * abs(accuracy - np.mean(confidence[mask]))
    return float(error)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_feature_dataset(args.data)
    folds = constrained_group_kfolds(
        frame,
        n_splits=6,
        random_seed=args.seed,
    )
    candidates = select_feature_columns(
        frame,
        feature_set="physics_enhanced",
        include_mode_command=True,
    )
    parameters, tuning_grid = model_setup(args.model)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    for fold in folds:
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
        columns = ranking["Feature"].head(args.feature_count).tolist()
        feature_rows.extend(
            {
                "Fold": fold.fold,
                "Rank": rank + 1,
                "Feature": feature,
            }
            for rank, feature in enumerate(columns)
        )
        x_train = feature_matrix(fold.train, columns)
        x_validation = feature_matrix(fold.validation, columns)
        x_test = feature_matrix(fold.test, columns)
        y_train = fold.train["WindowFaultID"].to_numpy(dtype=int)
        y_validation = fold.validation["WindowFaultID"].to_numpy(dtype=int)
        y_test = fold.test["WindowFaultID"].to_numpy(dtype=int)

        result = train_model(
            args.model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            feature_columns=columns,
            model_parameters=parameters,
            random_seed=args.seed + fold.fold,
            tune=True,
            tuning_grid=tuning_grid,
        )
        best_iteration = getattr(
            result.artifact.estimator,
            "best_iteration",
            None,
        )
        trials = result.tuning_trials.copy()
        trials.insert(0, "Fold", fold.fold)
        trials["Selected"] = trials["ValidationMacroF1"].eq(
            trials["ValidationMacroF1"].max()
        )
        tuning_rows.append(trials)
        validation_probability = result.artifact.predict_proba(x_validation)
        test_probability = result.artifact.predict_proba(x_test)
        if validation_probability is None or test_probability is None:
            raise RuntimeError(f"{args.model} does not provide probabilities.")

        temperature = fit_temperature(validation_probability, y_validation)
        calibrated_validation = temperature_scale(
            validation_probability,
            temperature,
        )
        calibrated_test = temperature_scale(test_probability, temperature)
        threshold, threshold_trials = optimize_threshold(
            calibrated_validation,
            y_validation,
            args.target_healthy_far,
        )
        threshold_trials.insert(0, "Fold", fold.fold)
        threshold_trials["Selected"] = threshold_trials["Threshold"].eq(
            threshold
        )
        threshold_rows.append(threshold_trials)

        variants = {
            "raw_argmax": np.argmax(test_probability, axis=1),
            "calibrated_argmax": np.argmax(calibrated_test, axis=1),
            "calibrated_threshold": threshold_prediction(
                calibrated_test,
                threshold,
            ),
        }
        for variant, prediction in variants.items():
            aggregate, _, _ = evaluate_classification(
                y_test,
                prediction,
                labels=LABELS,
                label_names=LABEL_NAMES,
            )
            metric_rows.append(
                {
                    "Fold": fold.fold,
                    "Model": args.model,
                    "Variant": variant,
                    "FeatureCount": len(columns),
                    "TestOperatingPointIDs": "|".join(fold.test_groups),
                    "ValidationOperatingPointIDs": "|".join(
                        fold.validation_groups
                    ),
                    "Temperature": temperature,
                    "AlarmThreshold": threshold,
                    "BestIteration": (
                        int(best_iteration)
                        if best_iteration is not None
                        else np.nan
                    ),
                    "TestLogLoss": log_loss(
                        y_test,
                        calibrated_test
                        if variant != "raw_argmax"
                        else test_probability,
                        labels=list(LABELS),
                    ),
                    "TestECE": expected_calibration_error(
                        calibrated_test
                        if variant != "raw_argmax"
                        else test_probability,
                        y_test,
                    ),
                    **aggregate,
                }
            )
            prediction_frame = fold.test.loc[
                :,
                [
                    "RunID",
                    "OperatingPointID",
                    "WindowID",
                    "WindowStart",
                    "WindowEnd",
                    "WindowFaultID",
                ],
            ].copy()
            prediction_frame.insert(0, "Fold", fold.fold)
            prediction_frame.insert(1, "Variant", variant)
            prediction_frame["PredictedClassID"] = prediction
            for class_id in LABELS:
                prediction_frame[f"ProbabilityClass{class_id}"] = (
                    calibrated_test[:, class_id]
                    if variant != "raw_argmax"
                    else test_probability[:, class_id]
                )
            prediction_rows.append(prediction_frame)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    (
        metrics.groupby(["Model", "Variant", "FeatureCount"], sort=True)
        .agg(
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            MacroF1Min=("MacroF1", "min"),
            HealthyFalseAlarmRateMean=("HealthyFalseAlarmRate", "mean"),
            HealthyFalseAlarmRateStd=("HealthyFalseAlarmRate", "std"),
            TestLogLossMean=("TestLogLoss", "mean"),
            TestECEMean=("TestECE", "mean"),
            AlarmThresholdMean=("AlarmThreshold", "mean"),
            AlarmThresholdStd=("AlarmThreshold", "std"),
        )
        .reset_index()
        .to_csv(args.output / "summary.csv", index=False)
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "window_predictions.csv",
        index=False,
    )
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        args.output / "tuning_trials.csv",
        index=False,
    )
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        args.output / "threshold_trials.csv",
        index=False,
    )
    pd.DataFrame(feature_rows).to_csv(
        args.output / "selected_features.csv",
        index=False,
    )
    print(pd.read_csv(args.output / "summary.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
