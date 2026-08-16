#!/usr/bin/env python
"""Robust grouped CV with operating-point balancing and worst-OP FAR control."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
for path in (SRC_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from energy_fault_ml.data import constrained_group_kfolds, load_feature_dataset
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model
from run_feature_study import LABEL_NAMES, LABELS
from run_nested_calibrated import (
    expected_calibration_error,
    fit_temperature,
    temperature_scale,
    threshold_prediction,
)


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
DEFAULT_OUTPUT = ML_ROOT / "results" / "robust_nested_expanded_v13_rf_full"

CANDIDATES = (
    {"healthy_multiplier": 1.0, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"healthy_multiplier": 2.0, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"healthy_multiplier": 4.0, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"healthy_multiplier": 2.0, "max_depth": 16, "min_samples_leaf": 3, "max_features": 0.5},
    {"healthy_multiplier": 4.0, "max_depth": 16, "min_samples_leaf": 3, "max_features": 0.5},
    {"healthy_multiplier": 8.0, "max_depth": 12, "min_samples_leaf": 8, "max_features": "sqrt"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-healthy-far", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=240727)
    parser.add_argument(
        "--include-operating-context",
        action="store_true",
        help="Add known SOCInit and Pload setpoints to the online feature set.",
    )
    return parser.parse_args()


def operating_point_cell_weights(
    frame: pd.DataFrame,
    healthy_multiplier: float,
) -> np.ndarray:
    cells = frame.groupby(
        ["OperatingPointID", "WindowFaultID"],
        observed=True,
    )["WindowFaultID"].transform("size")
    weight = 1.0 / cells.to_numpy(dtype=float)
    weight *= np.where(
        frame["WindowFaultID"].to_numpy(dtype=int) == 0,
        healthy_multiplier,
        1.0,
    )
    return weight / np.mean(weight)


def worst_operating_point_far(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> float:
    healthy = frame["WindowFaultID"].to_numpy(dtype=int) == 0
    values = []
    for operating_point in frame.loc[healthy, "OperatingPointID"].unique():
        mask = healthy & frame["OperatingPointID"].eq(operating_point).to_numpy()
        values.append(float(np.mean(prediction[mask] != 0)))
    return max(values, default=0.0)


def choose_robust_threshold(
    probability: np.ndarray,
    frame: pd.DataFrame,
    target_far: float,
) -> tuple[float, pd.DataFrame]:
    labels = frame["WindowFaultID"].to_numpy(dtype=int)
    healthy = labels == 0
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 1.0, 381):
        prediction = threshold_prediction(probability, float(threshold))
        rows.append(
            {
                "Threshold": float(threshold),
                "MacroF1": float(
                    f1_score(labels, prediction, average="macro", zero_division=0)
                ),
                "HealthyFAR": float(np.mean(prediction[healthy] != 0)),
                "WorstOperatingPointFAR": worst_operating_point_far(
                    frame,
                    prediction,
                ),
                "FaultMacroRecall": float(
                    recall_score(
                        labels[~healthy],
                        prediction[~healthy],
                        labels=[1, 2, 3, 4],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    trials = pd.DataFrame(rows)
    feasible = trials.loc[
        trials["WorstOperatingPointFAR"].le(target_far)
        & trials["HealthyFAR"].le(target_far)
    ]
    if feasible.empty:
        best_index = trials.sort_values(
            ["WorstOperatingPointFAR", "HealthyFAR", "MacroF1"],
            ascending=[True, True, False],
        ).index[0]
    else:
        best_index = feasible["MacroF1"].idxmax()
    trials["Selected"] = trials.index == best_index
    return float(trials.loc[best_index, "Threshold"]), trials


def write_progress(
    output: Path,
    *,
    state: str,
    fold: int,
    candidate: int,
    message: str,
) -> None:
    payload = {
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "State": state,
        "CompletedFolds": fold,
        "TotalFolds": 6,
        "CurrentCandidate": candidate,
        "TotalCandidates": len(CANDIDATES),
        "Message": message,
    }
    (output / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_feature_dataset(args.data)
    candidates = select_feature_columns(data, feature_set="physics_enhanced")
    if args.include_operating_context:
        candidates.extend(
            name for name in ("SOCInit", "Pload") if name not in candidates
        )
    folds = constrained_group_kfolds(data, n_splits=6, random_seed=args.seed)
    metric_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    threshold_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    write_progress(
        args.output,
        state="running",
        fold=0,
        candidate=0,
        message="Loading development data.",
    )

    for fold in folds:
        usable, _ = remove_unusable_training_features(
            fold.train,
            candidates,
            remove_constant=True,
        )
        columns = list(usable)
        feature_rows.extend(
            {"Fold": fold.fold, "Feature": feature} for feature in columns
        )
        x_train = feature_matrix(fold.train, columns)
        x_validation = feature_matrix(fold.validation, columns)
        x_test = feature_matrix(fold.test, columns)
        y_train = fold.train["WindowFaultID"].to_numpy(dtype=int)
        y_validation = fold.validation["WindowFaultID"].to_numpy(dtype=int)
        y_test = fold.test["WindowFaultID"].to_numpy(dtype=int)

        best: dict[str, object] | None = None
        for candidate_index, candidate in enumerate(CANDIDATES, start=1):
            write_progress(
                args.output,
                state="running",
                fold=fold.fold - 1,
                candidate=candidate_index,
                message=f"Fold {fold.fold}: fitting candidate {candidate_index}.",
            )
            weights = operating_point_cell_weights(
                fold.train,
                float(candidate["healthy_multiplier"]),
            )
            parameters = {
                "n_estimators": 350,
                "max_depth": candidate["max_depth"],
                "min_samples_leaf": candidate["min_samples_leaf"],
                "max_features": candidate["max_features"],
                "class_weight": None,
                "n_jobs": -1,
            }
            result = train_model(
                "random_forest",
                x_train,
                y_train,
                x_validation,
                y_validation,
                feature_columns=columns,
                model_parameters=parameters,
                random_seed=args.seed + fold.fold * 100 + candidate_index,
                sample_weight_train=weights,
            )
            validation_probability = result.artifact.predict_proba(x_validation)
            if validation_probability is None:
                raise RuntimeError("Random forest did not provide probabilities.")
            temperature = fit_temperature(validation_probability, y_validation)
            calibrated_validation = temperature_scale(
                validation_probability,
                temperature,
            )
            threshold, threshold_trials = choose_robust_threshold(
                calibrated_validation,
                fold.validation,
                args.target_healthy_far,
            )
            validation_prediction = threshold_prediction(
                calibrated_validation,
                threshold,
            )
            validation_macro_f1 = float(
                f1_score(
                    y_validation,
                    validation_prediction,
                    average="macro",
                    zero_division=0,
                )
            )
            validation_healthy_far = float(
                np.mean(validation_prediction[y_validation == 0] != 0)
            )
            validation_worst_far = worst_operating_point_far(
                fold.validation,
                validation_prediction,
            )
            trial_rows.append(
                {
                    "Fold": fold.fold,
                    "Candidate": candidate_index,
                    **candidate,
                    "Temperature": temperature,
                    "Threshold": threshold,
                    "ValidationMacroF1": validation_macro_f1,
                    "ValidationHealthyFAR": validation_healthy_far,
                    "ValidationWorstOperatingPointFAR": validation_worst_far,
                    "TrainingSeconds": result.training_seconds,
                }
            )
            threshold_trials.insert(0, "Fold", fold.fold)
            threshold_trials.insert(1, "Candidate", candidate_index)
            threshold_rows.append(threshold_trials)
            selection_key = (
                validation_worst_far <= args.target_healthy_far,
                validation_macro_f1,
                -validation_worst_far,
            )
            if best is None or selection_key > best["selection_key"]:
                best = {
                    "selection_key": selection_key,
                    "artifact": result.artifact,
                    "temperature": temperature,
                    "threshold": threshold,
                    "candidate": candidate_index,
                    "parameters": candidate,
                }
            del result
            gc.collect()

        if best is None:
            raise RuntimeError(f"No candidate completed for fold {fold.fold}.")
        artifact = best["artifact"]
        raw_probability = artifact.predict_proba(x_test)
        if raw_probability is None:
            raise RuntimeError("Random forest did not provide test probabilities.")
        calibrated_probability = temperature_scale(
            raw_probability,
            float(best["temperature"]),
        )
        variants = {
            "raw_argmax": np.argmax(raw_probability, axis=1),
            "calibrated_argmax": np.argmax(calibrated_probability, axis=1),
            "robust_threshold": threshold_prediction(
                calibrated_probability,
                float(best["threshold"]),
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
                    "Variant": variant,
                    "FeatureCount": len(columns),
                    "TestOperatingPointIDs": "|".join(fold.test_groups),
                    "ValidationOperatingPointIDs": "|".join(fold.validation_groups),
                    "SelectedCandidate": int(best["candidate"]),
                    "HealthyMultiplier": float(
                        best["parameters"]["healthy_multiplier"]
                    ),
                    "Temperature": float(best["temperature"]),
                    "AlarmThreshold": float(best["threshold"]),
                    "WorstOperatingPointFAR": worst_operating_point_far(
                        fold.test,
                        prediction,
                    ),
                    "TestECE": expected_calibration_error(
                        calibrated_probability
                        if variant != "raw_argmax"
                        else raw_probability,
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
                    calibrated_probability[:, class_id]
                    if variant != "raw_argmax"
                    else raw_probability[:, class_id]
                )
            prediction_rows.append(prediction_frame)

        pd.DataFrame(metric_rows).to_csv(
            args.output / "fold_metrics_partial.csv",
            index=False,
        )
        pd.DataFrame(trial_rows).to_csv(
            args.output / "candidate_trials_partial.csv",
            index=False,
        )
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            args.output / "window_predictions_partial.csv",
            index=False,
        )
        write_progress(
            args.output,
            state="running",
            fold=fold.fold,
            candidate=0,
            message=f"Fold {fold.fold} completed.",
        )
        print(f"Completed fold {fold.fold}/6", flush=True)
        del best, artifact, x_train, x_validation, x_test
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    (
        metrics.groupby(["Variant", "FeatureCount"], sort=True)
        .agg(
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            MacroF1Min=("MacroF1", "min"),
            HealthyFARMean=("HealthyFalseAlarmRate", "mean"),
            HealthyFARStd=("HealthyFalseAlarmRate", "std"),
            WorstOperatingPointFARMean=("WorstOperatingPointFAR", "mean"),
            WorstOperatingPointFARMax=("WorstOperatingPointFAR", "max"),
            TestECEMean=("TestECE", "mean"),
            AlarmThresholdMean=("AlarmThreshold", "mean"),
        )
        .reset_index()
        .to_csv(args.output / "summary.csv", index=False)
    )
    pd.DataFrame(trial_rows).to_csv(
        args.output / "candidate_trials.csv",
        index=False,
    )
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        args.output / "threshold_trials.csv",
        index=False,
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "window_predictions.csv",
        index=False,
    )
    pd.DataFrame(feature_rows).to_csv(
        args.output / "selected_features.csv",
        index=False,
    )
    write_progress(
        args.output,
        state="completed",
        fold=6,
        candidate=0,
        message="Robust nested training completed.",
    )
    print(pd.read_csv(args.output / "summary.csv").to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
