#!/usr/bin/env python
"""Evaluate a mode-gated RF ensemble with nested operating-point validation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

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
    softmax,
    temperature_scale,
    threshold_prediction,
)
from run_robust_nested_calibrated import operating_point_cell_weights


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
DEFAULT_OUTPUT = ML_ROOT / "results" / "mode_gated_nested_expanded_v13_rf_full"
HEALTHY_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-healthy-far", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=240727)
    return parser.parse_args()


def fit_temperature_generic(
    probability: np.ndarray,
    labels: np.ndarray,
) -> float:
    class_labels = list(range(probability.shape[1]))
    result = minimize_scalar(
        lambda log_temperature: log_loss(
            labels,
            softmax(
                np.log(np.clip(probability, 1e-12, 1.0))
                / np.exp(log_temperature)
            ),
            labels=class_labels,
        ),
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def choose_threshold(
    probability: np.ndarray,
    labels: np.ndarray,
    target_far: float,
) -> tuple[float, float, float]:
    present_labels = np.arange(probability.shape[1])
    rows = []
    for threshold in np.linspace(0.05, 1.0, 381):
        prediction = threshold_prediction(probability, float(threshold))
        far = float(np.mean(prediction[labels == 0] != 0))
        macro_f1 = float(
            f1_score(
                labels,
                prediction,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        )
        rows.append((float(threshold), macro_f1, far))
    feasible = [row for row in rows if row[2] <= target_far]
    selected = max(feasible or rows, key=lambda row: (row[1], -row[2]))
    return selected


def write_progress(
    output: Path,
    *,
    state: str,
    completed_folds: int,
    current_fold: int,
    current_mode: int,
    current_candidate: int,
    message: str,
) -> None:
    payload = {
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "State": state,
        "CompletedFolds": completed_folds,
        "TotalFolds": 6,
        "CurrentFold": current_fold,
        "CurrentMode": current_mode,
        "CurrentCandidate": current_candidate,
        "TotalCandidatesPerMode": len(HEALTHY_MULTIPLIERS),
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
    feature_candidates = select_feature_columns(
        data,
        feature_set="physics_enhanced",
    )
    outer_folds = constrained_group_kfolds(
        data,
        n_splits=6,
        random_seed=args.seed,
    )
    metric_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    write_progress(
        args.output,
        state="running",
        completed_folds=0,
        current_fold=1,
        current_mode=0,
        current_candidate=0,
        message="Loading development data.",
    )

    for outer in outer_folds:
        development = pd.concat(
            [outer.train, outer.validation],
            ignore_index=True,
        )
        combined_prediction = np.zeros(len(outer.test), dtype=int)
        combined_probability = np.zeros((len(outer.test), len(LABELS)))
        selected_modes: list[str] = []

        for mode in sorted(outer.test["ModeCommand"].unique()):
            mode = int(mode)
            test_mask = outer.test["ModeCommand"].eq(mode).to_numpy()
            mode_development = development.loc[
                development["ModeCommand"].eq(mode)
            ].copy()
            operating_points = sorted(
                mode_development["OperatingPointID"].unique()
            )
            validation_op = operating_points[
                (outer.fold + mode - 1) % len(operating_points)
            ]
            mode_validation = mode_development.loc[
                mode_development["OperatingPointID"].eq(validation_op)
            ].copy()
            mode_train = mode_development.loc[
                ~mode_development["OperatingPointID"].eq(validation_op)
            ].copy()
            usable, _ = remove_unusable_training_features(
                mode_train,
                feature_candidates,
                remove_constant=True,
            )
            columns = list(usable)
            feature_rows.extend(
                {
                    "Fold": outer.fold,
                    "ModeCommand": mode,
                    "Feature": feature,
                }
                for feature in columns
            )
            x_train = feature_matrix(mode_train, columns)
            x_validation = feature_matrix(mode_validation, columns)
            x_test = feature_matrix(outer.test.loc[test_mask], columns)
            y_train = mode_train["WindowFaultID"].to_numpy(dtype=int)
            y_validation = mode_validation["WindowFaultID"].to_numpy(dtype=int)

            best: dict[str, object] | None = None
            for candidate_index, multiplier in enumerate(
                HEALTHY_MULTIPLIERS,
                start=1,
            ):
                write_progress(
                    args.output,
                    state="running",
                    completed_folds=outer.fold - 1,
                    current_fold=outer.fold,
                    current_mode=mode,
                    current_candidate=candidate_index,
                    message=(
                        f"Fold {outer.fold}, mode {mode}: "
                        f"candidate {candidate_index}."
                    ),
                )
                weights = operating_point_cell_weights(
                    mode_train,
                    multiplier,
                )
                result = train_model(
                    "random_forest",
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    feature_columns=columns,
                    model_parameters={
                        "n_estimators": 300,
                        "max_depth": 16,
                        "min_samples_leaf": 3,
                        "max_features": "sqrt",
                        "class_weight": None,
                        "n_jobs": -1,
                    },
                    random_seed=(
                        args.seed
                        + outer.fold * 100
                        + mode * 10
                        + candidate_index
                    ),
                    sample_weight_train=weights,
                )
                probability = result.artifact.predict_proba(x_validation)
                if probability is None:
                    raise RuntimeError("Random forest has no probabilities.")
                temperature = fit_temperature_generic(
                    probability,
                    y_validation,
                )
                calibrated = temperature_scale(probability, temperature)
                threshold, macro_f1, far = choose_threshold(
                    calibrated,
                    y_validation,
                    args.target_healthy_far,
                )
                trial_rows.append(
                    {
                        "Fold": outer.fold,
                        "ModeCommand": mode,
                        "ValidationOperatingPointID": validation_op,
                        "Candidate": candidate_index,
                        "HealthyMultiplier": multiplier,
                        "Temperature": temperature,
                        "Threshold": threshold,
                        "ValidationMacroF1": macro_f1,
                        "ValidationHealthyFAR": far,
                        "TrainingSeconds": result.training_seconds,
                    }
                )
                key = (far <= args.target_healthy_far, macro_f1, -far)
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "artifact": result.artifact,
                        "temperature": temperature,
                        "threshold": threshold,
                        "multiplier": multiplier,
                        "candidate": candidate_index,
                    }
                del result
                gc.collect()

            if best is None:
                raise RuntimeError(
                    f"No model completed for fold {outer.fold}, mode {mode}."
                )
            raw_probability = best["artifact"].predict_proba(x_test)
            if raw_probability is None:
                raise RuntimeError("Random forest has no test probabilities.")
            calibrated_probability = temperature_scale(
                raw_probability,
                float(best["temperature"]),
            )
            prediction = threshold_prediction(
                calibrated_probability,
                float(best["threshold"]),
            )
            combined_prediction[test_mask] = prediction
            combined_probability[
                np.ix_(test_mask, np.arange(calibrated_probability.shape[1]))
            ] = calibrated_probability
            selected_modes.append(
                f"M{mode}:C{best['candidate']}/H{best['multiplier']}"
            )
            del best, x_train, x_validation, x_test
            gc.collect()

        y_test = outer.test["WindowFaultID"].to_numpy(dtype=int)
        aggregate, _, _ = evaluate_classification(
            y_test,
            combined_prediction,
            labels=LABELS,
            label_names=LABEL_NAMES,
        )
        healthy = y_test == 0
        per_op_far = []
        for operating_point in outer.test.loc[
            healthy,
            "OperatingPointID",
        ].unique():
            mask = (
                healthy
                & outer.test["OperatingPointID"].eq(operating_point).to_numpy()
            )
            per_op_far.append(
                float(np.mean(combined_prediction[mask] != 0))
            )
        metric_rows.append(
            {
                "Fold": outer.fold,
                "Variant": "mode_gated_threshold",
                "TestOperatingPointIDs": "|".join(outer.test_groups),
                "SelectedModeCandidates": "|".join(selected_modes),
                "WorstOperatingPointFAR": max(per_op_far, default=0.0),
                "TestECE": expected_calibration_error(
                    combined_probability,
                    y_test,
                ),
                **aggregate,
            }
        )
        prediction_frame = outer.test.loc[
            :,
            [
                "RunID",
                "OperatingPointID",
                "ModeCommand",
                "WindowID",
                "WindowStart",
                "WindowEnd",
                "WindowFaultID",
            ],
        ].copy()
        prediction_frame.insert(0, "Fold", outer.fold)
        prediction_frame["PredictedClassID"] = combined_prediction
        for class_id in LABELS:
            prediction_frame[f"ProbabilityClass{class_id}"] = (
                combined_probability[:, class_id]
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
            completed_folds=outer.fold,
            current_fold=outer.fold + 1,
            current_mode=0,
            current_candidate=0,
            message=f"Fold {outer.fold} completed.",
        )
        print(f"Completed fold {outer.fold}/6", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    summary = {
        "Variant": "mode_gated_threshold",
        "MacroF1Mean": float(metrics["MacroF1"].mean()),
        "MacroF1Std": float(metrics["MacroF1"].std()),
        "MacroF1Min": float(metrics["MacroF1"].min()),
        "HealthyFARMean": float(metrics["HealthyFalseAlarmRate"].mean()),
        "HealthyFARStd": float(metrics["HealthyFalseAlarmRate"].std()),
        "WorstOperatingPointFARMean": float(
            metrics["WorstOperatingPointFAR"].mean()
        ),
        "WorstOperatingPointFARMax": float(
            metrics["WorstOperatingPointFAR"].max()
        ),
        "TestECEMean": float(metrics["TestECE"].mean()),
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(
        args.output / "candidate_trials.csv",
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
        completed_folds=6,
        current_fold=6,
        current_mode=0,
        current_candidate=0,
        message="Mode-gated nested evaluation completed.",
    )
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
