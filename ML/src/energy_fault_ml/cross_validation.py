"""Leakage-safe grouped cross-validation for fault-diagnosis models."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .data import constrained_group_kfolds
from .diagnostics import (
    build_prediction_frame,
    write_healthy_false_alarm_diagnostics,
)
from .evaluation import evaluate_classification
from .features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from .models import train_model
from .plotting import save_confusion_matrix


SUMMARY_METRICS = (
    "Accuracy",
    "BalancedAccuracy",
    "MacroF1",
    "HealthyFalseAlarmRate",
)


def _fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, part in fold_metrics.groupby("Model", sort=True):
        row: dict[str, Any] = {
            "Model": model_name,
            "FoldCount": int(part["Fold"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}Mean"] = float(part[metric].mean())
            row[f"{metric}Std"] = float(part[metric].std(ddof=1))
            row[f"{metric}Min"] = float(part[metric].min())
            row[f"{metric}Max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_group_cross_validation(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    model_names: Sequence[str],
    model_parameters: Mapping[str, Mapping[str, Any]],
    tuning_grid: Mapping[str, Mapping[str, Sequence[Any]]],
    tune: bool,
    random_seed: int,
    feature_set: str,
    include_mode_command: bool,
    remove_constant: bool,
    label_column: str,
    expected_labels: Sequence[int],
    label_names: Mapping[int, str],
    group_column: str = "OperatingPointID",
    run_column: str = "RunID",
    n_splits: int = 6,
    max_attempts: int = 2000,
) -> pd.DataFrame:
    """Run constrained operating-point GroupKFold evaluation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds = constrained_group_kfolds(
        frame,
        n_splits=n_splits,
        random_seed=random_seed,
        group_column=group_column,
        run_column=run_column,
        label_column=label_column,
        expected_labels=expected_labels,
        max_attempts=max_attempts,
    )
    candidate_features = select_feature_columns(
        frame,
        feature_set=feature_set,
        include_mode_command=include_mode_command,
    )

    fold_metric_rows: list[dict[str, Any]] = []
    fold_per_class_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for fold in folds:
        test_group_text = "|".join(fold.test_groups)
        validation_group_text = "|".join(fold.validation_groups)
        feature_columns, removed_features = remove_unusable_training_features(
            fold.train,
            candidate_features,
            remove_constant=remove_constant,
        )
        feature_rows.append(
            {
                "Fold": fold.fold,
                "TestOperatingPointIDs": test_group_text,
                "ValidationOperatingPointIDs": validation_group_text,
                "SelectedFeatureCount": len(feature_columns),
                "RemovedFeatureCount": len(removed_features),
                "SelectedFeatures": "|".join(feature_columns),
                "RemovedFeatures": json.dumps(
                    removed_features,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

        train_groups = sorted(
            fold.train[group_column].astype(str).unique().tolist()
        )
        for group in train_groups:
            assignment_rows.append(
                {
                    "Fold": fold.fold,
                    "OperatingPointID": group,
                    "Role": "train",
                }
            )
        assignment_rows.extend(
            {
                "Fold": fold.fold,
                "OperatingPointID": group,
                "Role": "validation",
            }
            for group in fold.validation_groups
        )
        assignment_rows.extend(
            {
                "Fold": fold.fold,
                "OperatingPointID": group,
                "Role": "test",
            }
            for group in fold.test_groups
        )

        X_train = feature_matrix(fold.train, feature_columns)
        X_validation = feature_matrix(fold.validation, feature_columns)
        X_test = feature_matrix(fold.test, feature_columns)
        y_train = fold.train[label_column].to_numpy(dtype=int)
        y_validation = fold.validation[label_column].to_numpy(dtype=int)
        y_test = fold.test[label_column].to_numpy(dtype=int)

        for model_name in model_names:
            result = train_model(
                model_name,
                X_train,
                y_train,
                X_validation,
                y_validation,
                feature_columns=feature_columns,
                model_parameters=model_parameters[model_name],
                random_seed=random_seed + fold.fold,
                tune=tune,
                tuning_grid=tuning_grid.get(model_name, {}),
            )
            trials = result.tuning_trials.copy()
            trials.insert(0, "ValidationOperatingPointIDs", validation_group_text)
            trials.insert(0, "TestOperatingPointIDs", test_group_text)
            trials.insert(0, "Fold", fold.fold)
            trials["Selected"] = trials["ValidationMacroF1"].eq(
                trials["ValidationMacroF1"].max()
            )
            tuning_rows.append(trials)

            prediction_start = time.perf_counter()
            prediction = result.artifact.predict(X_test)
            prediction_seconds = time.perf_counter() - prediction_start
            aggregate, per_class, _ = evaluate_classification(
                y_test,
                prediction,
                labels=expected_labels,
                label_names=label_names,
            )
            aggregate.update(
                {
                    "Model": model_name,
                    "Fold": fold.fold,
                    "TestOperatingPointIDs": test_group_text,
                    "ValidationOperatingPointIDs": validation_group_text,
                    "TrainGroupCount": len(train_groups),
                    "ValidationGroupCount": len(fold.validation_groups),
                    "TestGroupCount": len(fold.test_groups),
                    "SelectedFeatureCount": len(feature_columns),
                    "TrainSeconds": result.training_seconds,
                    "PredictSeconds": prediction_seconds,
                    "PredictMillisecondsPerSample": (
                        1000.0 * prediction_seconds / max(len(y_test), 1)
                    ),
                    "SelectedParameters": json.dumps(
                        result.artifact.selected_parameters,
                        ensure_ascii=False,
                        default=str,
                        sort_keys=True,
                    ),
                }
            )
            fold_metric_rows.append(aggregate)
            per_class.insert(
                0,
                "ValidationOperatingPointIDs",
                validation_group_text,
            )
            per_class.insert(0, "TestOperatingPointIDs", test_group_text)
            per_class.insert(0, "Fold", fold.fold)
            per_class.insert(0, "Model", model_name)
            fold_per_class_rows.append(per_class)
            prediction_rows.append(
                build_prediction_frame(
                    fold.test,
                    prediction,
                    model_name=model_name,
                    split_name="cross_validation",
                    label_column=label_column,
                    label_names=label_names,
                    fold=fold.fold,
                )
            )

    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(
        ["Model", "Fold"], kind="stable"
    )
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary = _fold_summary(fold_metrics)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(fold_per_class_rows, ignore_index=True).to_csv(
        output / "per_class_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame(feature_rows).to_csv(
        output / "fold_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(assignment_rows).sort_values(
        ["Fold", "Role", "OperatingPointID"],
        kind="stable",
    ).to_csv(output / "fold_group_assignments.csv", index=False)
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        output / "tuning_trials.csv",
        index=False,
    )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output / "window_predictions.csv", index=False)
    write_healthy_false_alarm_diagnostics(predictions, output)

    pooled_rows: list[dict[str, Any]] = []
    pooled_per_class_rows: list[pd.DataFrame] = []
    for model_name, part in predictions.groupby("Model", sort=True):
        aggregate, per_class, confusion = evaluate_classification(
            part["TrueClassID"].to_numpy(dtype=int),
            part["PredictedClassID"].to_numpy(dtype=int),
            labels=expected_labels,
            label_names=label_names,
        )
        aggregate.update({"Model": model_name, "Split": "pooled_oof"})
        pooled_rows.append(aggregate)
        per_class.insert(0, "Split", "pooled_oof")
        per_class.insert(0, "Model", model_name)
        pooled_per_class_rows.append(per_class)
        save_confusion_matrix(
            confusion,
            labels=expected_labels,
            label_names=label_names,
            title=f"{model_name} - pooled group CV confusion matrix",
            output_path=output / f"{model_name}_pooled_confusion_matrix.png",
        )

    pd.DataFrame(pooled_rows).to_csv(output / "pooled_metrics.csv", index=False)
    pd.concat(pooled_per_class_rows, ignore_index=True).to_csv(
        output / "pooled_per_class_metrics.csv",
        index=False,
    )
    return summary
