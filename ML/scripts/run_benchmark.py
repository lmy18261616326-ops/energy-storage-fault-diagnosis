#!/usr/bin/env python
"""Run a leakage-safe RF/SVM/XGBoost comparison."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml

from energy_fault_ml.data import (
    constrained_group_split,
    load_feature_dataset,
    run_split_assignments,
    split_class_summary,
)
from energy_fault_ml.cross_validation import run_group_cross_validation
from energy_fault_ml.diagnostics import (
    build_prediction_frame,
    write_healthy_false_alarm_diagnostics,
)
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import SUPPORTED_MODELS, train_model
from energy_fault_ml.plotting import save_confusion_matrix


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
DEFAULT_CONFIG = ML_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT = ML_ROOT / "results"
DEFAULT_MODELS = ML_ROOT / "models"


def _path_from_argument(value: str | Path, *, is_output: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0].lower() == "ml":
        return (PROJECT_ROOT / path).resolve()
    if is_output:
        return (Path.cwd() / path).resolve()
    candidates = [Path.cwd() / path, PROJECT_ROOT / path, ML_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in ("project", "data", "features", "models", "tuning"):
        if section not in config:
            raise ValueError(f"Configuration is missing section: {section}")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Random Forest, RBF-SVM, and XGBoost on grouped "
            "energy-storage fault windows."
        )
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="MATLAB-generated feature_dataset.csv.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory for metrics, figures, and split metadata.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODELS),
        help="Directory for serialized model artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="One or more models to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the default random seed 240727.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("statistical", "physics_enhanced"),
        default=None,
        help="Override the configured feature set.",
    )
    parser.add_argument(
        "--exclude-mode-command",
        action="store_true",
        help="Exclude ModeCommand for the mode-dependence ablation.",
    )
    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument(
        "--tune",
        action="store_true",
        dest="tune",
        help="Enable the configured lightweight validation search.",
    )
    tune_group.add_argument(
        "--no-tune",
        action="store_false",
        dest="tune",
        help="Disable tuning even if enabled in YAML.",
    )
    parser.set_defaults(tune=None)
    parser.add_argument(
        "--group-cv",
        action="store_true",
        help=(
            "After the standard holdout benchmark, run constrained "
            "OperatingPointID GroupKFold under OUTPUT/group_cv."
        ),
    )
    return parser.parse_args()


def _write_run_metadata(
    output_dir: Path,
    *,
    data_path: Path,
    config_path: Path,
    seed: int,
    feature_set: str,
    include_mode: bool,
    models: list[str],
    group_cv: bool,
) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "config_path": str(config_path),
        "random_seed": seed,
        "feature_set": feature_set,
        "include_mode_command": include_mode,
        "models": models,
        "group_cross_validation": group_cv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    data_path = _path_from_argument(args.data)
    config_path = _path_from_argument(args.config)
    output_dir = _path_from_argument(args.output, is_output=True)
    model_dir = _path_from_argument(args.model_dir, is_output=True)
    config = _load_config(config_path)

    data_config = config["data"]
    feature_config = config["features"]
    seed = int(args.seed if args.seed is not None else 240727)
    feature_set = args.feature_set or str(
        feature_config.get("feature_set", "physics_enhanced")
    )
    include_mode = bool(feature_config.get("include_mode_command", True))
    if args.exclude_mode_command:
        include_mode = False
    tune = (
        bool(config["tuning"].get("enabled", False))
        if args.tune is None
        else bool(args.tune)
    )

    expected_labels = [int(value) for value in data_config["expected_labels"]]
    raw_label_names = config["project"]["label_names"]
    label_names = {int(key): str(value) for key, value in raw_label_names.items()}
    label_column = str(data_config.get("label_column", "WindowFaultID"))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = load_feature_dataset(
        data_path,
        label_column=label_column,
        eligibility_column=str(
            data_config.get("eligibility_column", "IsTrainingEligible")
        ),
        expected_labels=expected_labels,
    )
    bundle = constrained_group_split(
        frame,
        ratios=data_config.get("split_ratios", [0.70, 0.15, 0.15]),
        random_seed=seed,
        group_column=str(data_config.get("group_column", "OperatingPointID")),
        run_column=str(data_config.get("run_column", "RunID")),
        label_column=label_column,
        expected_labels=expected_labels,
        max_attempts=int(data_config.get("split_max_attempts", 2000)),
    )

    candidates = select_feature_columns(
        frame,
        feature_set=feature_set,
        include_mode_command=include_mode,
    )
    feature_columns, removed_features = remove_unusable_training_features(
        bundle.train,
        candidates,
        remove_constant=bool(
            feature_config.get("remove_constant_from_training", True)
        ),
    )
    (output_dir / "feature_list.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"Feature": name, "Reason": reason}
            for name, reason in sorted(removed_features.items())
        ],
        columns=["Feature", "Reason"],
    ).to_csv(output_dir / "removed_features.csv", index=False)

    bundle.group_assignments.to_csv(
        output_dir / "split_group_assignments.csv", index=False
    )
    run_split_assignments(bundle).to_csv(
        output_dir / "run_split_assignments.csv", index=False
    )
    split_class_summary(bundle, label_column).to_csv(
        output_dir / "split_summary.csv", index=False
    )

    used_config = json.loads(json.dumps(config))
    used_config["run_overrides"] = {
        "seed": seed,
        "feature_set": feature_set,
        "include_mode_command": include_mode,
        "tune": tune,
        "models": list(args.models),
        "group_cv": bool(args.group_cv),
    }
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(used_config, handle, sort_keys=False, allow_unicode=True)
    _write_run_metadata(
        output_dir,
        data_path=data_path,
        config_path=config_path,
        seed=seed,
        feature_set=feature_set,
        include_mode=include_mode,
        models=list(args.models),
        group_cv=bool(args.group_cv),
    )

    X_train = feature_matrix(bundle.train, feature_columns)
    X_validation = feature_matrix(bundle.validation, feature_columns)
    X_test = feature_matrix(bundle.test, feature_columns)
    y_train = bundle.train[label_column].to_numpy(dtype=int)
    y_validation = bundle.validation[label_column].to_numpy(dtype=int)
    y_test = bundle.test[label_column].to_numpy(dtype=int)

    comparison_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name in args.models:
        model_result = train_model(
            model_name,
            X_train,
            y_train,
            X_validation,
            y_validation,
            feature_columns=feature_columns,
            model_parameters=config["models"][model_name],
            random_seed=seed,
            tune=tune,
            tuning_grid=config["tuning"]
            .get("parameter_grid", {})
            .get(model_name, {}),
        )
        artifact = model_result.artifact
        joblib.dump(artifact, model_dir / f"{model_name}.joblib")
        trials = model_result.tuning_trials.copy()
        trials["Selected"] = trials["ValidationMacroF1"].eq(
            trials["ValidationMacroF1"].max()
        )
        tuning_rows.append(trials)

        for split_name, X_part, y_part in (
            ("validation", X_validation, y_validation),
            ("test", X_test, y_test),
        ):
            prediction_start = time.perf_counter()
            prediction = artifact.predict(X_part)
            prediction_seconds = time.perf_counter() - prediction_start
            aggregate, per_class, confusion = evaluate_classification(
                y_part,
                prediction,
                labels=expected_labels,
                label_names=label_names,
            )
            aggregate.update(
                {
                    "Model": model_name,
                    "Split": split_name,
                    "TrainSeconds": model_result.training_seconds,
                    "PredictSeconds": prediction_seconds,
                    "PredictMillisecondsPerSample": (
                        1000.0 * prediction_seconds / max(len(y_part), 1)
                    ),
                    "SelectedParameters": json.dumps(
                        artifact.selected_parameters,
                        ensure_ascii=False,
                        default=str,
                        sort_keys=True,
                    ),
                }
            )
            comparison_rows.append(aggregate)
            per_class.insert(0, "Split", split_name)
            per_class.insert(0, "Model", model_name)
            per_class_rows.append(per_class)
            prediction_rows.append(
                build_prediction_frame(
                    bundle.validation if split_name == "validation" else bundle.test,
                    prediction,
                    model_name=model_name,
                    split_name=split_name,
                    label_column=label_column,
                    label_names=label_names,
                )
            )
            save_confusion_matrix(
                confusion,
                labels=expected_labels,
                label_names=label_names,
                title=f"{model_name} - {split_name} confusion matrix",
                output_path=output_dir
                / f"{model_name}_{split_name}_confusion_matrix.png",
            )

    comparison = pd.DataFrame(comparison_rows)
    leading = [
        "Model",
        "Split",
        "Accuracy",
        "BalancedAccuracy",
        "MacroPrecision",
        "MacroRecall",
        "MacroF1",
        "HealthyFalseAlarmRate",
        "SampleCount",
        "TrainSeconds",
        "PredictSeconds",
        "PredictMillisecondsPerSample",
        "SelectedParameters",
    ]
    comparison.loc[:, leading].to_csv(
        output_dir / "model_comparison.csv", index=False
    )
    pd.concat(per_class_rows, ignore_index=True).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        output_dir / "tuning_trials.csv", index=False
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output_dir / "window_predictions.csv", index=False)
    write_healthy_false_alarm_diagnostics(predictions, output_dir)

    if args.group_cv:
        cv_output = output_dir / "group_cv"
        cv_summary = run_group_cross_validation(
            frame,
            output_dir=cv_output,
            model_names=list(args.models),
            model_parameters=config["models"],
            tuning_grid=config["tuning"].get("parameter_grid", {}),
            tune=tune,
            random_seed=seed,
            feature_set=feature_set,
            include_mode_command=include_mode,
            remove_constant=bool(
                feature_config.get("remove_constant_from_training", True)
            ),
            label_column=label_column,
            expected_labels=expected_labels,
            label_names=label_names,
            group_column=str(
                data_config.get("group_column", "OperatingPointID")
            ),
            run_column=str(data_config.get("run_column", "RunID")),
            n_splits=int(data_config.get("group_cv_folds", 6)),
            max_attempts=int(
                data_config.get("group_cv_max_attempts", 2000)
            ),
        )
        print(f"Group CV folds: {int(cv_summary['FoldCount'].max())}")
        print(f"Group CV metrics: {cv_output / 'summary.csv'}")

    print(f"Completed models: {', '.join(args.models)}")
    print(f"Eligible windows: {len(frame)}")
    print(f"Selected features: {len(feature_columns)}")
    print(f"Metrics: {output_dir / 'model_comparison.csv'}")
    print(f"Models: {model_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
