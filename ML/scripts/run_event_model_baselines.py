#!/usr/bin/env python
"""Run leakage-aware event-level RF/XGBoost baselines.

This is the first stage of the new model-selection cycle.  It does not use the
consumed blind set for fitting or qualification.  Each outer fold holds out
complete operating points; thresholds are chosen only on that fold's validation
operating points.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.data import constrained_group_kfolds
from energy_fault_ml.event_features import (
    DEFAULT_EVENT_SIGNALS,
    build_event_dataset,
    fine_probability_to_coarse,
    required_event_columns,
)
from energy_fault_ml.numpy_cnn import NumpyConv1DClassifier


def hierarchical_sample_weight(
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Balance coarse targets while optionally balancing mechanisms within them.

    Directly balancing all eleven fine labels assigns far less total weight to
    healthy than to the eight switch subclasses.  This helper first balances
    the five deployment classes, then normalizes any within-class fine-label
    weights to mean one so every coarse class keeps the same total influence.
    """

    coarse = np.asarray(coarse_labels, dtype=int)
    weights = compute_sample_weight("balanced", coarse).astype(float)
    if fine_labels is None:
        return weights

    fine = np.asarray(fine_labels, dtype=int)
    if fine.shape != coarse.shape:
        raise ValueError("fine_labels must have the same shape as coarse_labels")
    for coarse_class in np.unique(coarse):
        mask = coarse == coarse_class
        within = compute_sample_weight("balanced", fine[mask]).astype(float)
        weights[mask] *= within / within.mean()
    return weights


DEFAULT_DATA = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "combined_development_expanded_v13"
    / "feature_dataset.csv"
)
DEFAULT_OUTPUT = ML_ROOT / "results" / "event_baseline_v14"
LABELS = (0, 1, 2, 3, 4)
LABEL_NAMES = {
    0: "healthy",
    1: "vbus_sensor_bias",
    2: "inductor_current_sensor_bias",
    3: "switch_S1_fault",
    4: "switch_S2_fault",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--events-cache",
        type=Path,
        help="Reuse a previously generated event_features.csv instead of re-reading windows.",
    )
    parser.add_argument(
        "--event-index",
        type=Path,
        help="Optional event_index.csv supplying ModeCommand/context for a cache.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            "random_forest",
            "xgboost",
            "extra_trees",
            "mlp",
            "knn",
            "logistic_regression",
            "cnn_1d",
        ),
        default=("random_forest", "xgboost"),
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=240803)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument(
        "--fine-labels",
        action="store_true",
        help="Train eleven mechanism/location labels and collapse probabilities to five classes.",
    )
    parser.add_argument("--target-healthy-far", type=float, default=0.05)
    parser.add_argument(
        "--exclude-mechanisms",
        nargs="*",
        default=(),
        help="Research scope only; excluded mechanisms remain documented as unresolved.",
    )
    parser.add_argument(
        "--active-switch-scope",
        action="store_true",
        help=(
            "Evaluate switch location only while that switch is electrically active: "
            "S1 in Mode1 and S2 in Mode2. Other states require temporal excitation."
        ),
    )
    return parser.parse_args()


def write_progress(output: Path, **payload: object) -> None:
    data = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    (output / "progress.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_windows(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    columns = required_event_columns(header.columns, DEFAULT_EVENT_SIGNALS)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def make_model(name: str, seed: int, class_count: int) -> Pipeline:
    preprocessing: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=5.0,
            reg_alpha=0.05,
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            n_jobs=8,
            random_state=seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "mlp":
        preprocessing.extend(
            [
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=128)),
                ("scale", StandardScaler()),
            ]
        )
        estimator = MLPClassifier(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )
    elif name == "knn":
        preprocessing.extend(
            [
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=64)),
                ("scale", StandardScaler()),
            ]
        )
        estimator = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="manhattan",
            n_jobs=-1,
        )
    elif name == "logistic_regression":
        preprocessing.extend(
            [
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif, k=128)),
                ("scale", StandardScaler()),
            ]
        )
        estimator = LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        )
    elif name == "cnn_1d":
        preprocessing.extend(
            [
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
            ]
        )
        estimator = NumpyConv1DClassifier(random_state=seed)
    else:
        raise ValueError(f"Unsupported baseline model: {name}")
    return Pipeline([*preprocessing, ("model", estimator)])


def threshold_prediction(probability: np.ndarray, threshold: float) -> np.ndarray:
    fault_class = np.argmax(probability[:, 1:], axis=1) + 1
    return np.where(1.0 - probability[:, 0] >= threshold, fault_class, 0)


def apply_switch_mode_gate(
    probability: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Mask switch locations that are not electrically observable in the mode."""

    if "ModeCommand" not in frame:
        raise ValueError("ModeCommand is required for --active-switch-scope.")
    gated = np.asarray(probability, dtype=float).copy()
    modes = pd.to_numeric(frame["ModeCommand"], errors="raise").to_numpy(dtype=int)
    invalid_s1 = modes != 1
    invalid_s2 = modes != 2
    # An impossible switch hypothesis means "not observable in this mode".
    # Transfer that mass to healthy/abstain rather than amplifying another tiny
    # fault probability by renormalizing after a hard mask.
    gated[invalid_s1, 0] += gated[invalid_s1, 3]
    gated[invalid_s2, 0] += gated[invalid_s2, 4]
    gated[invalid_s1, 3] = 0.0
    gated[invalid_s2, 4] = 0.0
    row_sum = gated.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Mode gating removed all class probability from a row.")
    return gated / row_sum


def worst_operating_point_far(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    healthy = frame["WindowFaultID"].to_numpy(dtype=int) == 0
    values: list[float] = []
    for operating_point in frame.loc[healthy, "OperatingPointID"].unique():
        mask = healthy & frame["OperatingPointID"].eq(operating_point).to_numpy()
        values.append(float(np.mean(prediction[mask] != 0)))
    return max(values, default=0.0)


def choose_threshold(
    probability: np.ndarray,
    frame: pd.DataFrame,
    target_far: float,
) -> tuple[float, pd.DataFrame]:
    labels = frame["WindowFaultID"].to_numpy(dtype=int)
    healthy = labels == 0
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.995, 190):
        prediction = threshold_prediction(probability, float(threshold))
        rows.append(
            {
                "Threshold": float(threshold),
                "MacroF1": float(
                    f1_score(labels, prediction, average="macro", zero_division=0)
                ),
                "HealthyFAR": float(np.mean(prediction[healthy] != 0)),
                "WorstOperatingPointFAR": worst_operating_point_far(
                    frame, prediction
                ),
                "MinimumFaultRecall": min(
                    float(np.mean(prediction[labels == label] == label))
                    for label in LABELS[1:]
                ),
            }
        )
    trials = pd.DataFrame(rows)
    feasible = trials.loc[
        trials["HealthyFAR"].le(target_far + 1e-12)
        & trials["WorstOperatingPointFAR"].le(target_far + 1e-12)
    ]
    ranking = feasible if not feasible.empty else trials
    best = ranking.sort_values(
        ["MinimumFaultRecall", "MacroF1", "WorstOperatingPointFAR"],
        ascending=[False, False, True],
    ).index[0]
    trials["Selected"] = trials.index == best
    return float(trials.loc[best, "Threshold"]), trials


def aggregate_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, float]:
    healthy = labels == 0
    return {
        "Accuracy": float(accuracy_score(labels, prediction)),
        "BalancedAccuracy": float(balanced_accuracy_score(labels, prediction)),
        "MacroF1": float(
            f1_score(labels, prediction, average="macro", zero_division=0)
        ),
        "HealthyFalseAlarmRate": float(np.mean(prediction[healthy] != 0)),
        "WorstOperatingPointFAR": worst_operating_point_far(frame, prediction),
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_progress(args.output, state="loading", fold=0, model=None)
    if args.events_cache is not None:
        events = pd.read_csv(args.events_cache)
        features = [name for name in events.columns if "__" in name]
        if not features:
            raise ValueError("No aggregate features found in --events-cache.")
        source = args.events_cache.resolve()
        if args.event_index is not None:
            event_index = pd.read_csv(args.event_index)
            context_names = [
                name
                for name in ("ModeCommand", "SOCInit", "Pload")
                if name in event_index and name not in events
            ]
            events = events.merge(
                event_index[["RunID", *context_names]],
                on="RunID",
                how="left",
                validate="one_to_one",
            )
            if args.include_context:
                features.extend(
                    name
                    for name in context_names
                    if pd.api.types.is_numeric_dtype(events[name])
                )
        elif args.include_context:
            raise ValueError(
                "--include-context with a cache requires --event-index."
            )
    else:
        windows = load_windows(args.data)
        events, features = build_event_dataset(
            windows, include_context=args.include_context
        )
        del windows
        source = args.data.resolve()

    excluded_mechanisms = tuple(sorted(set(args.exclude_mechanisms)))
    full_event_rows = len(events)
    if excluded_mechanisms:
        events = events.loc[
            ~events["FaultMechanism"].isin(excluded_mechanisms)
        ].copy()
    inactive_switch_rows = 0
    if args.active_switch_scope:
        if "ModeCommand" not in events:
            raise ValueError(
                "--active-switch-scope requires ModeCommand; provide --event-index."
            )
        mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
        inactive_switch = (
            events["WindowFaultID"].eq(3) & mode.ne(1)
        ) | (
            events["WindowFaultID"].eq(4) & mode.ne(2)
        )
        inactive_switch_rows = int(inactive_switch.sum())
        events = events.loc[~inactive_switch].copy()
    missing_labels = sorted(set(LABELS).difference(events["WindowFaultID"].unique()))
    if missing_labels:
        raise ValueError(
            f"Research scope removed required deployment classes: {missing_labels}"
        )

    if args.active_switch_scope:
        qualification_scope = "active_switch_observable"
    elif excluded_mechanisms:
        qualification_scope = "observable_mechanisms"
    else:
        qualification_scope = "all_mechanisms"

    profile = {
        "source": str(source),
        "full_event_rows_before_scope_filter": int(full_event_rows),
        "excluded_mechanisms": list(excluded_mechanisms),
        "qualification_scope": qualification_scope,
        "active_switch_scope": bool(args.active_switch_scope),
        "inactive_switch_rows_removed": inactive_switch_rows,
        "event_rows": int(len(events)),
        "run_ids": int(events["RunID"].nunique()),
        "operating_points": int(events["OperatingPointID"].nunique()),
        "feature_count": int(len(features)),
        "include_context": bool(args.include_context),
        "class_counts": {
            str(key): int(value)
            for key, value in events["WindowFaultID"].value_counts().sort_index().items()
        },
        "mechanism_counts": {
            str(key): int(value)
            for key, value in events["FaultMechanism"].value_counts().items()
        },
        "duplicate_run_ids": int(events["RunID"].duplicated().sum()),
        "missing_feature_cells": int(events[features].isna().sum().sum()),
    }
    (args.output / "data_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    events.loc[
        :, [
            "RunID",
            "OperatingPointID",
            "FaultName",
            "FaultMechanism",
            "WindowFaultID",
            "FineFaultID",
            *[name for name in ("ModeCommand", "SOCInit", "Pload") if name in events],
        ],
    ].to_csv(args.output / "event_index.csv", index=False)
    events.loc[
        :,
        [
            "RunID",
            "OperatingPointID",
            "FaultName",
            "FaultMechanism",
            "WindowFaultID",
            "FineFaultID",
            *features,
        ],
    ].to_csv(args.output / "event_features.csv", index=False)

    folds = constrained_group_kfolds(
        events,
        n_splits=args.folds,
        random_seed=args.seed,
    )
    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []

    for fold in folds:
        for model_index, model_name in enumerate(args.models, start=1):
            write_progress(
                args.output,
                state="training",
                fold=fold.fold,
                total_folds=len(folds),
                model=model_name,
                model_index=model_index,
            )
            training_label = "FineFaultID" if args.fine_labels else "WindowFaultID"
            class_count = 11 if args.fine_labels else len(LABELS)
            model = make_model(
                model_name,
                args.seed + fold.fold * 100 + model_index,
                class_count,
            )
            x_train = fold.train[features]
            x_validation = fold.validation[features]
            x_test = fold.test[features]
            y_train = fold.train[training_label].to_numpy(dtype=int)
            y_validation = fold.validation["WindowFaultID"].to_numpy(dtype=int)
            y_test = fold.test["WindowFaultID"].to_numpy(dtype=int)
            coarse_train = fold.train["WindowFaultID"].to_numpy(dtype=int)
            weights = hierarchical_sample_weight(
                coarse_train,
                y_train if args.fine_labels else None,
            )
            start = time.perf_counter()
            fit_parameters = inspect.signature(
                model.named_steps["model"].fit
            ).parameters
            if "sample_weight" in fit_parameters:
                model.fit(x_train, y_train, model__sample_weight=weights)
            else:
                model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - start
            validation_probability = model.predict_proba(x_validation)
            test_probability = model.predict_proba(x_test)
            if args.fine_labels:
                learned_classes = model.named_steps["model"].classes_
                validation_probability = fine_probability_to_coarse(
                    validation_probability, learned_classes
                )
                test_probability = fine_probability_to_coarse(
                    test_probability, learned_classes
                )
            if args.active_switch_scope:
                validation_probability = apply_switch_mode_gate(
                    validation_probability, fold.validation
                )
                test_probability = apply_switch_mode_gate(
                    test_probability, fold.test
                )
            reported_model_name = (
                f"{model_name}_fine" if args.fine_labels else model_name
            )
            threshold, threshold_trials = choose_threshold(
                validation_probability,
                fold.validation,
                args.target_healthy_far,
            )
            threshold_trials.insert(0, "Fold", fold.fold)
            threshold_trials.insert(1, "Model", reported_model_name)
            threshold_rows.append(threshold_trials)

            variants = {
                "argmax": np.argmax(test_probability, axis=1),
                "robust_threshold": threshold_prediction(test_probability, threshold),
            }
            for variant, prediction in variants.items():
                metrics = aggregate_metrics(y_test, prediction, fold.test)
                metric_rows.append(
                    {
                        "Fold": fold.fold,
                        "Model": reported_model_name,
                        "Variant": variant,
                        "FeatureCount": len(features),
                        "TrainRunCount": len(fold.train),
                        "ValidationRunCount": len(fold.validation),
                        "TestRunCount": len(fold.test),
                        "TestOperatingPointIDs": "|".join(fold.test_groups),
                        "AlarmThreshold": threshold,
                        "TrainingSeconds": training_seconds,
                        **metrics,
                    }
                )
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test,
                    prediction,
                    labels=LABELS,
                    zero_division=0,
                )
                for position, label in enumerate(LABELS):
                    per_class_rows.append(
                        {
                            "Fold": fold.fold,
                            "Model": reported_model_name,
                            "Variant": variant,
                            "ClassID": label,
                            "ClassName": LABEL_NAMES[label],
                            "Precision": float(precision[position]),
                            "Recall": float(recall[position]),
                            "F1": float(f1[position]),
                            "Support": int(support[position]),
                        }
                    )
                diagnostic = fold.test.loc[
                    :, ["FaultMechanism", "WindowFaultID"]
                ].copy()
                diagnostic["Prediction"] = prediction
                for mechanism, part in diagnostic.groupby(
                    "FaultMechanism", observed=True
                ):
                    scenario_rows.append(
                        {
                            "Fold": fold.fold,
                            "Model": reported_model_name,
                            "Variant": variant,
                            "FaultMechanism": mechanism,
                            "Support": len(part),
                            "Recall": float(
                                np.mean(
                                    part["Prediction"].to_numpy()
                                    == part["WindowFaultID"].to_numpy()
                                )
                            ),
                        }
                    )
                output = fold.test.loc[
                    :,
                    [
                        "RunID",
                        "OperatingPointID",
                        "FaultName",
                        "FaultMechanism",
                        "WindowFaultID",
                    ],
                ].copy()
                output.insert(0, "Fold", fold.fold)
                output.insert(1, "Model", reported_model_name)
                output.insert(2, "Variant", variant)
                output["PredictedClassID"] = prediction
                for label in LABELS:
                    output[f"ProbabilityClass{label}"] = test_probability[:, label]
                prediction_rows.append(output)

            pd.DataFrame(metric_rows).to_csv(
                args.output / "fold_metrics_partial.csv", index=False
            )

    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows)
    scenarios = pd.DataFrame(scenario_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    per_class.to_csv(args.output / "per_class_metrics.csv", index=False)
    scenarios.to_csv(args.output / "scenario_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        args.output / "threshold_trials.csv", index=False
    )

    summary = (
        metrics.groupby(["Model", "Variant"], sort=True)
        .agg(
            FoldCount=("Fold", "nunique"),
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            MacroF1Min=("MacroF1", "min"),
            BalancedAccuracyMean=("BalancedAccuracy", "mean"),
            HealthyFARMean=("HealthyFalseAlarmRate", "mean"),
            WorstOperatingPointFARMax=("WorstOperatingPointFAR", "max"),
            TrainingSecondsMean=("TrainingSeconds", "mean"),
        )
        .reset_index()
    )
    class_summary = (
        per_class.groupby(["Model", "Variant", "ClassID", "ClassName"], sort=True)
        .agg(RecallMean=("Recall", "mean"), F1Mean=("F1", "mean"))
        .reset_index()
    )
    switch_recall = (
        class_summary.loc[class_summary["ClassID"].isin([3, 4])]
        .groupby(["Model", "Variant"])["RecallMean"]
        .min()
        .rename("MinimumSwitchRecallMean")
        .reset_index()
    )
    summary = summary.merge(switch_recall, on=["Model", "Variant"], how="left")
    summary.insert(
        0,
        "QualificationScope",
        qualification_scope,
    )
    summary.insert(1, "ExcludedMechanisms", "|".join(excluded_mechanisms))
    summary["ProvisionalQualified"] = (
        summary["MacroF1Mean"].ge(0.70)
        & summary["HealthyFARMean"].le(0.05)
        & summary["WorstOperatingPointFARMax"].le(0.10)
        & summary["MinimumSwitchRecallMean"].ge(0.60)
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    class_summary.to_csv(args.output / "per_class_summary.csv", index=False)
    write_progress(
        args.output,
        state="completed",
        fold=len(folds),
        model=None,
        qualified_models=int(summary["ProvisionalQualified"].sum()),
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
