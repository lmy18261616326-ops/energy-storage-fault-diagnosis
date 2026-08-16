#!/usr/bin/env python
"""Grouped pilot evaluation for load-step-excited high resistance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.event_features import (
    DEFAULT_EVENT_SIGNALS,
    RUN_METADATA,
    add_physics_normalized_features,
    build_event_dataset,
    required_event_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=816501)
    parser.add_argument("--load-step-time", type=float, default=0.35)
    return parser.parse_args()


def make_model(name: str, seed: int) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
    ]
    if name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=8.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=8,
            random_state=seed,
        )
    elif name == "logistic_regression":
        steps.append(("scale", StandardScaler()))
        estimator = LogisticRegression(
            C=0.10,
            class_weight="balanced",
            max_iter=4000,
            solver="liblinear",
            random_state=seed,
        )
    else:
        raise ValueError(name)
    steps.append(("model", estimator))
    return Pipeline(steps)


def step_features(
    windows: pd.DataFrame, load_step_time: float
) -> tuple[pd.DataFrame, list[str]]:
    values = add_physics_normalized_features(windows)
    excluded = set(RUN_METADATA)
    signals = [
        name
        for name in values.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(values[name])
    ]
    pre = values.loc[
        values["WindowStart"].between(0.10, load_step_time - 0.05)
    ]
    post = values.loc[values["WindowStart"].ge(load_step_time + 0.10)]
    pre_mean = pre.groupby("RunID", sort=False)[signals].mean()
    post_mean = post.groupby("RunID", sort=False)[signals].mean()
    delta = post_mean - pre_mean
    delta.columns = [f"{name}__load_step_delta" for name in signals]
    post_mean.columns = [f"{name}__load_step_post" for name in signals]
    result = post_mean.join(delta).reset_index()
    return result, [*post_mean.columns, *delta.columns]


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    healthy = labels == 0
    trials = []
    for threshold in np.linspace(0.05, 0.995, 190):
        prediction = probability >= threshold
        far = float(np.mean(prediction[healthy]))
        recall = float(np.mean(prediction[~healthy]))
        trials.append((float(threshold), far, recall))
    feasible = [row for row in trials if row[1] <= 0.05]
    ranking = feasible if feasible else trials
    ranking.sort(key=lambda row: (row[2], -row[1]), reverse=True)
    return ranking[0][0]


def worst_group_far(
    labels: np.ndarray,
    prediction: np.ndarray,
    groups: pd.Series,
) -> float:
    healthy = labels == 0
    return max(
        (
            float(np.mean(prediction[healthy & groups.eq(group).to_numpy()]))
            for group in groups[healthy].unique()
        ),
        default=0.0,
    )


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.root.glob("*/combined/feature_dataset.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four pilot feature tables, found {len(paths)}")
    full_parts = [pd.read_csv(path) for path in paths]
    full_windows = pd.concat(full_parts, ignore_index=True)
    magnitude = full_windows.groupby("RunID", sort=False)["FaultMagnitude"].first()
    selected_columns = required_event_columns(
        full_windows.columns, DEFAULT_EVENT_SIGNALS
    )
    windows = full_windows[selected_columns].copy()
    events, base_features = build_event_dataset(windows)
    excitation, excitation_features = step_features(windows, args.load_step_time)
    events = events.merge(excitation, on="RunID", how="left", validate="one_to_one")
    events["FaultMagnitude"] = events["RunID"].map(magnitude)
    events["BinaryLabel"] = events["FaultMechanism"].eq("high_resistance").astype(int)
    events.to_csv(args.output / "pilot_event_features.csv", index=False)

    groups = np.asarray(sorted(events["OperatingPointID"].unique()))
    if len(groups) < args.folds * 2:
        raise ValueError("Need at least two operating points per test fold")
    group_folds = [np.asarray(part) for part in np.array_split(groups, args.folds)]
    feature_sets = {
        "base_event": base_features,
        "event_plus_load_step": [*base_features, *excitation_features],
    }
    model_names = (
        "extra_trees",
        "random_forest",
        "xgboost",
        "logistic_regression",
    )
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_index in range(args.folds):
        test_groups = group_folds[fold_index]
        validation_groups = group_folds[(fold_index + 1) % args.folds]
        test_mask = events["OperatingPointID"].isin(test_groups)
        validation_mask = events["OperatingPointID"].isin(validation_groups)
        train_mask = ~(test_mask | validation_mask)
        train = events.loc[train_mask]
        validation = events.loc[validation_mask]
        test = events.loc[test_mask]
        for feature_set, features in feature_sets.items():
            for model_index, model_name in enumerate(model_names, start=1):
                model = make_model(
                    model_name,
                    args.seed + fold_index * 100 + model_index,
                )
                weights = compute_sample_weight("balanced", train["BinaryLabel"])
                model.fit(
                    train[features],
                    train["BinaryLabel"],
                    model__sample_weight=weights,
                )
                validation_probability = model.predict_proba(validation[features])[:, 1]
                probability = model.predict_proba(test[features])[:, 1]
                threshold = choose_threshold(
                    validation["BinaryLabel"].to_numpy(dtype=int),
                    validation_probability,
                )
                labels = test["BinaryLabel"].to_numpy(dtype=int)
                prediction = (probability >= threshold).astype(int)
                healthy = labels == 0
                fault = labels == 1
                magnitude_recalls = {}
                for magnitude_value in (0.02, 0.10):
                    mask = fault & np.isclose(
                        test["FaultMagnitude"].to_numpy(dtype=float), magnitude_value
                    )
                    magnitude_recalls[f"RecallRon{magnitude_value:.2f}"] = (
                        float(np.mean(prediction[mask])) if mask.any() else np.nan
                    )
                rows.append(
                    {
                        "Fold": fold_index + 1,
                        "FeatureSet": feature_set,
                        "Model": model_name,
                        "Threshold": threshold,
                        "ROC_AUC": float(roc_auc_score(labels, probability)),
                        "PR_AUC": float(average_precision_score(labels, probability)),
                        "F1": float(f1_score(labels, prediction, zero_division=0)),
                        "HealthyFAR": float(np.mean(prediction[healthy])),
                        "WorstOperatingPointFAR": worst_group_far(
                            labels, prediction, test["OperatingPointID"]
                        ),
                        "HighResistanceRecall": float(np.mean(prediction[fault])),
                        **magnitude_recalls,
                    }
                )
                output = test[
                    [
                        "RunID",
                        "OperatingPointID",
                        "FaultName",
                        "FaultMagnitude",
                        "BinaryLabel",
                    ]
                ].copy()
                output.insert(0, "Fold", fold_index + 1)
                output.insert(1, "FeatureSet", feature_set)
                output.insert(2, "Model", model_name)
                output["ProbabilityHighResistance"] = probability
                output["Prediction"] = prediction
                prediction_rows.append(output)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    summary = (
        metrics.groupby(["FeatureSet", "Model"], sort=True)
        .agg(
            FoldCount=("Fold", "nunique"),
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Min=("ROC_AUC", "min"),
            HighResistanceRecallMean=("HighResistanceRecall", "mean"),
            HighResistanceRecallMin=("HighResistanceRecall", "min"),
            RecallRon002Mean=("RecallRon0.02", "mean"),
            RecallRon010Mean=("RecallRon0.10", "mean"),
            HealthyFARMean=("HealthyFAR", "mean"),
            WorstOperatingPointFARMax=("WorstOperatingPointFAR", "max"),
        )
        .reset_index()
    )
    summary["PilotQualified"] = (
        summary["ROC_AUC_Mean"].ge(0.80)
        & summary["HighResistanceRecallMean"].ge(0.60)
        & summary["HighResistanceRecallMin"].ge(0.25)
        & summary["HealthyFARMean"].le(0.05)
        & summary["WorstOperatingPointFARMax"].le(0.25)
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    decision = {
        "pilot_runs": int(events["RunID"].nunique()),
        "operating_points": int(events["OperatingPointID"].nunique()),
        "qualified_candidates": int(summary["PilotQualified"].sum()),
        "proceed_to_full_expansion": bool(summary["PilotQualified"].any()),
    }
    (args.output / "pilot_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n" + json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
