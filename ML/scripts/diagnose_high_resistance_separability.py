#!/usr/bin/env python
"""Measure whether high-resistance events are separable from healthy events.

The script uses the cached event feature table produced by
``run_event_model_baselines.py``.  It is a diagnostic, not a final model-selection
result: thresholds are chosen on validation operating points and reported on
held-out operating points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.data import constrained_group_kfolds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--event-index",
        type=Path,
        help="Optional event_index.csv supplying ModeCommand for observability filtering.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=240803)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("random_forest", "extra_trees", "xgboost", "logistic_regression"),
        choices=("random_forest", "extra_trees", "xgboost", "logistic_regression"),
    )
    parser.add_argument(
        "--active-switch-scope",
        action="store_true",
        help="Keep healthy Mode1/2 and high resistance only on S1/Mode1 or S2/Mode2.",
    )
    return parser.parse_args()


def make_model(name: str, feature_count: int, seed: int) -> Pipeline:
    select_count = min(128, feature_count)
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=select_count)),
            ("model", estimator),
        ]
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=select_count)),
            ("model", estimator),
        ]
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=8.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=8,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=select_count)),
            ("model", estimator),
        ]
    elif name == "logistic_regression":
        estimator = LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", RobustScaler()),
            ("select", SelectKBest(f_classif, k=select_count)),
            ("model", estimator),
        ]
    else:
        raise ValueError(name)
    return Pipeline(steps)


def choose_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
    operating_points: pd.Series,
    target_far: float = 0.05,
) -> float:
    healthy = labels == 0
    rows: list[tuple[float, float, float, float]] = []
    for threshold in np.linspace(0.05, 0.995, 190):
        prediction = probability >= threshold
        far = float(np.mean(prediction[healthy]))
        worst = max(
            (
                float(np.mean(prediction[healthy & operating_points.eq(op).to_numpy()]))
                for op in operating_points[healthy].unique()
            ),
            default=0.0,
        )
        recall = float(np.mean(prediction[~healthy]))
        rows.append((float(threshold), far, worst, recall))
    feasible = [row for row in rows if row[1] <= target_far and row[2] <= target_far]
    ranking = feasible if feasible else rows
    ranking.sort(key=lambda row: (row[3], -row[2], -row[1]), reverse=True)
    return ranking[0][0]


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    if args.event_index is not None:
        event_index = pd.read_csv(args.event_index, usecols=["RunID", "ModeCommand"])
        events = events.merge(
            event_index, on="RunID", how="left", validate="one_to_one"
        )
    if args.active_switch_scope and "ModeCommand" not in events:
        raise ValueError("--active-switch-scope requires --event-index with ModeCommand")
    if args.active_switch_scope:
        mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
        active_high_resistance = (
            events["FineFaultID"].eq(9) & mode.eq(1)
        ) | (
            events["FineFaultID"].eq(10) & mode.eq(2)
        )
        active_healthy = events["FaultMechanism"].eq("healthy") & mode.isin([1, 2])
        events = events.copy()
        events["HighResistanceAuditEligible"] = (
            active_high_resistance | active_healthy
        )
    else:
        events["HighResistanceAuditEligible"] = events["FaultMechanism"].isin(
            ["healthy", "high_resistance"]
        )
    features = [name for name in events.columns if "__" in name]
    if not features:
        raise ValueError("No event aggregate features found.")
    folds = constrained_group_kfolds(events, n_splits=args.folds, random_seed=args.seed)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []

    for fold in folds:
        subsets = {
            "train": fold.train.loc[
                fold.train["HighResistanceAuditEligible"]
            ].copy(),
            "validation": fold.validation.loc[
                fold.validation["HighResistanceAuditEligible"]
            ].copy(),
            "test": fold.test.loc[
                fold.test["HighResistanceAuditEligible"]
            ].copy(),
        }
        for part in subsets.values():
            part["BinaryLabel"] = part["FaultMechanism"].eq("high_resistance").astype(int)

        for model_index, model_name in enumerate(args.models, start=1):
            model = make_model(model_name, len(features), args.seed + fold.fold * 100 + model_index)
            train = subsets["train"]
            validation = subsets["validation"]
            test = subsets["test"]
            weights = compute_sample_weight("balanced", train["BinaryLabel"])
            model.fit(train[features], train["BinaryLabel"], model__sample_weight=weights)
            validation_probability = model.predict_proba(validation[features])[:, 1]
            test_probability = model.predict_proba(test[features])[:, 1]
            threshold = choose_threshold(
                validation["BinaryLabel"].to_numpy(dtype=int),
                validation_probability,
                validation["OperatingPointID"],
            )
            labels = test["BinaryLabel"].to_numpy(dtype=int)
            prediction = (test_probability >= threshold).astype(int)
            healthy = labels == 0
            worst_far = max(
                (
                    float(
                        np.mean(
                            prediction[
                                healthy
                                & test["OperatingPointID"].eq(op).to_numpy()
                            ]
                        )
                    )
                    for op in test.loc[healthy, "OperatingPointID"].unique()
                ),
                default=0.0,
            )
            metric_rows.append(
                {
                    "Fold": fold.fold,
                    "Model": model_name,
                    "Threshold": threshold,
                    "ROC_AUC": float(roc_auc_score(labels, test_probability)),
                    "PR_AUC": float(average_precision_score(labels, test_probability)),
                    "BalancedAccuracy": float(balanced_accuracy_score(labels, prediction)),
                    "F1": float(f1_score(labels, prediction, zero_division=0)),
                    "HealthyFAR": float(np.mean(prediction[healthy])),
                    "WorstOperatingPointFAR": worst_far,
                    "HighResistanceRecall": float(np.mean(prediction[~healthy])),
                    "Support": len(test),
                }
            )
            output = test.loc[
                :,
                [
                    "RunID",
                    "OperatingPointID",
                    "FaultName",
                    "FaultMechanism",
                    "BinaryLabel",
                    *(["ModeCommand"] if "ModeCommand" in test else []),
                ],
            ].copy()
            output.insert(0, "Fold", fold.fold)
            output.insert(1, "Model", model_name)
            output["ProbabilityHighResistance"] = test_probability
            output["Prediction"] = prediction
            prediction_rows.append(output)

            variance = model.named_steps["variance"]
            selector = model.named_steps["select"]
            scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
            retained = variance.get_support(indices=True)
            selected = selector.get_support(indices=True)
            for local_index in selected:
                original_index = retained[local_index]
                importance_rows.append(
                    {
                        "Fold": fold.fold,
                        "Model": model_name,
                        "Feature": features[original_index],
                        "FScore": float(scores[local_index]),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        args.output / "selected_feature_scores.csv", index=False
    )
    summary = (
        metrics.groupby("Model", sort=True)
        .agg(
            FoldCount=("Fold", "nunique"),
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Min=("ROC_AUC", "min"),
            PR_AUC_Mean=("PR_AUC", "mean"),
            HighResistanceRecallMean=("HighResistanceRecall", "mean"),
            HighResistanceRecallMin=("HighResistanceRecall", "min"),
            HealthyFARMean=("HealthyFAR", "mean"),
            WorstOperatingPointFARMax=("WorstOperatingPointFAR", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
                "source": str(args.events.resolve()),
                "feature_count": len(features),
                "folds": args.folds,
                "models": list(args.models),
                "active_switch_scope": bool(args.active_switch_scope),
                "eligibility_rows": int(events["HighResistanceAuditEligible"].sum()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
