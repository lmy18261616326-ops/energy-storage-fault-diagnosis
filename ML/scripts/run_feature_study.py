#!/usr/bin/env python
"""Six-fold, leakage-safe feature-count and suspect-family ablation study."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.data import constrained_group_kfolds, load_feature_dataset
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model


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
DEFAULT_OUTPUT = ML_ROOT / "results" / "feature_study"
FEATURE_COUNTS = (30, 50, 80, 226)
LABELS = (0, 1, 2, 3, 4)
LABEL_NAMES = {
    0: "healthy",
    1: "vbus_sensor_bias",
    2: "inductor_current_sensor_bias",
    3: "switch_S1_open",
    4: "switch_S2_open",
}
TARGET_OPERATING_POINTS = {"op_0002", "op_0006", "op_0009"}
ABLATIONS = {
    "all_features": (),
    "without_current_error": ("CurrentError",),
    "without_ibat_dynamics": (
        "Ibat_measStd",
        "Ibat_measRange",
        "Ibat_measSlope",
        "Ibat_measDelta",
        "Ibat_measDiffRMS",
    ),
    "without_suspect_families": (
        "CurrentError",
        "Ibat_measStd",
        "Ibat_measRange",
        "Ibat_measSlope",
        "Ibat_measDelta",
        "Ibat_measDiffRMS",
        "CurrentPairResidual",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("random_forest", "xgboost"),
        default=("random_forest", "xgboost"),
    )
    parser.add_argument("--seed", type=int, default=240727)
    return parser.parse_args()


def model_parameters() -> dict[str, dict[str, object]]:
    return {
        "random_forest": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
            "n_jobs": -1,
        },
    }


def rank_features(
    train: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> pd.DataFrame:
    matrix = feature_matrix(train, columns)
    imputed = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    ).fit_transform(matrix)
    labels = train["WindowFaultID"].to_numpy(dtype=int)
    if len(labels) > 20000:
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(len(labels), size=20000, replace=False))
        imputed = imputed[sample]
        labels = labels[sample]
    score = mutual_info_classif(
        imputed,
        labels,
        discrete_features=False,
        random_state=seed,
    )
    return pd.DataFrame(
        {"Feature": columns, "MutualInformation": score}
    ).sort_values(
        ["MutualInformation", "Feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def evaluate_variant(
    *,
    fold: object,
    model_name: str,
    columns: list[str],
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    x_train = feature_matrix(fold.train, columns)
    x_validation = feature_matrix(fold.validation, columns)
    x_test = feature_matrix(fold.test, columns)
    y_train = fold.train["WindowFaultID"].to_numpy(dtype=int)
    y_validation = fold.validation["WindowFaultID"].to_numpy(dtype=int)
    y_test = fold.test["WindowFaultID"].to_numpy(dtype=int)
    result = train_model(
        model_name,
        x_train,
        y_train,
        x_validation,
        y_validation,
        feature_columns=columns,
        model_parameters=model_parameters()[model_name],
        random_seed=seed,
        tune=False,
        tuning_grid={},
    )
    prediction = result.artifact.predict(x_test)
    aggregate, _, _ = evaluate_classification(
        y_test,
        prediction,
        labels=LABELS,
        label_names=LABEL_NAMES,
    )
    diagnostics = fold.test.loc[
        :,
        ["RunID", "OperatingPointID", "WindowFaultID"],
    ].copy()
    diagnostics["Prediction"] = prediction
    return aggregate, diagnostics


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

    metric_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for fold in folds:
        usable, _ = remove_unusable_training_features(
            fold.train,
            candidates,
            remove_constant=True,
        )
        ranking = rank_features(fold.train, usable, args.seed + fold.fold)
        ranked = ranking["Feature"].tolist()
        for rank, row in ranking.iterrows():
            selection_rows.append(
                {
                    "Fold": fold.fold,
                    "Rank": rank + 1,
                    **row.to_dict(),
                }
            )

        variants: dict[str, list[str]] = {}
        for count in FEATURE_COUNTS:
            variants[f"top_{count}"] = ranked[: min(count, len(ranked))]
        for name, prefixes in ABLATIONS.items():
            variants[name] = [
                feature
                for feature in usable
                if not any(feature.startswith(prefix) for prefix in prefixes)
            ]

        for variant_name, columns in variants.items():
            for model_name in args.models:
                aggregate, diagnostics = evaluate_variant(
                    fold=fold,
                    model_name=model_name,
                    columns=columns,
                    seed=args.seed + fold.fold,
                )
                metric_rows.append(
                    {
                        "Fold": fold.fold,
                        "Model": model_name,
                        "Variant": variant_name,
                        "FeatureCount": len(columns),
                        "TestOperatingPointIDs": "|".join(fold.test_groups),
                        **aggregate,
                    }
                )
                healthy = diagnostics.loc[
                    diagnostics["WindowFaultID"].eq(0)
                    & diagnostics["OperatingPointID"].isin(
                        TARGET_OPERATING_POINTS
                    )
                ]
                for operating_point, part in healthy.groupby(
                    "OperatingPointID",
                    sort=True,
                ):
                    target_rows.append(
                        {
                            "Fold": fold.fold,
                            "Model": model_name,
                            "Variant": variant_name,
                            "OperatingPointID": operating_point,
                            "HealthyWindowCount": len(part),
                            "FalseAlarmCount": int(
                                part["Prediction"].ne(0).sum()
                            ),
                            "FalseAlarmRate": float(
                                part["Prediction"].ne(0).mean()
                            ),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        args.output / "target_false_alarm_rates.csv",
        index=False,
    )
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(args.output / "fold_feature_rankings.csv", index=False)

    summary = (
        metrics.groupby(["Model", "Variant", "FeatureCount"], sort=True)
        .agg(
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            MacroF1Min=("MacroF1", "min"),
            BalancedAccuracyMean=("BalancedAccuracy", "mean"),
            HealthyFalseAlarmRateMean=("HealthyFalseAlarmRate", "mean"),
            HealthyFalseAlarmRateStd=("HealthyFalseAlarmRate", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)

    stability_rows: list[dict[str, object]] = []
    by_fold = {
        fold: part.sort_values("Rank")["Feature"].tolist()
        for fold, part in selections.groupby("Fold")
    }
    for count in FEATURE_COUNTS:
        similarities = []
        for left, right in combinations(sorted(by_fold), 2):
            a = set(by_fold[left][:count])
            b = set(by_fold[right][:count])
            similarities.append(len(a & b) / len(a | b))
        stability_rows.append(
            {
                "FeatureCount": count,
                "MeanPairwiseJaccard": float(np.mean(similarities)),
                "StdPairwiseJaccard": float(np.std(similarities, ddof=1)),
                "MinPairwiseJaccard": float(np.min(similarities)),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        args.output / "selection_stability.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
