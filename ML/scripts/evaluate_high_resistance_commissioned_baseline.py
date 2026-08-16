#!/usr/bin/env python
"""Pilot evaluation with a commissioned healthy fingerprint per operating point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from diagnose_paired_high_resistance import healthy_reference_features
from evaluate_high_resistance_excitation_pilot import (
    choose_threshold,
    make_model,
    worst_group_far,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=816701)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    all_features = [name for name in events.columns if "__" in name]
    step_features = [name for name in all_features if "__load_step_" in name]
    base_features = [name for name in all_features if name not in step_features]
    feature_sources = {
        "commissioned_base_event": base_features,
        "commissioned_event_plus_step": all_features,
    }
    paired_sets = {}
    for name, source_features in feature_sources.items():
        paired, features, coverage = healthy_reference_features(
            events, source_features
        )
        paired_sets[name] = (paired, features)
        coverage.to_csv(args.output / f"{name}_coverage.csv", index=False)

    groups = np.asarray(sorted(events["OperatingPointID"].unique()))
    group_folds = [np.asarray(part) for part in np.array_split(groups, args.folds)]
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
        for feature_set, (paired, features) in paired_sets.items():
            test_mask = paired["OperatingPointID"].isin(test_groups)
            validation_mask = paired["OperatingPointID"].isin(validation_groups)
            train_mask = ~(test_mask | validation_mask)
            train = paired.loc[train_mask]
            validation = paired.loc[validation_mask]
            test = paired.loc[test_mask]
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
                magnitude = test["FaultMagnitude"].to_numpy(dtype=float)
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
                        "RecallRon0.02": float(
                            np.mean(prediction[fault & np.isclose(magnitude, 0.02)])
                        ),
                        "RecallRon0.10": float(
                            np.mean(prediction[fault & np.isclose(magnitude, 0.10)])
                        ),
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
        "deployment_assumption": (
            "A commissioned healthy fingerprint is available for every operating point."
        ),
        "qualified_candidates": int(summary["PilotQualified"].sum()),
        "proceed_to_full_expansion": bool(summary["PilotQualified"].any()),
    }
    (args.output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n" + json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
