#!/usr/bin/env python
"""Evaluate separate S1/Mode1 and S2/Mode2 high-resistance experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from evaluate_high_resistance_excitation_pilot import (
    choose_threshold,
    make_model,
    worst_group_far,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=816601)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    all_features = [name for name in events.columns if "__" in name]
    step_features = [name for name in all_features if "__load_step_" in name]
    base_features = [name for name in all_features if name not in step_features]
    feature_sets = {
        "base_event": base_features,
        "event_plus_load_step": all_features,
    }
    model_names = (
        "extra_trees",
        "random_forest",
        "xgboost",
        "logistic_regression",
    )
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for mode in (1, 2):
        scoped = events.loc[events["ModeCommand"].eq(mode)].copy()
        groups = np.asarray(sorted(scoped["OperatingPointID"].unique()))
        if len(groups) != 4:
            raise ValueError(f"Mode{mode} expected four operating points, found {len(groups)}")
        for fold_index in range(4):
            test_group = groups[fold_index]
            validation_group = groups[(fold_index + 1) % 4]
            test = scoped.loc[scoped["OperatingPointID"].eq(test_group)]
            validation = scoped.loc[
                scoped["OperatingPointID"].eq(validation_group)
            ]
            train = scoped.loc[
                ~scoped["OperatingPointID"].isin([test_group, validation_group])
            ]
            for feature_set, features in feature_sets.items():
                for model_index, model_name in enumerate(model_names, start=1):
                    model = make_model(
                        model_name,
                        args.seed + mode * 1000 + fold_index * 100 + model_index,
                    )
                    weights = compute_sample_weight("balanced", train["BinaryLabel"])
                    model.fit(
                        train[features],
                        train["BinaryLabel"],
                        model__sample_weight=weights,
                    )
                    validation_probability = model.predict_proba(
                        validation[features]
                    )[:, 1]
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
                            test["FaultMagnitude"].to_numpy(dtype=float),
                            magnitude_value,
                        )
                        magnitude_recalls[f"RecallRon{magnitude_value:.2f}"] = float(
                            np.mean(prediction[mask])
                        )
                    rows.append(
                        {
                            "ModeCommand": mode,
                            "Fold": fold_index + 1,
                            "FeatureSet": feature_set,
                            "Model": model_name,
                            "Threshold": threshold,
                            "ROC_AUC": float(roc_auc_score(labels, probability)),
                            "PR_AUC": float(
                                average_precision_score(labels, probability)
                            ),
                            "F1": float(
                                f1_score(labels, prediction, zero_division=0)
                            ),
                            "HealthyFAR": float(np.mean(prediction[healthy])),
                            "WorstOperatingPointFAR": worst_group_far(
                                labels, prediction, test["OperatingPointID"]
                            ),
                            "HighResistanceRecall": float(
                                np.mean(prediction[fault])
                            ),
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
                    output.insert(0, "ModeCommand", mode)
                    output.insert(1, "Fold", fold_index + 1)
                    output.insert(2, "FeatureSet", feature_set)
                    output.insert(3, "Model", model_name)
                    output["ProbabilityHighResistance"] = probability
                    output["Prediction"] = prediction
                    prediction_rows.append(output)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    by_mode = (
        metrics.groupby(["ModeCommand", "FeatureSet", "Model"], sort=True)
        .agg(
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
    by_mode.to_csv(args.output / "summary_by_mode.csv", index=False)
    overall = (
        metrics.groupby(["FeatureSet", "Model"], sort=True)
        .agg(
            FoldCount=("Fold", "count"),
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
    overall["PilotQualified"] = (
        overall["ROC_AUC_Mean"].ge(0.80)
        & overall["HighResistanceRecallMean"].ge(0.60)
        & overall["HighResistanceRecallMin"].ge(0.25)
        & overall["HealthyFARMean"].le(0.05)
        & overall["WorstOperatingPointFARMax"].le(0.25)
    )
    overall.to_csv(args.output / "summary.csv", index=False)
    decision = {
        "qualified_candidates": int(overall["PilotQualified"].sum()),
        "proceed_to_full_expansion": bool(overall["PilotQualified"].any()),
    }
    (args.output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\n" + json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
