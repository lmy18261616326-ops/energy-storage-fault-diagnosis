#!/usr/bin/env python
"""Test a commissioning-style healthy-reference feature for high resistance.

Each operating point must have at least two known-healthy runs.  Fault runs are
represented relative to the median healthy fingerprint at the same operating
point; healthy evaluation rows use a leave-one-out mean fingerprint.  This is
deliberately an *assumption test*: good performance means the fault is observable
when a commissioned healthy reference is available, not that a standalone
classifier can generalize to an unseen operating point without calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.data import constrained_group_kfolds
from diagnose_high_resistance_separability import choose_threshold, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=240803)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("extra_trees", "logistic_regression"),
        choices=("random_forest", "extra_trees", "xgboost", "logistic_regression"),
    )
    parser.add_argument("--active-switch-scope", action="store_true")
    return parser.parse_args()


def healthy_reference_features(
    events: pd.DataFrame,
    source_features: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    selected = events.loc[
        events["FaultMechanism"].isin(["healthy", "high_resistance"])
    ].copy()
    selected[source_features] = selected[source_features].apply(
        pd.to_numeric, errors="coerce"
    )
    coverage = (
        selected.loc[selected["FaultMechanism"].eq("healthy")]
        .groupby("OperatingPointID", sort=True)
        .size()
        .rename("HealthyReferenceCount")
        .reset_index()
    )
    missing = sorted(set(selected["OperatingPointID"]) - set(coverage["OperatingPointID"]))
    if missing:
        raise ValueError(f"Operating points without a healthy reference: {missing}")
    sparse = coverage.loc[coverage["HealthyReferenceCount"].lt(2)]
    if not sparse.empty:
        raise ValueError(
            "Leave-one-out FAR evaluation needs at least two healthy references per "
            f"operating point: {sparse.to_dict(orient='records')}"
        )

    values = selected[source_features].to_numpy(dtype=float)
    baseline = np.full_like(values, np.nan, dtype=float)
    healthy_mask = selected["FaultMechanism"].eq("healthy").to_numpy()
    operating_points = selected["OperatingPointID"].astype(str).to_numpy()

    for operating_point in np.unique(operating_points):
        group_rows = np.flatnonzero(operating_points == operating_point)
        healthy_rows = group_rows[healthy_mask[group_rows]]
        healthy_values = values[healthy_rows]
        group_reference = np.nanmedian(healthy_values, axis=0)
        baseline[group_rows] = group_reference
        for row in healthy_rows:
            other_rows = healthy_rows[healthy_rows != row]
            baseline[row] = np.nanmean(values[other_rows], axis=0)

    delta = values - baseline
    residual_names = [f"HealthyRefDelta__{name}" for name in source_features]
    absolute_names = [f"HealthyRefAbsDelta__{name}" for name in source_features]
    residual_frame = pd.DataFrame(
        np.concatenate([delta, np.abs(delta)], axis=1),
        columns=[*residual_names, *absolute_names],
        index=selected.index,
    )
    selected = pd.concat([selected, residual_frame], axis=1).copy()
    selected["BinaryLabel"] = (
        selected["FaultMechanism"].eq("high_resistance").astype(int)
    )
    return selected, [*residual_names, *absolute_names], coverage


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    fold_source = events.copy()
    if args.event_index is not None:
        event_index = pd.read_csv(args.event_index, usecols=["RunID", "ModeCommand"])
        events = events.merge(
            event_index, on="RunID", how="left", validate="one_to_one"
        )
    if args.active_switch_scope:
        if "ModeCommand" not in events:
            raise ValueError("--active-switch-scope requires --event-index")
        mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
        eligible = (
            events["FaultMechanism"].eq("healthy") & mode.isin([1, 2])
        ) | (
            events["FineFaultID"].eq(9) & mode.eq(1)
        ) | (
            events["FineFaultID"].eq(10) & mode.eq(2)
        )
        events = events.loc[eligible].copy()
    source_features = [name for name in events.columns if "__" in name]
    if not source_features:
        raise ValueError("No event aggregate features found.")
    paired, features, coverage = healthy_reference_features(events, source_features)
    coverage.to_csv(args.output / "healthy_reference_coverage.csv", index=False)
    # Construct folds from the complete five-class table so the established
    # constrained splitter can enforce all class/operating-point requirements.
    # The binary assumption test then selects matching RunIDs from each fold.
    folds = constrained_group_kfolds(
        fold_source, n_splits=args.folds, random_seed=args.seed
    )
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for fold in folds:
        fold_parts = {
            "train": paired.loc[paired["RunID"].isin(fold.train["RunID"])].copy(),
            "validation": paired.loc[
                paired["RunID"].isin(fold.validation["RunID"])
            ].copy(),
            "test": paired.loc[paired["RunID"].isin(fold.test["RunID"])].copy(),
        }
        for model_index, model_name in enumerate(args.models, start=1):
            seed = args.seed + fold.fold * 100 + model_index
            model = make_model(model_name, len(features), seed)
            train = fold_parts["train"]
            validation = fold_parts["validation"]
            test = fold_parts["test"]
            weights = compute_sample_weight("balanced", train["BinaryLabel"])
            model.fit(
                train[features],
                train["BinaryLabel"],
                model__sample_weight=weights,
            )
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
            metrics.append(
                {
                    "Fold": fold.fold,
                    "Model": model_name,
                    "Threshold": threshold,
                    "ROC_AUC": float(roc_auc_score(labels, test_probability)),
                    "PR_AUC": float(average_precision_score(labels, test_probability)),
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
                ],
            ].copy()
            output.insert(0, "Fold", fold.fold)
            output.insert(1, "Model", model_name)
            output["ProbabilityHighResistance"] = test_probability
            output["Prediction"] = prediction
            predictions.append(output)

    fold_metrics = pd.DataFrame(metrics)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )
    summary = (
        fold_metrics.groupby("Model", sort=True)
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
                "source_feature_count": len(source_features),
                "paired_feature_count": len(features),
                "folds": args.folds,
                "models": list(args.models),
                "assumption": (
                    "At inference, each operating point has a commissioned healthy "
                    "reference; healthy evaluation uses leave-one-out references."
                ),
                "active_switch_scope": bool(args.active_switch_scope),
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
