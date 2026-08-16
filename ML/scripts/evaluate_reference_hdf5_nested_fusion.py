#!/usr/bin/env python3
"""Nested grouped fusion audit for the two complementary HDF5 candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ML" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from optimize_reference_hdf5_models import (  # noqa: E402
    CANDIDATES,
    LABELS,
    SEED,
    fit_model,
    make_model,
    metrics,
    predict_aligned,
)


ML_ROOT = ROOT / "ML"
DATA_ROOT = ROOT / "output" / "reference_experiment_dataset_2026-08-04"
BASELINE_PATH = ML_ROOT / "results" / "reference_hdf5_model_evaluation_v18" / "controller_engineered_features.csv"
DELTA_PATH = ML_ROOT / "results" / "reference_hdf5_model_optimization_v19" / "differential_features.csv"
OUTPUT = ML_ROOT / "results" / "reference_hdf5_model_optimization_v19"
WEIGHTS = (0.50, 0.65, 0.80, 0.90)


def candidate(name: str):
    return next(item for item in CANDIDATES if item.name == name)


def fit_probabilities(spec, matrix, labels, train_idx, test_idx, seed):
    model = make_model(spec, matrix.shape[1], seed)
    fit_model(model, matrix[train_idx], labels[train_idx])
    _, probabilities = predict_aligned(model, matrix[test_idx])
    return probabilities


def main() -> int:
    manifest = pd.read_csv(DATA_ROOT / "run_manifest.csv")
    manifest["RunID"] = manifest["RunID"].astype(str)
    manifest["OperatingPointID"] = (
        manifest["Mode"].astype(str)
        + "_I"
        + manifest["CurrentLevel_A"].astype(str)
        + "_T"
        + manifest["Temperature_C"].astype(str)
    )
    label_map = {"healthy": 0, "vbus_bias": 1, "il_bias": 2, "S1_open": 3, "S2_open": 4, "high_resistance": 5}
    labels = manifest["FaultGroup"].map(label_map).to_numpy(dtype=int)
    groups = manifest["OperatingPointID"].to_numpy(dtype=str)

    baseline = pd.read_csv(BASELINE_PATH)
    delta = pd.read_csv(DELTA_PATH)
    if baseline["RunID"].astype(str).tolist() != manifest["RunID"].tolist() or delta["RunID"].astype(str).tolist() != manifest["RunID"].tolist():
        raise ValueError("Feature tables do not align with the manifest")
    baseline_index = ["RunID", "OperatingPointID", "Label", "FaultGroup", "FaultSubtype", "Mode", "Temperature_C", "CurrentLevel_A"]
    delta_index = ["RunID", "OperatingPointID", "Label", "FaultGroup", "Mode", "Temperature_C", "CurrentLevel_A"]
    matrices = {
        "baseline": baseline.drop(columns=baseline_index).to_numpy(dtype=float),
        "delta": delta.drop(columns=delta_index).to_numpy(dtype=float),
    }
    baseline_spec = candidate("baseline_extra_trees_full")
    delta_spec = candidate("random_forest_delta")

    outer_prediction = np.full(len(labels), -1, dtype=int)
    outer_probabilities = np.full((len(labels), len(LABELS)), np.nan)
    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(GroupKFold(n_splits=6).split(np.zeros(len(labels)), labels, groups), start=1):
        inner_labels = labels[outer_train]
        inner_groups = groups[outer_train]
        baseline_inner = np.full((len(outer_train), len(LABELS)), np.nan)
        delta_inner = np.full((len(outer_train), len(LABELS)), np.nan)
        for inner_fold, (inner_train_local, inner_test_local) in enumerate(
            GroupKFold(n_splits=5).split(np.zeros(len(outer_train)), inner_labels, inner_groups), start=1
        ):
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            baseline_inner[inner_test_local] = fit_probabilities(
                baseline_spec, matrices["baseline"], labels, inner_train, inner_test, SEED + 41000 + outer_fold * 100 + inner_fold
            )
            delta_inner[inner_test_local] = fit_probabilities(
                delta_spec, matrices["delta"], labels, inner_train, inner_test, SEED + 42000 + outer_fold * 100 + inner_fold
            )
        scores = {}
        for weight in WEIGHTS:
            prediction = np.argmax(weight * baseline_inner + (1 - weight) * delta_inner, axis=1)
            scores[weight] = metrics(inner_labels, prediction)["selection_objective"]
        chosen_weight = max(WEIGHTS, key=lambda weight: (scores[weight], weight))

        baseline_outer = fit_probabilities(
            baseline_spec, matrices["baseline"], labels, outer_train, outer_test, SEED + 43000 + outer_fold
        )
        delta_outer = fit_probabilities(
            delta_spec, matrices["delta"], labels, outer_train, outer_test, SEED + 44000 + outer_fold
        )
        combined = chosen_weight * baseline_outer + (1 - chosen_weight) * delta_outer
        prediction = np.argmax(combined, axis=1)
        outer_prediction[outer_test] = prediction
        outer_probabilities[outer_test] = combined
        rows.append(
            {
                "outer_fold": outer_fold,
                "chosen_baseline_weight": chosen_weight,
                "inner_objective": scores[chosen_weight],
                "test_groups": "|".join(sorted(np.unique(groups[outer_test]))),
                **{f"inner_weight_{weight:.2f}": score for weight, score in scores.items()},
                **{f"outer_{key}": value for key, value in metrics(labels[outer_test], prediction).items()},
            }
        )
    if np.any(outer_prediction < 0) or np.isnan(outer_probabilities).any():
        raise ValueError("Fusion OOF predictions are incomplete")
    result = metrics(labels, outer_prediction)
    result.update(
        {
            "method": "nested_weighted_probability_fusion",
            "members": [baseline_spec.name, delta_spec.name],
            "candidate_baseline_weights": list(WEIGHTS),
            "comparison_fixed_baseline_macro_f1": 0.9493206189890274,
            "macro_f1_improvement_over_fixed_baseline": result["macro_f1"] - 0.9493206189890274,
            "conclusion": "adopt" if result["macro_f1"] > 0.9493206189890274 and result["healthy_far"] <= 0.05 else "reject_no_credible_gain",
        }
    )
    (OUTPUT / "nested_fusion_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT / "nested_fusion_by_fold.csv", index=False)
    predictions = manifest[["RunID", "OperatingPointID", "FaultGroup"]].copy()
    predictions["true_label"] = labels
    predictions["predicted_label"] = outer_prediction
    predictions["correct"] = (outer_prediction == labels).astype(int)
    for label in LABELS:
        predictions[f"probability_{label}"] = outer_probabilities[:, label]
    predictions.to_csv(OUTPUT / "nested_fusion_oof_predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
