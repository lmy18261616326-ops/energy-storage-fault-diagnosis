#!/usr/bin/env python
"""Diagnose high-resistance features under a frozen pilot-to-validation transfer.

This is a development diagnostic after the first independent candidate failed.
Every architecture and threshold is derived from pilot grouped OOF predictions;
the independent-validation labels are used only to measure transfer.  Selecting
a new candidate from this table therefore requires another untouched blind set.
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

from energy_fault_ml.event_features import (
    DEFAULT_EVENT_SIGNALS,
    build_event_dataset,
    required_event_columns,
)
from diagnose_paired_high_resistance import healthy_reference_features
from evaluate_high_resistance_excitation_pilot import (
    choose_threshold,
    make_model,
    step_features,
    worst_group_far,
)
from validate_high_resistance_commissioned_specialist import (
    commissioned_validation_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-events", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=817101)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--load-step-time", type=float, default=0.35)
    return parser.parse_args()


def load_validation(root: Path, load_step_time: float) -> pd.DataFrame:
    paths = sorted(root.glob("*/combined/feature_dataset.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four validation tables, found {len(paths)}")
    full = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    magnitude = full.groupby("RunID", sort=False)["FaultMagnitude"].first()
    selected = required_event_columns(full.columns, DEFAULT_EVENT_SIGNALS)
    windows = full[selected].copy()
    events, _ = build_event_dataset(windows)
    steps, _ = step_features(windows, load_step_time)
    events = events.merge(steps, on="RunID", how="left", validate="one_to_one")
    events["FaultMagnitude"] = events["RunID"].map(magnitude)
    events["BinaryLabel"] = events["FaultMechanism"].eq("high_resistance").astype(int)
    return events


def add_relative_residuals(
    frame: pd.DataFrame,
    source_features: list[str],
    scale_floor: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    relative_columns: dict[str, pd.Series] = {}
    names: list[str] = []
    for name in source_features:
        delta_name = f"HealthyRefDelta__{name}"
        delta = pd.to_numeric(frame[delta_name], errors="coerce")
        value = pd.to_numeric(frame[name], errors="coerce")
        reference = value - delta
        denominator = np.maximum(np.abs(reference), float(scale_floor[name]))
        relative_name = f"HealthyRefRelativeDelta__{name}"
        absolute_name = f"HealthyRefAbsRelativeDelta__{name}"
        relative_columns[relative_name] = delta / denominator
        relative_columns[absolute_name] = np.abs(delta) / denominator
        names.extend((relative_name, absolute_name))
    relative_frame = pd.DataFrame(relative_columns, index=frame.index)
    return pd.concat([frame, relative_frame], axis=1).copy(), names


def evaluate_candidate(
    pilot: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    feature_set: str,
    model_name: str,
    folds: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    groups = np.asarray(sorted(pilot["OperatingPointID"].unique()))
    group_folds = [np.asarray(part) for part in np.array_split(groups, folds)]
    oof_probability = np.full(len(pilot), np.nan, dtype=float)
    for fold_index, test_groups in enumerate(group_folds):
        test_mask = pilot["OperatingPointID"].isin(test_groups).to_numpy()
        train = pilot.loc[~test_mask]
        model = make_model(model_name, seed + fold_index)
        weights = compute_sample_weight("balanced", train["BinaryLabel"])
        model.fit(
            train[features], train["BinaryLabel"], model__sample_weight=weights
        )
        oof_probability[test_mask] = model.predict_proba(
            pilot.loc[test_mask, features]
        )[:, 1]
    if np.isnan(oof_probability).any():
        raise RuntimeError("OOF probability coverage is incomplete")
    pilot_labels = pilot["BinaryLabel"].to_numpy(dtype=int)
    threshold = choose_threshold(pilot_labels, oof_probability)
    pilot_prediction = (oof_probability >= threshold).astype(int)

    final_model = make_model(model_name, seed + 100)
    weights = compute_sample_weight("balanced", pilot["BinaryLabel"])
    final_model.fit(
        pilot[features], pilot["BinaryLabel"], model__sample_weight=weights
    )
    probability = final_model.predict_proba(validation[features])[:, 1]
    labels = validation["BinaryLabel"].to_numpy(dtype=int)
    prediction = (probability >= threshold).astype(int)
    healthy = labels == 0
    fault = labels == 1
    mode = pd.to_numeric(validation["ModeCommand"], errors="raise").to_numpy(dtype=int)
    result = {
        "FeatureSet": feature_set,
        "Model": model_name,
        "FeatureCount": len(features),
        "PilotOOFThreshold": threshold,
        "PilotOOFHealthyFAR": float(np.mean(pilot_prediction[pilot_labels == 0])),
        "PilotOOFHighResistanceRecall": float(
            np.mean(pilot_prediction[pilot_labels == 1])
        ),
        "ValidationROC_AUC": float(roc_auc_score(labels, probability)),
        "ValidationPR_AUC": float(average_precision_score(labels, probability)),
        "ValidationF1": float(f1_score(labels, prediction, zero_division=0)),
        "ValidationHealthyFAR": float(np.mean(prediction[healthy])),
        "ValidationWorstOperatingPointFAR": worst_group_far(
            labels, prediction, validation["OperatingPointID"]
        ),
        "ValidationHighResistanceRecall": float(np.mean(prediction[fault])),
        "ValidationMode1Recall": float(np.mean(prediction[fault & (mode == 1)])),
        "ValidationMode2Recall": float(np.mean(prediction[fault & (mode == 2)])),
    }
    result["TransferGatePassed"] = bool(
        result["ValidationHealthyFAR"] <= 0.05
        and result["ValidationWorstOperatingPointFAR"] <= 0.25
        and result["ValidationHighResistanceRecall"] >= 0.60
        and result["ValidationMode1Recall"] >= 0.50
        and result["ValidationMode2Recall"] >= 0.50
    )
    predictions = validation[
        ["RunID", "OperatingPointID", "ModeCommand", "BinaryLabel"]
    ].copy()
    predictions.insert(0, "FeatureSet", feature_set)
    predictions.insert(1, "Model", model_name)
    predictions["Threshold"] = threshold
    predictions["ProbabilityHighResistance"] = probability
    predictions["Prediction"] = prediction
    return result, predictions


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(args.pilot_events)
    validation = load_validation(args.validation_root, args.load_step_time)
    pilot["BinaryLabel"] = pilot["FaultMechanism"].eq("high_resistance").astype(int)

    all_features = [name for name in pilot.columns if "__" in name]
    step_features_list = [name for name in all_features if "__load_step_" in name]
    base_features = [name for name in all_features if name not in step_features_list]
    missing = sorted(set(all_features) - set(validation.columns))
    if missing:
        raise ValueError(f"Validation is missing pilot features: {missing}")

    pilot_commissioned, commissioned_names, _ = healthy_reference_features(
        pilot, base_features
    )
    validation_commissioned, validation_commissioned_names, _ = (
        commissioned_validation_features(pilot, validation, base_features)
    )
    if commissioned_names != validation_commissioned_names:
        raise ValueError("Commissioned base feature order mismatch")

    pilot_commissioned_all, commissioned_all_names, _ = healthy_reference_features(
        pilot, all_features
    )
    validation_commissioned_all, validation_commissioned_all_names, _ = (
        commissioned_validation_features(pilot, validation, all_features)
    )
    if commissioned_all_names != validation_commissioned_all_names:
        raise ValueError("Commissioned all-feature order mismatch")

    pilot_health_scale = (
        pilot.loc[pilot["FaultMechanism"].eq("healthy"), all_features]
        .apply(pd.to_numeric, errors="coerce")
        .abs()
        .median()
        .mul(0.05)
        .clip(lower=1e-9)
    )
    pilot_relative, relative_names = add_relative_residuals(
        pilot_commissioned_all, all_features, pilot_health_scale
    )
    validation_relative, validation_relative_names = add_relative_residuals(
        validation_commissioned_all, all_features, pilot_health_scale
    )
    if relative_names != validation_relative_names:
        raise ValueError("Relative feature order mismatch")

    physics_tokens = (
        "Relative",
        "Residual",
        "Duty",
        "Gate",
        "CurrentError",
        "VoltageError",
        "IL_meas",
        "Ibat_meas",
    )
    physics_features = [
        name for name in all_features if any(token in name for token in physics_tokens)
    ]
    feature_sets = {
        "raw_base_event": (pilot, validation, base_features),
        "raw_load_step": (pilot, validation, step_features_list),
        "raw_event_plus_step": (pilot, validation, all_features),
        "raw_physics_subset": (pilot, validation, physics_features),
        "commissioned_base_event": (
            pilot_commissioned,
            validation_commissioned,
            commissioned_names,
        ),
        "commissioned_event_plus_step": (
            pilot_commissioned_all,
            validation_commissioned_all,
            commissioned_all_names,
        ),
        "relative_commissioned_event_plus_step": (
            pilot_relative,
            validation_relative,
            relative_names,
        ),
    }
    model_names = (
        "extra_trees",
        "random_forest",
        "xgboost",
        "logistic_regression",
    )
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for feature_index, (feature_set, parts) in enumerate(feature_sets.items()):
        pilot_part, validation_part, features = parts
        for model_index, model_name in enumerate(model_names):
            row, prediction = evaluate_candidate(
                pilot_part,
                validation_part,
                features,
                feature_set,
                model_name,
                args.folds,
                args.seed + feature_index * 1000 + model_index * 100,
            )
            rows.append(row)
            predictions.append(prediction)

    results = pd.DataFrame(rows).sort_values(
        [
            "TransferGatePassed",
            "ValidationHealthyFAR",
            "ValidationHighResistanceRecall",
            "ValidationROC_AUC",
        ],
        ascending=[False, True, False, False],
    )
    results.to_csv(args.output / "candidate_transfer_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output / "candidate_predictions.csv", index=False
    )
    decision = {
        "diagnostic_only": True,
        "reason": (
            "The first independent validation has now been inspected; a candidate "
            "selected here must pass a second untouched blind validation."
        ),
        "candidate_count": int(len(results)),
        "transfer_gate_passed_count": int(results["TransferGatePassed"].sum()),
    }
    (args.output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(results.head(12).to_string(index=False))
    print("\n" + json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
