#!/usr/bin/env python
"""Independently validate the frozen commissioned high-resistance specialist.

The ExtraTrees architecture and decision threshold are fixed before this script
sees the validation labels.  Pilot healthy runs provide the commissioned
operating-point fingerprints; validation runs use new plant randomization seeds
and an unseen 0.05-ohm switch resistance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
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
from evaluate_high_resistance_excitation_pilot import make_model, worst_group_far


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-events", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=816702)
    return parser.parse_args()


def load_validation_events(root: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(root.glob("*/combined/feature_dataset.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four validation feature tables, found {len(paths)}")
    full_windows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    magnitude = full_windows.groupby("RunID", sort=False)["FaultMagnitude"].first()
    selected_columns = required_event_columns(
        full_windows.columns, DEFAULT_EVENT_SIGNALS
    )
    events, base_features = build_event_dataset(full_windows[selected_columns].copy())
    events["FaultMagnitude"] = events["RunID"].map(magnitude)
    events["BinaryLabel"] = events["FaultMechanism"].eq("high_resistance").astype(int)
    if "ModeCommand" not in events:
        events["ModeCommand"] = (
            events["OperatingPointID"].astype(str).str.extract(r"_m([12])_")[0].astype(int)
        )
    return events, base_features


def commissioned_validation_features(
    pilot: pd.DataFrame,
    validation: pd.DataFrame,
    source_features: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    pilot_health = pilot.loc[pilot["FaultMechanism"].eq("healthy")].copy()
    pilot_health[source_features] = pilot_health[source_features].apply(
        pd.to_numeric, errors="coerce"
    )
    counts = (
        pilot_health.groupby("OperatingPointID", sort=True)
        .size()
        .rename("PilotHealthyReferenceCount")
        .reset_index()
    )
    baselines = pilot_health.groupby("OperatingPointID", sort=True)[source_features].median()
    missing = sorted(set(validation["OperatingPointID"]) - set(baselines.index))
    if missing:
        raise ValueError(f"Validation operating points lack pilot baselines: {missing}")

    values = validation[source_features].apply(pd.to_numeric, errors="coerce")
    reference = baselines.loc[validation["OperatingPointID"]].set_axis(
        validation.index
    )
    delta = values - reference
    residual_names = [f"HealthyRefDelta__{name}" for name in source_features]
    absolute_names = [f"HealthyRefAbsDelta__{name}" for name in source_features]
    residual = pd.DataFrame(
        np.concatenate([delta.to_numpy(), np.abs(delta.to_numpy())], axis=1),
        columns=[*residual_names, *absolute_names],
        index=validation.index,
    )
    return pd.concat([validation, residual], axis=1), [*residual_names, *absolute_names], counts


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(args.pilot_events)
    validation, validation_base_features = load_validation_events(args.validation_root)
    all_pilot_features = [name for name in pilot.columns if "__" in name]
    source_features = [
        name for name in all_pilot_features if "__load_step_" not in name
    ]
    if source_features != validation_base_features:
        missing = sorted(set(source_features) - set(validation_base_features))
        extra = sorted(set(validation_base_features) - set(source_features))
        if missing or extra:
            raise ValueError(
                f"Pilot/validation base feature mismatch: missing={missing}, extra={extra}"
            )
        validation_base_features = source_features

    pilot_paired, residual_features, pilot_coverage = healthy_reference_features(
        pilot, source_features
    )
    validation_paired, validation_features, commissioned_coverage = (
        commissioned_validation_features(
            pilot, validation, validation_base_features
        )
    )
    if residual_features != validation_features:
        raise ValueError("Pilot and validation residual feature order differs")

    model = make_model("extra_trees", args.seed)
    weights = compute_sample_weight("balanced", pilot_paired["BinaryLabel"])
    model.fit(
        pilot_paired[residual_features],
        pilot_paired["BinaryLabel"],
        model__sample_weight=weights,
    )
    probability = model.predict_proba(validation_paired[residual_features])[:, 1]
    labels = validation_paired["BinaryLabel"].to_numpy(dtype=int)
    prediction = (probability >= args.threshold).astype(int)
    healthy = labels == 0
    fault = labels == 1
    mode = validation_paired["ModeCommand"].to_numpy(dtype=int)

    metrics = {
        "FrozenThreshold": float(args.threshold),
        "ValidationRuns": int(len(labels)),
        "HealthyRuns": int(healthy.sum()),
        "HighResistanceRuns": int(fault.sum()),
        "ROC_AUC": float(roc_auc_score(labels, probability)),
        "PR_AUC": float(average_precision_score(labels, probability)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "HealthyFAR": float(np.mean(prediction[healthy])),
        "WorstOperatingPointFAR": worst_group_far(
            labels, prediction, validation_paired["OperatingPointID"]
        ),
        "HighResistanceRecall": float(np.mean(prediction[fault])),
        "Mode1HighResistanceRecall": float(np.mean(prediction[fault & (mode == 1)])),
        "Mode2HighResistanceRecall": float(np.mean(prediction[fault & (mode == 2)])),
    }
    qualified = bool(
        metrics["HealthyFAR"] <= 0.05
        and metrics["WorstOperatingPointFAR"] <= 0.25
        and metrics["HighResistanceRecall"] >= 0.60
        and metrics["Mode1HighResistanceRecall"] >= 0.50
        and metrics["Mode2HighResistanceRecall"] >= 0.50
    )
    metrics["IndependentValidationQualified"] = qualified

    predictions = validation_paired[
        [
            "RunID",
            "OperatingPointID",
            "ModeCommand",
            "FaultName",
            "FaultMagnitude",
            "BinaryLabel",
        ]
    ].copy()
    predictions["ProbabilityHighResistance"] = probability
    predictions["Prediction"] = prediction
    predictions.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output / "metrics.csv", index=False)
    pilot_coverage.to_csv(args.output / "pilot_healthy_coverage.csv", index=False)
    commissioned_coverage.to_csv(
        args.output / "commissioned_healthy_coverage.csv", index=False
    )

    decision = {
        "decision_locked_before_validation": {
            "model": "extra_trees",
            "feature_set": "commissioned_base_event",
            "threshold": float(args.threshold),
        },
        "independent_condition": (
            "New domain-randomization seeds and unseen Ron=0.05 ohm; pilot healthy "
            "fingerprints are retained as the commissioned references."
        ),
        "qualification": metrics,
        "deployment_scope": (
            "Only operating points with a commissioned healthy fingerprint and an "
            "active switch (S1 in Mode1; S2 in Mode2)."
        ),
    }
    (args.output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if qualified:
        args.model_output.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": model,
            "threshold": float(args.threshold),
            "source_features": source_features,
            "residual_features": residual_features,
            "healthy_baselines": pilot.loc[
                pilot["FaultMechanism"].eq("healthy"),
                ["OperatingPointID", *source_features],
            ]
            .groupby("OperatingPointID", sort=True)
            .median(numeric_only=True),
            "supported_modes": [1, 2],
            "supported_faults": {1: "switch_S1_high_resistance", 2: "switch_S2_high_resistance"},
            "requires_commissioned_operating_point": True,
            "independent_validation_metrics": metrics,
        }
        joblib.dump(artifact, args.model_output / "commissioned_high_resistance_et.joblib")
        (args.model_output / "manifest.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
