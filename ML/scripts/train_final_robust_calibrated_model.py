#!/usr/bin/env python
"""Freeze the final robust model from six-fold nested development results.

The blind dataset is intentionally not accepted as an input. Hyperparameters are
selected only from the inner-validation candidate table, while temperature and
the deployment alarm threshold are fitted from honest outer-fold OOF
probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
for path in (SRC_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from energy_fault_ml.calibration import CalibratedAlarmArtifact, temperature_scale
from energy_fault_ml.data import load_feature_dataset
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import (
    feature_matrix,
    remove_unusable_training_features,
    select_feature_columns,
)
from energy_fault_ml.models import train_model
from run_feature_study import LABEL_NAMES, LABELS
from run_nested_calibrated import fit_temperature, threshold_prediction
from run_robust_nested_calibrated import (
    choose_robust_threshold,
    operating_point_cell_weights,
    worst_operating_point_far,
)


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
DEFAULT_NESTED = (
    ML_ROOT
    / "results"
    / "robust_nested_expanded_v13_bridge_696_rf_full"
)
DEFAULT_OUTPUT = (
    ML_ROOT
    / "models"
    / "final_robust_expanded_v13_bridge_696_rf_full"
)
MODEL_FILENAME = "random_forest.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--nested-results", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-healthy-far", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=240727)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_max_features(value: Any) -> str | float:
    text = str(value)
    if text in {"sqrt", "log2"}:
        return text
    return float(text)


def aggregate_candidates(
    trials: pd.DataFrame,
    *,
    target_far: float,
) -> tuple[pd.DataFrame, pd.Series]:
    required = {
        "Fold",
        "Candidate",
        "healthy_multiplier",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "ValidationMacroF1",
        "ValidationHealthyFAR",
        "ValidationWorstOperatingPointFAR",
    }
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Candidate trials are missing columns: {missing}")
    if trials.duplicated(["Fold", "Candidate"]).any():
        raise ValueError("Candidate trials contain duplicate Fold/Candidate rows.")
    folds = sorted(pd.to_numeric(trials["Fold"]).astype(int).unique().tolist())
    if folds != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"Expected folds 1..6, found {folds}.")

    rows: list[dict[str, Any]] = []
    for candidate, group in trials.groupby("Candidate", sort=True):
        group = group.copy()
        if sorted(group["Fold"].astype(int).tolist()) != folds:
            raise ValueError(f"Candidate {candidate} does not cover all six folds.")
        feasible = (
            group["ValidationHealthyFAR"].astype(float).le(target_far)
            & group["ValidationWorstOperatingPointFAR"]
            .astype(float)
            .le(target_far)
        )
        rows.append(
            {
                "Candidate": int(candidate),
                "HealthyMultiplier": float(group.iloc[0]["healthy_multiplier"]),
                "MaxDepth": int(float(group.iloc[0]["max_depth"])),
                "MinSamplesLeaf": int(
                    float(group.iloc[0]["min_samples_leaf"])
                ),
                "MaxFeatures": str(group.iloc[0]["max_features"]),
                "FoldCount": len(group),
                "FeasibleFoldCount": int(feasible.sum()),
                "ValidationMacroF1Mean": float(
                    group["ValidationMacroF1"].astype(float).mean()
                ),
                "ValidationMacroF1Std": float(
                    group["ValidationMacroF1"].astype(float).std(ddof=1)
                ),
                "ValidationMacroF1Min": float(
                    group["ValidationMacroF1"].astype(float).min()
                ),
                "ValidationHealthyFARMean": float(
                    group["ValidationHealthyFAR"].astype(float).mean()
                ),
                "ValidationWorstOperatingPointFARMean": float(
                    group["ValidationWorstOperatingPointFAR"]
                    .astype(float)
                    .mean()
                ),
                "ValidationWorstOperatingPointFARMax": float(
                    group["ValidationWorstOperatingPointFAR"]
                    .astype(float)
                    .max()
                ),
            }
        )
    aggregation = pd.DataFrame(rows)
    aggregation = aggregation.sort_values(
        [
            "FeasibleFoldCount",
            "ValidationMacroF1Min",
            "ValidationMacroF1Mean",
            "ValidationWorstOperatingPointFARMax",
            "Candidate",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    if int(aggregation.iloc[0]["FeasibleFoldCount"]) != 6:
        raise RuntimeError(
            "No candidate satisfied both healthy-FAR constraints in every fold."
        )
    aggregation["Selected"] = False
    aggregation.loc[0, "Selected"] = True
    return aggregation, aggregation.iloc[0]


def validate_nested_features(
    nested_features: pd.DataFrame,
    final_features: list[str],
) -> None:
    required = {"Fold", "Feature"}
    missing = sorted(required.difference(nested_features.columns))
    if missing:
        raise ValueError(f"Selected-feature table is missing columns: {missing}")
    expected = set(final_features)
    for fold in range(1, 7):
        observed = set(
            nested_features.loc[
                nested_features["Fold"].astype(int).eq(fold),
                "Feature",
            ].astype(str)
        )
        if observed != expected:
            raise ValueError(
                "Final feature interface differs from nested fold "
                f"{fold}: missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )


def validate_oof(
    oof: pd.DataFrame,
    development: pd.DataFrame,
) -> None:
    probability_columns = [f"ProbabilityClass{index}" for index in LABELS]
    required = {
        "Fold",
        "Variant",
        "RunID",
        "OperatingPointID",
        "WindowID",
        "WindowFaultID",
        *probability_columns,
    }
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    if len(oof) != len(development):
        raise ValueError(
            f"OOF row count {len(oof)} does not match development "
            f"row count {len(development)}."
        )
    key = ["RunID", "WindowID"]
    if oof.duplicated(key).any() or development.duplicated(key).any():
        raise ValueError("RunID/WindowID keys are not unique.")
    expected = set(map(tuple, development[key].astype(str).to_numpy()))
    observed = set(map(tuple, oof[key].astype(str).to_numpy()))
    if observed != expected:
        raise ValueError("OOF RunID/WindowID coverage differs from development.")
    if sorted(oof["Fold"].astype(int).unique().tolist()) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("OOF predictions do not cover all six outer folds.")
    probability = oof[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("OOF probabilities contain non-finite values.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("OOF probability rows do not sum to one.")


def main() -> int:
    args = parse_args()
    model_path = args.output / MODEL_FILENAME
    freeze_path = args.output / "freeze_manifest.json"
    if model_path.exists() or freeze_path.exists():
        raise FileExistsError(
            "Final freeze output already exists; refusing to overwrite "
            f"{args.output.resolve()}."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    nested_files = {
        "candidate_trials": args.nested_results / "candidate_trials.csv",
        "fold_metrics": args.nested_results / "fold_metrics.csv",
        "summary": args.nested_results / "summary.csv",
        "selected_features": args.nested_results / "selected_features.csv",
        "window_predictions": args.nested_results / "window_predictions.csv",
    }
    missing_files = [
        str(path) for path in nested_files.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Nested training outputs are incomplete: {missing_files}"
        )

    development = load_feature_dataset(args.data)
    candidates = select_feature_columns(
        development,
        feature_set="physics_enhanced",
    )
    columns, removed = remove_unusable_training_features(
        development,
        candidates,
        remove_constant=True,
    )
    nested_features = pd.read_csv(nested_files["selected_features"])
    validate_nested_features(nested_features, columns)

    trials = pd.read_csv(nested_files["candidate_trials"])
    aggregation, selected = aggregate_candidates(
        trials,
        target_far=args.target_healthy_far,
    )
    aggregation.to_csv(args.output / "candidate_aggregation.csv", index=False)

    parameters = {
        "n_estimators": 350,
        "max_depth": int(selected["MaxDepth"]),
        "min_samples_leaf": int(selected["MinSamplesLeaf"]),
        "max_features": parse_max_features(selected["MaxFeatures"]),
        "class_weight": None,
        "n_jobs": -1,
    }
    healthy_multiplier = float(selected["HealthyMultiplier"])
    weights = operating_point_cell_weights(
        development,
        healthy_multiplier,
    )
    matrix = feature_matrix(development, columns)
    labels = development["WindowFaultID"].to_numpy(dtype=int)
    trained = train_model(
        "random_forest",
        matrix,
        labels,
        matrix,
        labels,
        feature_columns=columns,
        model_parameters=parameters,
        random_seed=args.seed,
        tune=False,
        tuning_grid={},
        sample_weight_train=weights,
    )

    all_oof = pd.read_csv(nested_files["window_predictions"])
    oof = all_oof.loc[all_oof["Variant"].eq("raw_argmax")].copy()
    validate_oof(oof, development)
    probability_columns = [f"ProbabilityClass{index}" for index in LABELS]
    raw_oof_probability = oof[probability_columns].to_numpy(dtype=float)
    oof_labels = oof["WindowFaultID"].to_numpy(dtype=int)
    temperature = fit_temperature(raw_oof_probability, oof_labels)
    calibrated_oof_probability = temperature_scale(
        raw_oof_probability,
        temperature,
    )
    threshold_frame = oof.loc[
        :, ["OperatingPointID", "WindowFaultID"]
    ].copy()
    threshold, threshold_trials = choose_robust_threshold(
        calibrated_oof_probability,
        threshold_frame,
        args.target_healthy_far,
    )
    threshold_trials.to_csv(
        args.output / "global_robust_threshold_trials.csv",
        index=False,
    )
    oof_prediction = threshold_prediction(
        calibrated_oof_probability,
        threshold,
    )
    aggregate_metrics, _, confusion = evaluate_classification(
        oof_labels,
        oof_prediction,
        labels=LABELS,
        label_names=LABEL_NAMES,
    )
    healthy = oof_labels == 0
    oof_diagnostics = {
        "calibration_fit_macro_f1": float(aggregate_metrics["MacroF1"]),
        "calibration_fit_balanced_accuracy": float(
            aggregate_metrics["BalancedAccuracy"]
        ),
        "calibration_fit_healthy_far": float(
            np.mean(oof_prediction[healthy] != 0)
        ),
        "calibration_fit_worst_operating_point_far": float(
            worst_operating_point_far(threshold_frame, oof_prediction)
        ),
        "confusion_matrix": confusion.tolist(),
        "note": (
            "These values are calibration-fit diagnostics, not an additional "
            "unbiased performance estimate. Use nested fold_metrics.csv for "
            "development performance and the untouched blind set once."
        ),
    }

    artifact = CalibratedAlarmArtifact(
        base_artifact=trained.artifact,
        temperature=float(temperature),
        alarm_threshold=float(threshold),
    )
    joblib.dump(artifact, model_path)
    (args.output / "selected_features.txt").write_text(
        "\n".join(columns) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"Feature": feature, "Reason": reason}
            for feature, reason in sorted(removed.items())
        ]
    ).to_csv(args.output / "removed_features.csv", index=False)

    frozen_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "state": "frozen_pre_blind",
        "frozen_at_utc": frozen_at,
        "model": "random_forest",
        "model_file": MODEL_FILENAME,
        "data": str(args.data.resolve()),
        "nested_results": str(args.nested_results.resolve()),
        "development_run_ids": int(development["RunID"].nunique()),
        "development_operating_points": int(
            development["OperatingPointID"].nunique()
        ),
        "development_windows": int(len(development)),
        "feature_count": len(columns),
        "selected_candidate": int(selected["Candidate"]),
        "candidate_selection_rule": (
            "all six folds feasible, then maximum worst-fold validation "
            "Macro-F1, then mean validation Macro-F1, then lower maximum "
            "worst-operating-point FAR"
        ),
        "healthy_multiplier": healthy_multiplier,
        "selected_parameters": parameters,
        "temperature": float(temperature),
        "alarm_threshold": float(threshold),
        "target_healthy_false_alarm_rate": args.target_healthy_far,
        "calibration_source": (
            "raw probabilities from six honest outer-fold development tests"
        ),
        "oof_calibration_diagnostics": oof_diagnostics,
        "nested_robust_summary": pd.read_csv(
            nested_files["summary"]
        ).to_dict(orient="records"),
        "blind_data_used": False,
        "blind_data_generated": False,
        "blind_evaluation_count": 0,
        "development_data_sha256": sha256_file(args.data),
        "nested_input_sha256": {
            name: sha256_file(path) for name, path in nested_files.items()
        },
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    loaded = joblib.load(model_path)
    validation_matrix = matrix.iloc[: min(256, len(matrix))]
    validation_probability = loaded.predict_proba(validation_matrix)
    validation_prediction = loaded.predict(validation_matrix)
    validation = {
        "passed": bool(
            validation_probability.shape
            == (len(validation_matrix), len(LABELS))
            and np.isfinite(validation_probability).all()
            and np.allclose(
                validation_probability.sum(axis=1),
                1.0,
                atol=1e-6,
            )
            and set(np.unique(validation_prediction)).issubset(set(LABELS))
        ),
        "sample_rows": int(len(validation_matrix)),
        "probability_shape": list(validation_probability.shape),
        "predicted_classes": sorted(
            np.unique(validation_prediction).astype(int).tolist()
        ),
    }
    if not validation["passed"]:
        raise RuntimeError("Reloaded frozen-model structural validation failed.")
    validation_path = args.output / "freeze_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    frozen_files = [
        model_path,
        metadata_path,
        validation_path,
        args.output / "candidate_aggregation.csv",
        args.output / "global_robust_threshold_trials.csv",
        args.output / "selected_features.txt",
        args.output / "removed_features.csv",
    ]
    freeze_manifest = {
        "state": "frozen_pre_blind",
        "frozen_at_utc": frozen_at,
        "blind_data_used": False,
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in frozen_files
        },
    }
    freeze_path.write_text(
        json.dumps(
            freeze_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    print(f"Frozen model: {model_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
