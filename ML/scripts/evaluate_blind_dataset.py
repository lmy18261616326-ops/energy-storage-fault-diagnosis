#!/usr/bin/env python
"""Evaluate a locked calibrated artifact once on the untouched blind set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.data import load_feature_dataset
from energy_fault_ml.diagnostics import (
    build_prediction_frame,
    write_healthy_false_alarm_diagnostics,
)
from energy_fault_ml.evaluation import evaluate_classification
from energy_fault_ml.features import feature_matrix
from run_feature_study import LABEL_NAMES, LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_feature_dataset(args.data)
    artifact = joblib.load(args.model)
    matrix = feature_matrix(frame, artifact.feature_columns)
    prediction = artifact.predict(matrix)
    probability = artifact.predict_proba(matrix)
    aggregate, per_class, confusion = evaluate_classification(
        frame["WindowFaultID"].to_numpy(dtype=int),
        prediction,
        labels=LABELS,
        label_names=LABEL_NAMES,
    )
    pd.DataFrame([aggregate]).to_csv(
        args.output / "blind_metrics.csv",
        index=False,
    )
    per_class.to_csv(args.output / "blind_per_class_metrics.csv", index=False)
    prediction_frame = build_prediction_frame(
        frame,
        prediction,
        model_name=artifact.base_artifact.name,
        split_name="final_blind",
        label_column="WindowFaultID",
        label_names=LABEL_NAMES,
    )
    for class_id in LABELS:
        prediction_frame[f"ProbabilityClass{class_id}"] = (
            probability[:, class_id]
        )
    prediction_frame.to_csv(
        args.output / "blind_window_predictions.csv",
        index=False,
    )
    write_healthy_false_alarm_diagnostics(prediction_frame, args.output)
    (args.output / "blind_evaluation_metadata.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "model": str(args.model.resolve()),
                "sample_count": len(frame),
                "operating_point_count": int(
                    frame["OperatingPointID"].nunique()
                ),
                "confusion_matrix": confusion.tolist(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([aggregate]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
