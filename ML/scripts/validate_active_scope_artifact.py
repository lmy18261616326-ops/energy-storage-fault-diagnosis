#!/usr/bin/env python
"""Integrity-check the packaged active-scope model on its documented scope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    events = pd.read_csv(args.events)
    index = pd.read_csv(args.event_index, usecols=["RunID", "ModeCommand"])
    events = events.merge(index, on="RunID", how="left", validate="one_to_one")
    events = events.loc[~events["FaultMechanism"].eq("high_resistance")].copy()
    mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
    inactive = (events["WindowFaultID"].eq(3) & mode.ne(1)) | (
        events["WindowFaultID"].eq(4) & mode.ne(2)
    )
    events = events.loc[~inactive].copy()
    mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
    output = model.predict_with_status(events, mode)
    labels = events["WindowFaultID"].to_numpy(dtype=int)
    prediction = output["PredictedClassID"].to_numpy(dtype=int)
    healthy = labels == 0
    validation = {
        "artifact_loaded": True,
        "rows": int(len(events)),
        "feature_count": len(model.feature_names),
        "resubstitution_macro_f1": float(
            f1_score(labels, prediction, average="macro", zero_division=0)
        ),
        "resubstitution_healthy_far": float(np.mean(prediction[healthy] != 0)),
        "primary_verifier_agreement": float(output["ModelsAgree"].mean()),
        "high_resistance_supported_values": sorted(
            output["HighResistanceSupported"].unique().tolist()
        ),
        "status_counts": {
            str(key): int(value)
            for key, value in output["ObservabilityStatus"].value_counts().items()
        },
        "qualification_evidence": (
            "Use the six-fold grouped OOF comparison; resubstitution metrics here "
            "only verify serialization and inference integrity."
        ),
    }
    (args.output / "artifact_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(
        [events[["RunID", "OperatingPointID", "WindowFaultID"]], output], axis=1
    ).to_csv(args.output / "artifact_predictions.csv", index=False)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
