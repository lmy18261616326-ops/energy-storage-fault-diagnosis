#!/usr/bin/env python
"""Fit and package the qualified active-observability event model."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_fault_ml.active_scope_model import ActiveScopeFaultModel
from run_event_model_baselines import hierarchical_sample_weight, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--comparison-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=240803)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    event_index = pd.read_csv(args.event_index)
    context = event_index[["RunID", "ModeCommand"]]
    events = events.merge(context, on="RunID", how="left", validate="one_to_one")
    events = events.loc[~events["FaultMechanism"].eq("high_resistance")].copy()
    mode = pd.to_numeric(events["ModeCommand"], errors="raise").astype(int)
    inactive = (events["WindowFaultID"].eq(3) & mode.ne(1)) | (
        events["WindowFaultID"].eq(4) & mode.ne(2)
    )
    events = events.loc[~inactive].copy()
    features = [name for name in events.columns if "__" in name]
    labels = events["WindowFaultID"].to_numpy(dtype=int)
    weights = hierarchical_sample_weight(labels)

    models = {
        "primary_logistic_regression": make_model(
            "logistic_regression", args.seed, class_count=5
        ),
        "verifier_extra_trees": make_model(
            "extra_trees", args.seed + 1, class_count=5
        ),
    }
    for model in models.values():
        fit_parameters = inspect.signature(model.named_steps["model"].fit).parameters
        fit_kwargs = (
            {"model__sample_weight": weights}
            if "sample_weight" in fit_parameters
            else {}
        )
        model.fit(events[features], labels, **fit_kwargs)

    packaged = ActiveScopeFaultModel(
        primary_model=models["primary_logistic_regression"],
        verifier_model=models["verifier_extra_trees"],
        feature_names=tuple(features),
    )
    model_path = args.output / "active_scope_fault_model.joblib"
    joblib.dump(packaged, model_path, compress=3)
    summary = pd.read_csv(args.comparison_results / "summary.csv")
    selected_metrics = summary.loc[
        summary["Variant"].eq("argmax")
        & summary["Model"].isin(["logistic_regression", "extra_trees"])
    ].to_dict(orient="records")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(model_path.resolve()),
        "primary_model": "logistic_regression",
        "verifier_model": "extra_trees",
        "selection_reason": (
            "Both reached 100% six-fold grouped OOF Macro-F1 and zero healthy FAR; "
            "logistic regression had the lowest OOF log loss and training cost."
        ),
        "qualification_scope": "active_switch_observable",
        "supported_classes": [0, 1, 2, 3, 4],
        "mode_rules": {
            "0": "sensor faults only; wait for switch excitation",
            "1": "sensor faults and S1 location observable",
            "2": "sensor faults and S2 location observable",
        },
        "unsupported": [
            "high_resistance with the current t=0 compile-time Ron dataset",
            "S1 localization outside Mode1",
            "S2 localization outside Mode2",
        ],
        "training_rows": int(len(events)),
        "operating_points": int(events["OperatingPointID"].nunique()),
        "feature_count": len(features),
        "feature_names_file": "feature_names.json",
        "comparison_results": str(args.comparison_results.resolve()),
        "selected_oof_metrics": selected_metrics,
        "sklearn_version": sklearn.__version__,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "feature_names.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
