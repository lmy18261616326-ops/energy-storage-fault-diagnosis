#!/usr/bin/env python
"""Merge completed simulation phases while enforcing schema and ID isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "combined_development"
DEFAULT_PHASES = (
    "phase1_target_health",
    "phase2_core_health",
    "phase3_sensor_faults",
    "phase4_switch_full_open",
    "phase5_switch_partial_open",
    "phase6_switch_intermittent",
    "phase7_switch_high_resistance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", nargs="+", default=DEFAULT_PHASES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_frames: list[pd.DataFrame] = []
    case_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    reference_columns: list[str] | None = None
    for phase_name in args.phases:
        combined = args.root / phase_name / "combined"
        feature_file = combined / "feature_dataset.csv"
        case_file = combined / "simulation_cases.csv"
        failed_file = combined / "failed_runs.csv"
        if not feature_file.is_file() or not case_file.is_file():
            raise FileNotFoundError(
                f"Phase is incomplete: {phase_name} ({combined})"
            )
        features = pd.read_csv(feature_file)
        cases = pd.read_csv(case_file)
        if reference_columns is None:
            reference_columns = features.columns.tolist()
        elif features.columns.tolist() != reference_columns:
            missing = sorted(set(reference_columns) - set(features.columns))
            extra = sorted(set(features.columns) - set(reference_columns))
            raise ValueError(
                f"Feature schema mismatch in {phase_name}: "
                f"missing={missing}, extra={extra}"
            )
        failed_count = 0
        if failed_file.is_file():
            failed_count = len(pd.read_csv(failed_file))
        feature_frames.append(features)
        case_frames.append(cases)
        manifest.append(
            {
                "Phase": phase_name,
                "RunCount": int(cases["RunID"].nunique()),
                "FeatureWindowCount": int(len(features)),
                "EligibleWindowCount": int(
                    pd.to_numeric(
                        features["IsTrainingEligible"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .ne(0)
                    .sum()
                ),
                "FailedRunCount": int(failed_count),
                "FeatureFile": str(feature_file.resolve()),
            }
        )

    merged_features = pd.concat(feature_frames, ignore_index=True)
    merged_cases = pd.concat(case_frames, ignore_index=True)
    duplicate_runs = merged_cases["RunID"].astype(str).duplicated(keep=False)
    if duplicate_runs.any():
        duplicates = (
            merged_cases.loc[duplicate_runs, "RunID"]
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )
        raise ValueError(f"Duplicate RunID values across phases: {duplicates}")
    duplicate_windows = merged_features.duplicated(
        ["RunID", "WindowID"],
        keep=False,
    )
    if duplicate_windows.any():
        raise ValueError("Duplicate (RunID, WindowID) keys across phases.")

    args.output.mkdir(parents=True, exist_ok=True)
    merged_features.to_csv(args.output / "feature_dataset.csv", index=False)
    merged_cases.to_csv(args.output / "simulation_cases.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.output / "phase_manifest.csv", index=False)
    summary = {
        "phases": list(args.phases),
        "run_count": int(merged_cases["RunID"].nunique()),
        "operating_point_count": int(
            merged_features["OperatingPointID"].nunique()
        ),
        "feature_window_count": int(len(merged_features)),
        "eligible_window_count": int(
            pd.to_numeric(
                merged_features["IsTrainingEligible"],
                errors="coerce",
            )
            .fillna(0)
            .ne(0)
            .sum()
        ),
        "class_counts": {
            str(int(key)): int(value)
            for key, value in merged_features.loc[
                pd.to_numeric(
                    merged_features["IsTrainingEligible"],
                    errors="coerce",
                )
                .fillna(0)
                .ne(0)
            ]
            .groupby("WindowFaultID")
            .size()
            .items()
        },
    }
    (args.output / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
