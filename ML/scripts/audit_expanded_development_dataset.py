#!/usr/bin/env python
"""Audit the repaired 696-RunID development merge before ML training."""

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
DEFAULT_OLD = DEFAULT_ROOT / "combined_development"
DEFAULT_MERGED = DEFAULT_ROOT / "combined_development_expanded_v13"
DEFAULT_OUTPUT = PROJECT_ROOT / "ML" / "results" / "expanded_v13_merge_audit"

PHASES = (
    "phase1_target_health",
    "phase2_core_health",
    "phase3_sensor_faults",
    "phase4_switch_full_open",
    "phase5_switch_partial_open",
    "phase6_switch_intermittent",
    "phase7_switch_high_resistance",
    "phase8_bridge_health",
    "phase9_bridge_sensor_faults",
    "phase10_bridge_switch_full_open",
    "phase11_bridge_switch_partial_open",
    "phase12_bridge_switch_intermittent",
    "phase13_bridge_switch_high_resistance",
)
BRIDGE_PHASES = PHASES[7:]
KEY_COLUMNS = ["RunID", "WindowID"]
AUDIT_COLUMNS = [
    "RunID",
    "OperatingPointID",
    "WindowID",
    "WindowStart",
    "WindowEnd",
    "ModeCommand",
    "ScenarioFaultID",
    "WindowFaultID",
    "IsTrainingEligible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append(
            {"Check": name, "Passed": bool(passed), "Evidence": evidence}
        )

    phase_cases: list[pd.DataFrame] = []
    phase_keys: list[pd.DataFrame] = []
    phase_rows: list[dict[str, object]] = []
    reference_columns: list[str] | None = None
    total_failed = 0
    total_raw = 0
    for phase in PHASES:
        combined = args.root / phase / "combined"
        feature_path = combined / "feature_dataset.csv"
        case_path = combined / "simulation_cases.csv"
        failed_path = combined / "failed_runs.csv"
        raw_path = args.root / phase / "raw_runs"

        columns = pd.read_csv(feature_path, nrows=0).columns.tolist()
        if reference_columns is None:
            reference_columns = columns
        schema_match = columns == reference_columns
        cases = pd.read_csv(case_path)
        keys = pd.read_csv(feature_path, usecols=KEY_COLUMNS)
        failed = pd.read_csv(failed_path) if failed_path.is_file() else pd.DataFrame()
        raw_ids = {path.stem for path in raw_path.glob("*.mat")}
        case_ids = set(cases["RunID"].astype(str))

        phase_cases.append(cases.assign(SourcePhase=phase))
        phase_keys.append(keys.assign(SourcePhase=phase))
        failed_count = int(len(failed))
        total_failed += failed_count
        total_raw += len(raw_ids)
        phase_rows.append(
            {
                "Phase": phase,
                "RunIDs": len(case_ids),
                "RawMatFiles": len(raw_ids),
                "FeatureWindows": len(keys),
                "FailedRuns": failed_count,
                "SchemaMatches": schema_match,
                "RawRunIDsMatchCases": raw_ids == case_ids,
            }
        )

    phase_case_frame = pd.concat(phase_cases, ignore_index=True)
    phase_key_frame = pd.concat(phase_keys, ignore_index=True)
    old_cases = pd.read_csv(args.old / "simulation_cases.csv")
    old_features = pd.read_csv(
        args.old / "feature_dataset.csv",
        usecols=AUDIT_COLUMNS,
    )
    merged_cases = pd.read_csv(args.merged / "simulation_cases.csv")
    merged_features = pd.read_csv(
        args.merged / "feature_dataset.csv",
        usecols=AUDIT_COLUMNS,
    )

    old_ids = set(old_cases["RunID"].astype(str))
    phase_ids = set(phase_case_frame["RunID"].astype(str))
    bridge_ids = set(
        phase_case_frame.loc[
            phase_case_frame["SourcePhase"].isin(BRIDGE_PHASES),
            "RunID",
        ].astype(str)
    )
    merged_ids = set(merged_cases["RunID"].astype(str))
    expected_ids = old_ids | bridge_ids

    merged_key_index = pd.MultiIndex.from_frame(
        merged_features[KEY_COLUMNS].astype({"RunID": str})
    )
    phase_key_index = pd.MultiIndex.from_frame(
        phase_key_frame[KEY_COLUMNS].astype({"RunID": str})
    )
    old_key_index = pd.MultiIndex.from_frame(
        old_features[KEY_COLUMNS].astype({"RunID": str})
    )

    check("phase_schema_identical", all(row["SchemaMatches"] for row in phase_rows), len(reference_columns or []))
    check("all_failed_runs_zero", total_failed == 0, total_failed)
    check("raw_mat_count", total_raw == 696, total_raw)
    check(
        "raw_runids_match_cases",
        all(row["RawRunIDsMatchCases"] for row in phase_rows),
        sum(not row["RawRunIDsMatchCases"] for row in phase_rows),
    )
    check(
        "phase_runids_unique",
        not phase_case_frame["RunID"].astype(str).duplicated().any(),
        int(phase_case_frame["RunID"].astype(str).duplicated().sum()),
    )
    check("old_run_count", len(old_ids) == 456, len(old_ids))
    check("bridge_run_count", len(bridge_ids) == 240, len(bridge_ids))
    check("old_bridge_disjoint", old_ids.isdisjoint(bridge_ids), len(old_ids & bridge_ids))
    check("merged_run_count", len(merged_ids) == 696, len(merged_ids))
    check("merged_is_exact_union", merged_ids == expected_ids == phase_ids, len(merged_ids ^ expected_ids))
    check(
        "merged_case_runids_unique",
        not merged_cases["RunID"].astype(str).duplicated().any(),
        int(merged_cases["RunID"].astype(str).duplicated().sum()),
    )
    check(
        "merged_windows_unique",
        not merged_key_index.duplicated().any(),
        int(merged_key_index.duplicated().sum()),
    )
    check(
        "merged_windows_exact_phase_union",
        len(merged_key_index) == len(phase_key_index)
        and set(merged_key_index) == set(phase_key_index),
        {
            "merged": len(merged_key_index),
            "phase_union": len(phase_key_index),
            "symmetric_difference": len(set(merged_key_index) ^ set(phase_key_index)),
        },
    )
    check(
        "old_windows_preserved",
        set(old_key_index).issubset(set(merged_key_index)),
        {
            "old": len(old_key_index),
            "missing": len(set(old_key_index) - set(merged_key_index)),
        },
    )
    check(
        "feature_runids_have_cases",
        set(merged_features["RunID"].astype(str)).issubset(merged_ids),
        len(set(merged_features["RunID"].astype(str)) - merged_ids),
    )
    check(
        "required_fields_complete",
        not merged_features[AUDIT_COLUMNS].isna().any().any(),
        int(merged_features[AUDIT_COLUMNS].isna().sum().sum()),
    )
    check(
        "window_time_valid",
        bool((merged_features["WindowEnd"] > merged_features["WindowStart"]).all()),
        int((merged_features["WindowEnd"] <= merged_features["WindowStart"]).sum()),
    )
    eligible = pd.to_numeric(
        merged_features["IsTrainingEligible"], errors="coerce"
    ).fillna(0).ne(0)
    labels = pd.to_numeric(
        merged_features.loc[eligible, "WindowFaultID"], errors="coerce"
    )
    check("eligible_labels_valid", set(labels.dropna().astype(int)) == set(range(5)), sorted(labels.dropna().unique().tolist()))
    mode_values = set(
        pd.to_numeric(
            merged_features["ModeCommand"], errors="coerce"
        ).dropna().astype(int)
    )
    check("mode_coverage", mode_values == {0, 1, 2}, sorted(mode_values))
    check(
        "operating_point_count",
        merged_features["OperatingPointID"].nunique() == 28,
        int(merged_features["OperatingPointID"].nunique()),
    )
    manifest_text = (args.merged / "phase_manifest.csv").read_text(
        encoding="utf-8"
    )
    check(
        "no_prefix_data_source",
        "dataset_output\\combined" not in manifest_text
        and "dataset_output/combined" not in manifest_text,
        "Only dataset_output_v13 phase1-13 paths are present.",
    )

    eligible_frame = merged_features.loc[eligible].copy()
    mode_class = (
        eligible_frame.groupby(["ModeCommand", "WindowFaultID"], observed=True)
        .size()
        .rename("WindowCount")
        .reset_index()
    )
    mode_class_cells = {
        (int(row.ModeCommand), int(row.WindowFaultID))
        for row in mode_class.itertuples(index=False)
    }
    expected_mode_class_cells = {
        (0, 0),
        (0, 1),
        (0, 2),
        *{
            (mode, class_id)
            for mode in (1, 2)
            for class_id in range(5)
        },
    }
    check(
        "mode_class_expected_coverage",
        mode_class_cells == expected_mode_class_cells,
        len(mode_class_cells),
    )
    op_class = (
        eligible_frame.groupby(
            ["OperatingPointID", "WindowFaultID"], observed=True
        )
        .size()
        .rename("WindowCount")
        .reset_index()
    )
    op_classes = {
        str(operating_point): set(
            part["WindowFaultID"].astype(int).tolist()
        )
        for operating_point, part in op_class.groupby("OperatingPointID")
    }
    bridge_op_classes = {
        key: value
        for key, value in op_classes.items()
        if key.startswith("bridge_op_")
    }
    legacy_op_classes = {
        key: value
        for key, value in op_classes.items()
        if key.startswith("op_")
    }
    check(
        "bridge_operating_points_full_class_coverage",
        len(bridge_op_classes) == 16
        and all(value == set(range(5)) for value in bridge_op_classes.values()),
        {
            "operating_points": len(bridge_op_classes),
            "incomplete": sorted(
                key
                for key, value in bridge_op_classes.items()
                if value != set(range(5))
            ),
        },
    )
    check(
        "legacy_operating_points_minimum_class_coverage",
        len(legacy_op_classes) == 12
        and all({0, 1, 2}.issubset(value) for value in legacy_op_classes.values()),
        {
            "operating_points": len(legacy_op_classes),
            "missing_base_classes": sorted(
                key
                for key, value in legacy_op_classes.items()
                if not {0, 1, 2}.issubset(value)
            ),
        },
    )
    run_coverage = (
        merged_cases.groupby(
            ["ModeCommand", "FaultID"], observed=True
        )
        .size()
        .rename("RunCount")
        .reset_index()
    )
    pd.DataFrame(phase_rows).to_csv(
        args.output / "phase_checks.csv", index=False
    )
    mode_class.to_csv(args.output / "mode_class_coverage.csv", index=False)
    op_class.to_csv(
        args.output / "operating_point_class_coverage.csv", index=False
    )
    run_coverage.to_csv(args.output / "run_coverage.csv", index=False)
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(args.output / "checks.csv", index=False)

    payload = {
        "passed": bool(check_frame["Passed"].all()),
        "source_root": str(args.root.resolve()),
        "old_development": str(args.old.resolve()),
        "merged_development": str(args.merged.resolve()),
        "run_count": len(merged_ids),
        "bridge_run_count": len(bridge_ids),
        "feature_window_count": len(merged_features),
        "eligible_window_count": int(eligible.sum()),
        "feature_column_count": len(reference_columns or []),
        "operating_point_count": int(
            merged_features["OperatingPointID"].nunique()
        ),
        "checks": checks,
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
