#!/usr/bin/env python
"""Analyze whether one global alarm threshold can control OOF false alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    frame = frame.loc[frame["Variant"].eq("calibrated_argmax")].copy()
    probability_columns = [f"ProbabilityClass{index}" for index in range(5)]
    probability = frame[probability_columns].to_numpy(dtype=float)
    labels = frame["WindowFaultID"].to_numpy(dtype=int)
    alarm_probability = 1.0 - probability[:, 0]
    fault_class = np.argmax(probability[:, 1:], axis=1) + 1
    healthy = labels == 0
    operating_points = frame["OperatingPointID"].astype(str).to_numpy()

    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 1.0, 951):
        prediction = np.where(alarm_probability >= threshold, fault_class, 0)
        per_op_far = [
            float(np.mean(prediction[healthy & (operating_points == op)] != 0))
            for op in np.unique(operating_points[healthy])
        ]
        rows.append(
            {
                "Threshold": float(threshold),
                "MacroF1": float(
                    f1_score(labels, prediction, average="macro", zero_division=0)
                ),
                "HealthyFAR": float(np.mean(prediction[healthy] != 0)),
                "WorstOperatingPointFAR": float(max(per_op_far)),
                "FaultMacroRecall": float(
                    recall_score(
                        labels[~healthy],
                        prediction[~healthy],
                        labels=[1, 2, 3, 4],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    tradeoff = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(args.output / "global_threshold_tradeoff.csv", index=False)

    best_macro = tradeoff.loc[tradeoff["MacroF1"].idxmax()]
    overall_feasible = tradeoff.loc[tradeoff["HealthyFAR"].le(0.05)]
    worst_op_feasible = tradeoff.loc[
        tradeoff["WorstOperatingPointFAR"].le(0.05)
    ]

    healthy_quantiles = (
        frame.loc[healthy, ["OperatingPointID"]]
        .assign(AlarmProbability=alarm_probability[healthy])
        .groupby("OperatingPointID")["AlarmProbability"]
        .quantile([0.5, 0.9, 0.95, 0.99])
        .unstack()
        .reset_index()
    )
    healthy_quantiles.columns = [
        "OperatingPointID",
        "P50",
        "P90",
        "P95",
        "P99",
    ]
    healthy_quantiles.to_csv(
        args.output / "healthy_alarm_probability_quantiles.csv",
        index=False,
    )

    def row_or_none(candidates: pd.DataFrame) -> dict[str, float] | None:
        if candidates.empty:
            return None
        row = candidates.loc[candidates["MacroF1"].idxmax()]
        return {key: float(value) for key, value in row.items()}

    summary = {
        "best_macro_f1": {key: float(value) for key, value in best_macro.items()},
        "best_with_overall_far_le_5pct": row_or_none(overall_feasible),
        "best_with_every_op_far_le_5pct": row_or_none(worst_op_feasible),
    }
    (args.output / "global_threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
