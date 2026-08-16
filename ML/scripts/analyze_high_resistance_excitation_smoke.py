#!/usr/bin/env python
"""Decide whether load-step excitation makes high resistance observable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXCLUDED_COLUMNS = {
    "WindowID",
    "WindowStart",
    "WindowEnd",
    "ModeCommand",
    "ModeID",
    "SOCInit",
    "IrefLevel",
    "VbusRefSetting",
    "VbatInit",
    "Rload",
    "Pload",
    "Rbat",
    "Cbus",
    "CbusESR",
    "RandomSeed",
    "FaultID",
    "WindowFaultID",
    "FaultMagnitude",
    "FaultParameter1",
    "FaultParameter2",
    "FaultStartTime",
    "FaultEndTime",
    "FaultActiveRatio",
    "IsTrainingEligible",
    "IsHighResistance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-step-time", type=float, default=0.35)
    return parser.parse_args()


def usable_features(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [
        name
        for name in numeric
        if name not in EXCLUDED_COLUMNS
        and "Fault" not in name
        and not name.startswith("Validation_")
        and "true" not in name.lower()
    ]


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.root.glob("*/combined/feature_dataset.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four smoke feature tables, found {len(paths)}")
    parts = []
    for path in paths:
        part = pd.read_csv(path).copy()
        phase = path.parents[1].name
        part = part.assign(
            SmokePhase=phase,
            IsHighResistance=int("high_r" in phase),
        )
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    features = usable_features(frame)
    pre = frame["WindowStart"].between(0.10, args.load_step_time - 0.05)
    post = frame["WindowStart"].ge(args.load_step_time + 0.10)
    rows: list[dict[str, object]] = []

    for mode in (1, 2):
        mode_rows = frame["ModeCommand"].eq(mode)
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            post_rows = mode_rows & post & values.notna()
            labels = frame.loc[post_rows, "IsHighResistance"].to_numpy(dtype=int)
            post_values = values[post_rows].to_numpy(dtype=float)
            if len(np.unique(labels)) < 2 or np.nanstd(post_values) <= 1e-15:
                continue
            healthy = post_values[labels == 0]
            fault = post_values[labels == 1]
            pooled_std = np.sqrt((np.var(healthy) + np.var(fault)) / 2.0)
            effect = float((np.mean(fault) - np.mean(healthy)) / max(pooled_std, 1e-12))
            auc = float(roc_auc_score(labels, post_values))

            changes = {}
            for label, label_name in ((0, "Healthy"), (1, "HighResistance")):
                label_rows = mode_rows & frame["IsHighResistance"].eq(label)
                pre_values = values[label_rows & pre].dropna().to_numpy(dtype=float)
                after_values = values[label_rows & post].dropna().to_numpy(dtype=float)
                changes[label_name] = (
                    float(np.mean(after_values) - np.mean(pre_values))
                    if len(pre_values) and len(after_values)
                    else np.nan
                )
            rows.append(
                {
                    "ModeCommand": mode,
                    "Feature": feature,
                    "HealthyPostMean": float(np.mean(healthy)),
                    "HighResistancePostMean": float(np.mean(fault)),
                    "EffectSize": effect,
                    "AbsoluteEffectSize": abs(effect),
                    "AUC": auc,
                    "AUCSeparability": max(auc, 1.0 - auc),
                    "HealthyStepChange": changes["Healthy"],
                    "HighResistanceStepChange": changes["HighResistance"],
                    "StepChangeContrast": (
                        changes["HighResistance"] - changes["Healthy"]
                    ),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(
        ["ModeCommand", "AUCSeparability", "AbsoluteEffectSize"],
        ascending=[True, False, False],
    )
    metrics.to_csv(args.output / "feature_excitation_metrics.csv", index=False)
    top = metrics.groupby("ModeCommand", sort=True).head(30)
    top.to_csv(args.output / "top_excitation_features.csv", index=False)
    qualifying = metrics.loc[
        metrics["AUCSeparability"].ge(0.90)
        & metrics["AbsoluteEffectSize"].ge(1.0)
    ]
    counts = qualifying.groupby("ModeCommand")["Feature"].nunique().to_dict()
    proceed = all(counts.get(mode, 0) >= 5 for mode in (1, 2))
    summary = {
        "smoke_runs": int(frame["RunID"].nunique()),
        "window_rows": int(len(frame)),
        "features_evaluated": int(metrics["Feature"].nunique()),
        "load_step_time": args.load_step_time,
        "strong_feature_counts": {str(key): int(value) for key, value in counts.items()},
        "proceed_to_expansion": proceed,
        "decision_rule": (
            "At least five non-metadata features in each active mode with "
            "post-step AUC separability >=0.90 and |effect size| >=1.0."
        ),
    }
    (args.output / "smoke_decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 高阻负载阶跃 smoke 分析",
        "",
        f"- 运行数：{summary['smoke_runs']}；窗口数：{summary['window_rows']}。",
        f"- Mode1 强特征数：{counts.get(1, 0)}；Mode2 强特征数：{counts.get(2, 0)}。",
        f"- 是否进入完整扩展：{'是' if proceed else '否'}。",
        "",
        "该 smoke 仅验证物理激励是否产生可测差异，不能作为模型泛化成绩。",
    ]
    (args.output / "smoke_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop features:\n" + top.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
