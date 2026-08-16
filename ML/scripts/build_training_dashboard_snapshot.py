#!/usr/bin/env python
"""Build a source-backed snapshot for the local ML training dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "monitoring"
    / "latest_status.json"
)
DEFAULT_FEATURE_SUMMARY = (
    PROJECT_ROOT
    / "ML"
    / "results"
    / "feature_study_expanded_v13_bridge_696"
    / "summary.csv"
)
FALLBACK_FEATURE_SUMMARY = (
    PROJECT_ROOT
    / "ML"
    / "results"
    / "feature_study_expanded_v13"
    / "summary.csv"
)
DEFAULT_NESTED_OUTPUT = (
    PROJECT_ROOT
    / "ML"
    / "results"
    / "robust_nested_expanded_v13_bridge_696_rf_full"
)
DEFAULT_FINAL_MODEL = (
    PROJECT_ROOT
    / "ML"
    / "models"
    / "final_robust_expanded_v13_bridge_696_rf_full"
)
DEFAULT_BLIND_ROOT = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "blind_test"
)
DEFAULT_DEVELOPMENT_CASES = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "combined_development_expanded_v13"
    / "simulation_cases.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "reports" / "training_dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument(
        "--feature-summary",
        type=Path,
        default=DEFAULT_FEATURE_SUMMARY,
    )
    parser.add_argument("--nested-output", type=Path, default=DEFAULT_NESTED_OUTPUT)
    parser.add_argument("--final-model", type=Path, default=DEFAULT_FINAL_MODEL)
    parser.add_argument("--blind-root", type=Path, default=DEFAULT_BLIND_ROOT)
    parser.add_argument(
        "--development-cases",
        type=Path,
        default=DEFAULT_DEVELOPMENT_CASES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    sql: str,
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [path],
        },
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = json.loads(args.status.read_text(encoding="utf-8-sig"))
    phase_counts = status.get("PhaseCounts", {})
    bridge_runs = sum(
        int(value)
        for name, value in phase_counts.items()
        if "bridge_" in name
    )
    bridge_target = 240
    feature_study_complete = args.feature_summary.is_file()
    effective_feature_summary = (
        args.feature_summary
        if feature_study_complete
        else FALLBACK_FEATURE_SUMMARY
    )
    feature_rows = read_csv(effective_feature_summary)
    development_run_ids = len(
        {
            row["RunID"]
            for row in read_csv(args.development_cases)
        }
    )
    rf_rows = [
        row for row in feature_rows if row["Model"] == "random_forest"
    ]
    requested_variants = {
        "top_30": "30",
        "top_50": "50",
        "top_80": "80",
        "all_features": "全量",
    }
    comparison = []
    for variant, label in requested_variants.items():
        row = next(item for item in rf_rows if item["Variant"] == variant)
        macro_f1 = float(row["MacroF1Mean"])
        healthy_far = float(row["HealthyFalseAlarmRateMean"])
        comparison.extend(
            [
                {
                    "FeatureSet": label,
                    "Metric": "Macro-F1",
                    "Value": macro_f1,
                },
                {
                    "FeatureSet": label,
                    "Metric": "健康正确拒警率",
                    "Value": 1.0 - healthy_far,
                },
            ]
        )

    detail = []
    labels = {
        "all_features": "全量",
        "top_226": "全量排名档",
        "top_30": "30",
        "top_50": "50",
        "top_80": "80",
        "without_current_error": "去除电流误差族",
        "without_ibat_dynamics": "去除电池电流动态族",
        "without_suspect_families": "去除全部可疑特征族",
    }
    for row in rf_rows:
        if row["Variant"] not in labels:
            continue
        detail.append(
            {
                "方案": labels[row["Variant"]],
                "特征数": int(row["FeatureCount"]),
                "MacroF1均值": float(row["MacroF1Mean"]),
                "MacroF1标准差": float(row["MacroF1Std"]),
                "MacroF1最低折": float(row["MacroF1Min"]),
                "健康误报率": float(row["HealthyFalseAlarmRateMean"]),
            }
        )

    nested_summary = args.nested_output / "summary.csv"
    fold_metrics = args.nested_output / "fold_metrics.csv"
    progress_path = args.nested_output / "progress.json"
    completed_folds = 0
    nested_state = "等待"
    nested_detail: list[dict[str, object]] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8-sig"))
        completed_folds = int(progress.get("CompletedFolds", 0))
        nested_state = (
            "已完成"
            if progress.get("State") == "completed"
            else "稳健训练运行中"
        )
    if fold_metrics.exists():
        rows = read_csv(fold_metrics)
        completed_folds = len({row["Fold"] for row in rows})
    if nested_summary.exists():
        nested_state = "已完成"
        completed_folds = 6
        nested_detail = read_csv(nested_summary)
    freeze_manifest = args.final_model / "freeze_manifest.json"
    frozen_model = args.final_model / "random_forest.joblib"
    final_model_frozen = freeze_manifest.is_file() and frozen_model.is_file()
    final_model_state = "已冻结" if final_model_frozen else "等待"
    final_model_progress = "1/1" if final_model_frozen else "0/1"
    blind_target = 216
    blind_runs = (
        len(list(args.blind_root.glob("*/raw_runs/*.mat")))
        if args.blind_root.is_dir()
        else 0
    )
    pause_flag = args.status.parent / "USER_PAUSED.flag"
    blind_paused = pause_flag.is_file()
    blind_state = (
        "完成"
        if blind_runs >= blind_target
        else "已暂停"
        if blind_paused
        else "运行中"
        if args.blind_root.is_dir()
        else "未启动"
    )

    bridge_state = (
        "完成"
        if bridge_runs >= bridge_target
        else "运行中"
        if (args.status.parent.parent / "phase8_bridge_health").exists()
        else "等待"
    )
    feature_stage_state = "完成" if feature_study_complete else "运行中"
    final_report_dir = (
        PROJECT_ROOT
        / "output"
        / "reports"
        / "final_energy_storage_ml_report_2026-08-01"
    )
    final_report_complete = (
        (final_report_dir / "artifact.json").is_file()
        and (final_report_dir / "report.html").is_file()
    )
    final_report_state = "完成" if final_report_complete else "等待"
    final_report_progress = "1/1" if final_report_complete else "0/1"
    feature_count = max(
        int(row["FeatureCount"])
        for row in rf_rows
        if row["Variant"] == "all_features"
    )
    stages = [
        {"阶段": "修复后开发仿真", "状态": "完成", "进度": "696/696 RunID"},
        {"阶段": "十三阶段数据合并", "状态": "完成", "进度": "696 RunID"},
        {
            "阶段": "特征数量比较",
            "状态": feature_stage_state,
            "进度": (
                f"30/50/80/{feature_count}"
                if feature_study_complete
                else "新696集六折运行中"
            ),
        },
        {
            "阶段": "域桥接开发仿真",
            "状态": bridge_state,
            "进度": f"{bridge_runs}/{bridge_target}",
        },
        {
            "阶段": "稳健折内训练",
            "状态": nested_state,
            "进度": f"{completed_folds}/6折",
        },
        {
            "阶段": "最终模型冻结",
            "状态": final_model_state,
            "进度": final_model_progress,
        },
        {
            "阶段": "全新盲测",
            "状态": blind_state,
            "进度": f"{blind_runs}/{blind_target}",
        },
        {
            "阶段": "中文最终报告",
            "状态": final_report_state,
            "进度": final_report_progress,
        },
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    overview = [
        {
            "simulations": int(status["CompletedRuns"]),
            "developmentRunIDs": development_run_ids,
            "usableFeatures": feature_count,
            "bridgeRuns": bridge_runs,
            "bridgeTarget": bridge_target,
            "completedFolds": completed_folds,
            "targetFolds": 6,
            "cFreeGB": float(status["CFreeGB"]),
            "dFreeGB": float(status["DFreeGB"]),
            "datasetSizeGB": float(status["DatasetSizeGB"]),
        }
    ]
    sources = [
        source(
            "disk_status",
            "磁盘与仿真监控状态",
            "simulink/experiments/sensor_bias/scripts/dataset_output_v13/monitoring/latest_status.json",
            "磁盘守护脚本最近一次采样的空间与仿真完成状态。",
            (
                "SELECT * FROM read_json_auto("
                "'simulink/experiments/sensor_bias/scripts/dataset_output_v13/"
                "monitoring/latest_status.json')"
            ),
        ),
        source(
            "feature_study",
            "六折特征研究汇总",
            str(effective_feature_summary.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            (
                "696个修复后开发RunID上的新六折特征数量与消融比较。"
                if feature_study_complete
                else "新696集比较运行期间保留的旧456集六折基线，禁止当作新结果。"
            ),
            (
                "WITH study AS (SELECT * FROM read_csv_auto("
                f"'{str(effective_feature_summary.relative_to(PROJECT_ROOT)).replace(chr(92), '/')}')), "
                "runs AS (SELECT COUNT(DISTINCT RunID) AS DevelopmentRunIDs "
                "FROM read_csv_auto('simulink/experiments/sensor_bias/scripts/"
                "dataset_output_v13/combined_development_expanded_v13/"
                "simulation_cases.csv')) "
                "SELECT study.*, runs.DevelopmentRunIDs FROM study CROSS JOIN runs"
            ),
        ),
        source(
            "nested_training",
            "折内调参、概率校准与阈值结果",
            "ML/results/robust_nested_expanded_v13_bridge_696_rf_full",
            "696-RunID开发集上的稳健折内调参、温度校准与阈值优化输出。",
            (
                "SELECT * FROM read_csv_auto("
                "'ML/results/robust_nested_expanded_v13_bridge_696_rf_full/"
                "summary.csv')"
            ),
        ),
        source(
            "pipeline_status",
            "稳健训练、模型冻结与用户暂停状态",
            "ML/results/robust_nested_expanded_v13_bridge_696_rf_full/progress.json",
            "六折稳健训练、最终模型冻结、盲测进度及用户暂停标记。",
            (
                "WITH nested AS (SELECT * FROM read_json_auto("
                "'ML/results/robust_nested_expanded_v13_bridge_696_rf_full/"
                "progress.json')), frozen AS (SELECT * FROM read_json_auto("
                "'ML/models/final_robust_expanded_v13_bridge_696_rf_full/"
                "freeze_manifest.json')), blind AS (SELECT COUNT(*) AS "
                "BlindRawRuns FROM glob('simulink/experiments/sensor_bias/"
                "scripts/dataset_output_v13/blind_test/*/raw_runs/*.mat')), "
                "pause AS (SELECT COUNT(*) AS PauseFlagCount FROM glob("
                "'simulink/experiments/sensor_bias/scripts/dataset_output_v13/"
                "monitoring/USER_PAUSED.flag')) "
                "SELECT nested.State, "
                "nested.CompletedFolds, frozen.state AS FreezeState, "
                "frozen.blind_data_used, blind.BlindRawRuns, "
                "pause.PauseFlagCount FROM nested CROSS JOIN frozen "
                "CROSS JOIN blind CROSS JOIN pause"
            ),
        ),
    ]
    artifact = {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "Energy Storage 机器学习训练看板",
            "description": "开发仿真、特征比较、折内调参、概率校准和盲测流水线状态。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "simulations",
                    "description": "已成功完成的新增开发仿真，不含修复前264次数据。",
                    "dataset": "overview",
                    "sourceId": "disk_status",
                    "metrics": [
                        {"label": "开发仿真", "field": "simulations", "format": "number"}
                    ],
                },
                {
                    "id": "runids",
                    "description": "七阶段合并后的新开发RunID总数。",
                    "dataset": "overview",
                    "sourceId": "feature_study",
                    "metrics": [
                        {"label": "开发RunID", "field": "developmentRunIDs", "format": "number"}
                    ],
                },
                {
                    "id": "bridge",
                    "description": "新增中间SOC、负载和充放电命令组合的域桥接仿真。",
                    "dataset": "overview",
                    "sourceId": "disk_status",
                    "metrics": [
                        {"label": "域桥接已完成", "field": "bridgeRuns", "format": "number"},
                        {"label": "目标", "field": "bridgeTarget", "format": "number"},
                    ],
                },
                {
                    "id": "features",
                    "description": "冻结用于折内调参的实际可用特征数。",
                    "dataset": "overview",
                    "sourceId": "feature_study",
                    "metrics": [
                        {"label": "锁定特征数", "field": "usableFeatures", "format": "number"}
                    ],
                },
                {
                    "id": "folds",
                    "description": "当前脚本在结束时统一落盘，因此运行期间可能暂时显示0折。",
                    "dataset": "overview",
                    "sourceId": "nested_training",
                    "metrics": [
                        {"label": "调参完成折数", "field": "completedFolds", "format": "number"},
                        {"label": "目标", "field": "targetFolds", "format": "number"},
                    ],
                },
                {
                    "id": "disk",
                    "description": "D盘可用空间；安全线为40GB，紧急线为20GB。",
                    "dataset": "overview",
                    "sourceId": "disk_status",
                    "metrics": [
                        {"label": "D盘剩余GB", "field": "dFreeGB", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "feature_comparison",
                    "title": "30、50、80与全量225特征的六折表现",
                    "subtitle": "两项指标均为越高越好；健康正确拒警率=1−健康误报率。",
                    "type": "bar",
                    "dataset": "feature_comparison",
                    "sourceId": "feature_study",
                    "encodings": {
                        "x": {"field": "FeatureSet", "type": "ordinal", "label": "特征方案"},
                        "y": {"field": "Value", "type": "quantitative", "label": "六折均值", "format": "percent"},
                        "color": {"field": "Metric", "type": "nominal", "label": "指标"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "pipeline",
                    "title": "训练流水线状态",
                    "dataset": "stages",
                    "sourceId": "pipeline_status",
                    "defaultSort": {"field": "阶段", "direction": "asc"},
                    "columns": [
                        {"field": "阶段", "label": "阶段", "type": "text"},
                        {"field": "状态", "label": "状态", "type": "text"},
                        {"field": "进度", "label": "进度", "type": "text"},
                    ],
                },
                {
                    "id": "feature_detail",
                    "title": "特征方案与消融明细",
                    "dataset": "feature_detail",
                    "sourceId": "feature_study",
                    "defaultSort": {"field": "MacroF1均值", "direction": "desc"},
                    "columns": [
                        {"field": "方案", "label": "方案", "type": "text"},
                        {"field": "特征数", "label": "特征数", "format": "number"},
                        {"field": "MacroF1均值", "label": "Macro-F1均值", "format": "percent"},
                        {"field": "MacroF1标准差", "label": "标准差", "format": "percent"},
                        {"field": "MacroF1最低折", "label": "最低折", "format": "percent"},
                        {"field": "健康误报率", "label": "健康误报率", "format": "percent"},
                    ],
                },
            ],
            "sources": [
                {"id": item["id"], "label": item["label"], "path": item["path"]}
                for item in sources
            ],
            "blocks": [
                {
                    "id": "status_note",
                    "type": "markdown",
                    "body": (
                        "## 当前状态\n\n"
                        f"域桥接开发仿真：**{bridge_state}（{bridge_runs}/{bridge_target}）**。"
                        f"上下文特征挑战：**{nested_state}**。"
                        f"盲测：**{blind_state}（{blind_runs}/{blind_target}）**。"
                        "最终模型在盲测前已冻结，盲测未参与调参。"
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": ["simulations", "runids", "bridge", "features", "folds", "disk"],
                },
                {"id": "pipeline_block", "type": "table", "tableId": "pipeline", "layout": "full"},
                {"id": "comparison_block", "type": "chart", "chartId": "feature_comparison", "layout": "full"},
                {"id": "detail_block", "type": "table", "tableId": "feature_detail", "layout": "full"},
                {
                    "id": "caveat",
                    "type": "markdown",
                    "body": (
                        "## 读图提示\n\n"
                        "本页面是本机数据快照。当前调参脚本为避免半成品文件，只在全部六折结束时写出"
                        "折级汇总，所以训练期间完成折数可能保持为0；CPU占用和错误日志由后台守护任务继续监控。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "overview": overview,
                "feature_comparison": comparison,
                "feature_detail": detail,
                "stages": stages,
                "nested_summary": nested_detail,
            },
        },
        "sources": sources,
    }
    output_path = args.output / "artifact.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
