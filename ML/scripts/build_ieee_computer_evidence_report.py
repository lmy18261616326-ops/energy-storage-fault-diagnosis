#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the IEEE computer evidence."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ML" / "results" / "ieee_computer_evidence_v17"
OUT = ROOT / "ML" / "reports" / "ieee_computer_evidence_2026-08-04"


MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "knn": "KNN",
    "mlp": "MLP",
    "cnn_1d": "1D-CNN",
}

SCENARIO_NAMES = {
    "ideal": "理想",
    "nominal": "标称",
    "moderate": "中等扰动",
    "harsh": "严苛扰动",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | int | None:
    if value is None or value == "":
        return None
    value = value.strip()
    if value == "":
        return None
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    tables: list[str],
    filters: list[str],
    definitions: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": "2026-08-04T16:30:00+08:00",
            "tables_used": tables,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    main_rows_raw = read_csv("main_model_run_metrics_ci.csv")
    main_rows = []
    for row in main_rows_raw:
        if row["variant"] != "argmax":
            continue
        main_rows.append(
            {
                "model": MODEL_NAMES[row["model"]],
                "model_key": row["model"],
                "runs": number(row["runs"]),
                "operating_points": number(row["operating_points"]),
                "accuracy": number(row["accuracy"]),
                "accuracy_ci95_low": number(row["accuracy_ci95_low"]),
                "accuracy_ci95_high": number(row["accuracy_ci95_high"]),
                "macro_f1": number(row["macro_f1"]),
                "macro_f1_cluster_ci95_low": number(
                    row["macro_f1_cluster_bootstrap_ci95_low"]
                ),
                "macro_f1_cluster_ci95_high": number(
                    row["macro_f1_cluster_bootstrap_ci95_high"]
                ),
                "balanced_accuracy": number(row["balanced_accuracy"]),
                "healthy_far": number(row["healthy_false_alarm_rate"]),
                "healthy_far_ci95_high": number(row["healthy_far_ci95_high"]),
            }
        )
    main_rows.sort(key=lambda row: (-row["macro_f1"], row["model"]))

    recalls = []
    class_names = {
        "0": "Healthy",
        "1": "Voltage sensor bias",
        "2": "Current sensor bias",
        "3": "S1 open-circuit",
        "4": "S2 open-circuit",
    }
    for row in read_csv("main_model_class_recall_ci.csv"):
        if row["model"] == "logistic_regression" and row["variant"] == "argmax":
            recalls.append(
                {
                    "class": class_names[row["class_id"]],
                    "support_runs": number(row["support_runs"]),
                    "recall": number(row["recall"]),
                    "recall_ci95_low": number(row["recall_ci95_low"]),
                    "recall_ci95_high": number(row["recall_ci95_high"]),
                    "precision": number(row["precision"]),
                }
            )

    mcnemar = []
    for row in read_csv("main_model_pairwise_mcnemar.csv"):
        mcnemar.append(
            {
                "candidate": MODEL_NAMES[row["candidate_model"]],
                "reference_only_correct": number(row["reference_only_correct"]),
                "candidate_only_correct": number(row["candidate_only_correct"]),
                "discordant_runs": number(row["discordant_runs"]),
                "exact_p_value": number(row["exact_mcnemar_p_value"]),
                "interpretation": (
                    "显著弱于逻辑回归"
                    if number(row["exact_mcnemar_p_value"]) < 0.05
                    else "未检出显著差异"
                ),
            }
        )

    stress = []
    for row in read_csv("high_r_measurement_stress_summary.csv"):
        stress.append(
            {
                "scenario": SCENARIO_NAMES[row["scenario"]],
                "scenario_key": row["scenario"],
                "severity_mohm": number(row["severity_ohm"]) * 1000,
                "simulated_runs": number(row["simulated_runs"]),
                "source_profiles": number(row["source_profiles"]),
                "detection_rate": number(row["detection_rate"]),
                "detection_ci95_low": number(row["detection_rate_ci95_low"]),
                "detection_ci95_high": number(row["detection_rate_ci95_high"]),
                "pre_fault_window_far": number(
                    row["pre_fault_window_false_alarm_rate"]
                ),
                "median_latency_ms": number(row["median_detection_latency_ms"]),
                "p95_latency_ms": number(row["p95_detection_latency_ms"]),
            }
        )
    stress.sort(key=lambda row: (row["severity_mohm"], row["scenario_key"]))
    stress_focus = [
        row for row in stress if row["severity_mohm"] in {8, 10, 12, 15}
    ]

    threshold = []
    for row in read_csv("high_r_threshold_tradeoff.csv"):
        threshold.append(
            {
                "threshold_mohm": number(row["threshold_ohm"]) * 1000,
                "accuracy": number(row["accuracy"]),
                "balanced_accuracy": number(row["balanced_accuracy"]),
                "macro_f1": number(row["macro_f1"]),
                "fault_recall": number(row["fault_recall"]),
                "healthy_far": number(row["healthy_false_alarm_rate"]),
            }
        )

    quality = json.loads((RESULTS / "data_quality.json").read_text(encoding="utf-8"))
    moderate_12 = next(
        row
        for row in stress
        if row["scenario_key"] == "moderate" and row["severity_mohm"] == 12
    )
    lr = next(row for row in main_rows if row["model_key"] == "logistic_regression")
    cnn = next(row for row in main_rows if row["model_key"] == "cnn_1d")

    headline = [
        {
            "main_macro_f1": lr["macro_f1"],
            "main_accuracy_ci_low": lr["accuracy_ci95_low"],
            "healthy_far_ci_high": lr["healthy_far_ci95_high"],
            "highr_12_detection": moderate_12["detection_rate"],
            "highr_12_detection_ci_low": moderate_12["detection_ci95_low"],
            "highr_12_p95_latency_ms": moderate_12["p95_latency_ms"],
            "stress_runs": quality["stress_rows"],
        }
    ]

    sources = [
        source(
            "main_run_stats",
            "主任务 Run 级指标与置信区间",
            "ML/results/ieee_computer_evidence_v17/main_model_run_metrics_ci.csv",
            "SELECT * FROM read_csv_auto('ML/results/ieee_computer_evidence_v17/main_model_run_metrics_ci.csv') WHERE variant='argmax'",
            "读取七类模型在统一 6 折 Group OOF 口径下的 Run 级指标与区间估计。",
            [
                "ML/results/ieee_computer_evidence_v17/main_model_run_metrics_ci.csv",
                "ML/results/event_model_comparison_v14_active_6fold_v2/predictions.csv",
            ],
            ["variant=argmax", "416 Runs", "28 OperatingPointID", "6-fold Group OOF"],
            [
                "Accuracy 的 95% 区间为 Run 级 Clopper-Pearson 精确区间。",
                "Macro-F1 区间按 OperatingPointID 聚类自助法计算。",
                "Healthy FAR 为健康 Run 被预测为任一故障类别的比例。",
            ],
        ),
        source(
            "class_recall",
            "逻辑回归逐类召回置信区间",
            "ML/results/ieee_computer_evidence_v17/main_model_class_recall_ci.csv",
            "SELECT * FROM read_csv_auto('ML/results/ieee_computer_evidence_v17/main_model_class_recall_ci.csv') WHERE model='logistic_regression' AND variant='argmax'",
            "读取最终推荐主模型的逐类 Run 级召回与精确区间。",
            ["ML/results/ieee_computer_evidence_v17/main_model_class_recall_ci.csv"],
            ["model=logistic_regression", "variant=argmax"],
            ["Recall 的 95% 区间为按类 Run 计数的 Clopper-Pearson 精确区间。"],
        ),
        source(
            "mcnemar",
            "与逻辑回归的配对 McNemar 检验",
            "ML/results/ieee_computer_evidence_v17/main_model_pairwise_mcnemar.csv",
            "SELECT * FROM read_csv_auto('ML/results/ieee_computer_evidence_v17/main_model_pairwise_mcnemar.csv')",
            "对同一批 OOF Run 的正确/错误差异做精确 McNemar 检验。",
            ["ML/results/ieee_computer_evidence_v17/main_model_pairwise_mcnemar.csv"],
            ["reference_model=logistic_regression", "paired OOF Runs"],
            ["双侧精确 p 值仅用于检验配对错误差异，不表示工程等价。"],
        ),
        source(
            "highr_stress",
            "高阻传感链 Monte Carlo 压力测试",
            "ML/results/ieee_computer_evidence_v17/high_r_measurement_stress_summary.csv",
            "SELECT * FROM read_csv_auto('ML/results/ieee_computer_evidence_v17/high_r_measurement_stress_summary.csv')",
            "汇总四种测量扰动、九种故障电阻下的检测率、误报和延迟。",
            [
                "ML/results/ieee_computer_evidence_v17/high_r_measurement_stress_summary.csv",
                "ML/results/ieee_computer_evidence_v17/high_r_measurement_stress_runs.csv",
            ],
            [
                "24 个真实 Simulink 健康电流轨迹",
                "每组合 5 次 Monte Carlo",
                "冻结阈值 10.5 mΩ",
                "连续两个 10 ms 窗口触发",
            ],
            [
                "Detection rate 为故障后在仿真结束前触发的 Run 比例。",
                "Pre-fault window FAR 为故障注入前窗口被错误触发的比例。",
                "Latency 从随机故障时刻到第二个连续阳性窗口确认时刻计算。",
            ],
        ),
        source(
            "threshold_tradeoff",
            "高阻阈值敏感性分析",
            "ML/results/ieee_computer_evidence_v17/high_r_threshold_tradeoff.csv",
            "SELECT * FROM read_csv_auto('ML/results/ieee_computer_evidence_v17/high_r_threshold_tradeoff.csv')",
            "在中等扰动下扫描阈值，比较明确健康/早期区间与明确故障区间。",
            ["ML/results/ieee_computer_evidence_v17/high_r_threshold_tradeoff.csv"],
            ["scenario=moderate", "benign/incipient <=8 mΩ", "fault >=12 mΩ"],
            [
                "Macro-F1 对二元工程判据等权平均。",
                "Fault recall 仅以 >=12 mΩ 为必须检出的故障。",
                "Healthy FAR 在 <=8 mΩ 的非强制告警范围上计算。",
            ],
        ),
        source(
            "data_quality",
            "证据数据质量清单",
            "ML/results/ieee_computer_evidence_v17/data_quality.json",
            "SELECT * FROM read_json_auto('ML/results/ieee_computer_evidence_v17/data_quality.json')",
            "读取预测唯一性、缺失值、轨迹数和压力测试规模检查。",
            ["ML/results/ieee_computer_evidence_v17/data_quality.json"],
            ["duplicate rows=0", "missing required values=0"],
            ["Stress runs 为 24 轨迹 × 9 严重度 × 4 扰动场景 × 5 重复。"],
        ),
        {
            "id": "problem_log",
            "label": "完整问题日志 P01–P32",
            "path": "ML/research/problem_log.md",
            "query": {
                "engine": "workspace-files",
                "language": "markdown",
                "description": "记录历史基线、数据泄漏、可观测性、权限、MATLAB 会话和证据边界问题。",
                "executed_at": "2026-08-04T16:30:00+08:00",
                "tables_used": ["ML/research/problem_log.md"],
                "filters": ["P01–P32", "截至 2026-08-04"],
            },
        },
    ]

    cards = [
        {
            "id": "main_f1_card",
            "description": "逻辑回归在 416 个 Run、28 个工况上的 6 折 Group OOF Macro-F1。",
            "dataset": "headline",
            "sourceId": "main_run_stats",
            "metrics": [{"label": "主模型 Macro-F1", "field": "main_macro_f1", "format": "percent"}],
        },
        {
            "id": "accuracy_ci_card",
            "description": "主模型 Accuracy 的 Run 级 95% 精确区间下限。",
            "dataset": "headline",
            "sourceId": "main_run_stats",
            "metrics": [{"label": "Accuracy 95% 下限", "field": "main_accuracy_ci_low", "format": "percent"}],
        },
        {
            "id": "far_bound_card",
            "description": "144 个健康 Run 零误报时，Healthy FAR 的 95% 精确上限。",
            "dataset": "headline",
            "sourceId": "main_run_stats",
            "metrics": [{"label": "Healthy FAR 95% 上限", "field": "healthy_far_ci_high", "format": "percent"}],
        },
        {
            "id": "highr_detection_card",
            "description": "12 mΩ、中等扰动、120 次压力测试的检测率。",
            "dataset": "headline",
            "sourceId": "highr_stress",
            "metrics": [{"label": "12 mΩ 检测率", "field": "highr_12_detection", "format": "percent"}],
        },
        {
            "id": "highr_latency_card",
            "description": "12 mΩ、中等扰动下从故障发生到确认的 95 分位延迟。",
            "dataset": "headline",
            "sourceId": "highr_stress",
            "metrics": [{"label": "12 mΩ P95 延迟", "field": "highr_12_p95_latency_ms", "format": "number", "unit": "ms"}],
        },
        {
            "id": "stress_runs_card",
            "description": "传感链压力测试的 Run 级 Monte Carlo 总量。",
            "dataset": "headline",
            "sourceId": "data_quality",
            "metrics": [{"label": "压力测试规模", "field": "stress_runs", "format": "number", "unit": "Runs"}],
        },
    ]

    charts = [
        {
            "id": "model_f1_chart",
            "title": "统一 Group OOF 口径下的模型 Macro-F1",
            "subtitle": "树模型与逻辑回归达到上限；1D-CNN 在当前统计特征表示上明显失配",
            "type": "horizontalBar",
            "dataset": "main_models",
            "sourceId": "main_run_stats",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "模型"},
                "y": {"field": "macro_f1", "type": "quantitative", "format": "percent", "label": "Macro-F1"},
                "tooltip": [
                    {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                    {"field": "accuracy_ci95_low", "type": "quantitative", "format": "percent", "label": "Accuracy 95% 下限"},
                    {"field": "healthy_far", "type": "quantitative", "format": "percent", "label": "Healthy FAR"},
                ],
            },
            "xAxisTitle": "模型",
            "yAxisTitle": "Macro-F1",
            "valueFormat": "percent",
            "referenceLines": [{"axis": "y", "value": 0.95, "label": "95%"}],
            "layout": "full",
        },
        {
            "id": "highr_detection_curve",
            "title": "高阻检测率随故障电阻的变化",
            "subtitle": "冻结阈值 10.5 mΩ；10–12 mΩ 之间存在清晰的工程转折区",
            "type": "line",
            "dataset": "highr_stress",
            "sourceId": "highr_stress",
            "encodings": {
                "x": {"field": "severity_mohm", "type": "quantitative", "label": "故障电阻 (mΩ)"},
                "y": {"field": "detection_rate", "type": "quantitative", "format": "percent", "label": "检测率"},
                "color": {"field": "scenario", "type": "nominal", "label": "测量场景"},
                "tooltip": [
                    {"field": "detection_ci95_low", "type": "quantitative", "format": "percent", "label": "95% 下限"},
                    {"field": "detection_ci95_high", "type": "quantitative", "format": "percent", "label": "95% 上限"},
                    {"field": "p95_latency_ms", "type": "quantitative", "format": "number", "label": "P95 延迟 (ms)"},
                ],
            },
            "xAxisTitle": "故障电阻 (mΩ)",
            "yAxisTitle": "检测率",
            "valueFormat": "percent",
            "referenceLines": [{"axis": "x", "value": 10.5, "label": "冻结阈值 10.5 mΩ"}],
            "layout": "full",
        },
        {
            "id": "threshold_tradeoff_chart",
            "title": "中等扰动下的阈值敏感性",
            "subtitle": "明确非强制告警范围 ≤8 mΩ、必须检出范围 ≥12 mΩ",
            "type": "line",
            "dataset": "threshold_tradeoff",
            "sourceId": "threshold_tradeoff",
            "encodings": {
                "x": {"field": "threshold_mohm", "type": "quantitative", "label": "阈值 (mΩ)"},
                "y": {"field": "macro_f1", "type": "quantitative", "format": "percent", "label": "二元 Macro-F1"},
                "tooltip": [
                    {"field": "fault_recall", "type": "quantitative", "format": "percent", "label": "故障召回"},
                    {"field": "healthy_far", "type": "quantitative", "format": "percent", "label": "低阻区误报"},
                ],
            },
            "xAxisTitle": "阈值 (mΩ)",
            "yAxisTitle": "二元 Macro-F1",
            "valueFormat": "percent",
            "referenceLines": [{"axis": "x", "value": 10.5, "label": "当前阈值"}],
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "model_metrics_table",
            "title": "主任务模型比较与不确定性",
            "subtitle": "416 Runs、28 工况、6 折 OperatingPointID 分组 OOF",
            "dataset": "main_models",
            "sourceId": "main_run_stats",
            "density": "comfortable",
            "defaultSort": {"field": "macro_f1", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "macro_f1", "label": "Macro-F1", "format": "percent", "type": "percent"},
                {"field": "macro_f1_cluster_ci95_low", "label": "F1 聚类区间下限", "format": "percent", "type": "percent"},
                {"field": "accuracy", "label": "Accuracy", "format": "percent", "type": "percent"},
                {"field": "accuracy_ci95_low", "label": "Accuracy 95% 下限", "format": "percent", "type": "percent"},
                {"field": "healthy_far", "label": "Healthy FAR", "format": "percent", "type": "percent"},
                {"field": "healthy_far_ci95_high", "label": "FAR 95% 上限", "format": "percent", "type": "percent"},
            ],
        },
        {
            "id": "class_recall_table",
            "title": "推荐逻辑回归的逐类召回",
            "subtitle": "区间下限受每类 Run 数量约束；S1/S2 各 56 Run",
            "dataset": "class_recall",
            "sourceId": "class_recall",
            "density": "comfortable",
            "defaultSort": {"field": "support_runs", "direction": "desc"},
            "columns": [
                {"field": "class", "label": "类别", "type": "text"},
                {"field": "support_runs", "label": "Run 数", "type": "number"},
                {"field": "recall", "label": "Recall", "format": "percent", "type": "percent"},
                {"field": "recall_ci95_low", "label": "Recall 95% 下限", "format": "percent", "type": "percent"},
                {"field": "precision", "label": "Precision", "format": "percent", "type": "percent"},
            ],
        },
        {
            "id": "mcnemar_table",
            "title": "与逻辑回归的配对错误检验",
            "subtitle": "同一 OOF Run 上的双侧精确 McNemar；p<0.05 仅表示错误差异显著",
            "dataset": "mcnemar",
            "sourceId": "mcnemar",
            "density": "comfortable",
            "defaultSort": {"field": "exact_p_value", "direction": "asc"},
            "columns": [
                {"field": "candidate", "label": "候选模型", "type": "text"},
                {"field": "reference_only_correct", "label": "仅 LR 正确", "type": "number"},
                {"field": "candidate_only_correct", "label": "仅候选正确", "type": "number"},
                {"field": "exact_p_value", "label": "精确 p 值", "format": "number", "type": "number"},
                {"field": "interpretation", "label": "解释", "type": "text"},
            ],
        },
        {
            "id": "highr_focus_table",
            "title": "高阻转折区与延迟明细",
            "subtitle": "每个场景-严重度 120 次；仅展示 8、10、12、15 mΩ",
            "dataset": "highr_focus",
            "sourceId": "highr_stress",
            "density": "comfortable",
            "defaultSort": {"field": "severity_mohm", "direction": "asc"},
            "columns": [
                {"field": "scenario", "label": "测量场景", "type": "text"},
                {"field": "severity_mohm", "label": "故障电阻 (mΩ)", "format": "number", "type": "number"},
                {"field": "detection_rate", "label": "检测率", "format": "percent", "type": "percent"},
                {"field": "detection_ci95_low", "label": "检测率 95% 下限", "format": "percent", "type": "percent"},
                {"field": "pre_fault_window_far", "label": "故障前窗口 FAR", "format": "percent", "type": "percent"},
                {"field": "median_latency_ms", "label": "中位延迟 (ms)", "format": "number", "type": "number"},
                {"field": "p95_latency_ms", "label": "P95 延迟 (ms)", "format": "number", "type": "number"},
            ],
        },
    ]

    title = "储能变流器故障诊断：IEEE 一般级计算机证据补强"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\n面向论文 Methods、Results、Discussion 与组会问题汇报的可复现技术报告。"},
        {"id": "summary", "type": "markdown", "body": "## 技术摘要\n\n当前机器学习部分已经从“只报点估计”提升为“统一分组验证、置信区间、配对检验、阈值敏感性和测量扰动压力测试”的完整计算机证据链。主任务建议使用逻辑回归作为论文主模型，Extra Trees/Random Forest 作为非线性一致性对照；当前统计特征不支持把 1D-CNN 写成优势模型。"},
        {"id": "headline_strip", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {"id": "main_finding", "type": "markdown", "sourceId": "main_run_stats", "body": "## 主任务模型结论\n\n逻辑回归、Extra Trees 和 Random Forest 在 416 个 Run、28 个独立工况、6 折 OperatingPointID 分组 OOF 下均达到 Macro-F1=1.000。零误差不等于真实误差为零：Accuracy 的 95% 精确下限为 99.12%，144 个健康 Run 零误报对应 Healthy FAR 的 95% 上限仍为 2.53%。因此论文应同时报告点估计和区间，不写“完全可靠”。"},
        {"id": "model_chart_block", "type": "chart", "chartId": "model_f1_chart"},
        {"id": "model_table_block", "type": "table", "tableId": "model_metrics_table"},
        {"id": "class_recall_intro", "type": "markdown", "sourceId": "class_recall", "body": "## 逐类可靠性\n\n逻辑回归五类召回均为 100%，但 S1/S2 各只有 56 个 Run，其 95% 精确下限为 93.62%。这说明下一轮最有价值的补数不是继续增加模型，而是增加独立工况与开关故障 Run。"},
        {"id": "class_recall_block", "type": "table", "tableId": "class_recall_table"},
        {"id": "mcnemar_intro", "type": "markdown", "sourceId": "mcnemar", "body": "## 配对统计检验\n\nExtra Trees 与 Random Forest 和逻辑回归在这批 OOF Run 上没有不一致错误；KNN 未检出显著差异。MLP 和 1D-CNN 显著弱于逻辑回归，说明在当前具名统计特征空间中继续加深网络没有证据收益。"},
        {"id": "mcnemar_block", "type": "table", "tableId": "mcnemar_table"},
        {"id": "highr_finding", "type": "markdown", "sourceId": "highr_stress", "body": "## 高阻检测的工程边界\n\n冻结 10.5 mΩ 阈值后，12 mΩ 在中等和严苛扰动下均为 120/120 检出，95% 精确下限 96.97%；中等扰动 P95 确认延迟 14.78 ms，故障前窗口误报为 0。10 mΩ 是灰区：中等扰动 0/120，严苛扰动仅 8/120；≤8 mΩ 在所有场景均未触发。因此当前可辩护的论文表述是“≥12 mΩ 的仿真测量链保证”，而不是“任意高阻故障全覆盖”。"},
        {"id": "highr_chart_block", "type": "chart", "chartId": "highr_detection_curve"},
        {"id": "highr_table_block", "type": "table", "tableId": "highr_focus_table"},
        {"id": "threshold_finding", "type": "markdown", "sourceId": "threshold_tradeoff", "body": "## 阈值敏感性\n\n在中等扰动下，以 ≤8 mΩ 作为非强制告警/早期区间、≥12 mΩ 作为必须检出故障，当前 10.5 mΩ 位于 Macro-F1=1.000 的平台区，而不是单点碰巧最优。这比只给一个经验阈值更适合作为论文消融证据。"},
        {"id": "threshold_chart_block", "type": "chart", "chartId": "threshold_tradeoff_chart"},
        {"id": "scope", "type": "markdown", "body": "## 数据范围与指标定义\n\n主任务是五类主动可观测范围分类：健康、电压传感器偏置、电流传感器偏置、S1 开路、S2 开路。划分单位为 OperatingPointID，防止同工况信息跨折泄漏。高阻分支使用 24 条现有 Simulink 健康电流轨迹，叠加电压/电流噪声、增益误差、偏置、量化、1–5 μs 不同步和健康电阻漂移；9 个严重度、4 个场景、5 次重复，共 4320 个压力 Run。"},
        {"id": "method", "type": "markdown", "body": "## 方法与可复现性\n\n主任务使用完整 OOF 预测重新计算 Run 级 Clopper-Pearson 区间、OperatingPointID 聚类自助 Macro-F1 区间，以及以逻辑回归为参照的精确 McNemar 检验。高阻分支固定随机种子 20260804，10 ms 窗口、5 ms 步长、连续两个阳性窗口确认，阈值在压力测试前冻结为 10.5 mΩ。分析脚本、输入摘要、哈希、数据质量检查和全部 CSV 均已落盘。"},
        {"id": "limitations", "type": "markdown", "body": "## 局限与证据边界\n\n高阻压力测试复用了真实 Simulink 电流轨迹，但电阻阶跃和测量误差是在分析层施加；它不是动态功率级再仿真、HIL 或硬件验证。当前主任务的满分也可能反映所选工况内的类别几何可分，而非跨拓扑、跨参数和跨设备泛化。MATLAB 会话无法稳定附着，批处理加载 v06 模型超时，因此本轮没有修改 .slx；该问题已记为 P32。"},
        {"id": "paper_position", "type": "markdown", "body": "## IEEE 一般级论文定位\n\n机器学习部分现在足以作为 Simulink 建模论文中的完整验证模块：有基线、有传统/深度模型对照、有严格分组、有不确定性、有失败模型解释、有高阻阈值消融和扰动鲁棒性。论文主张应限定为“仿真工况内诊断与测量链鲁棒性”，不把它包装成工业部署或硬件认证。复杂融合当前不值得加入，因为主模型和两类树模型错误完全一致；融合没有可互补弱项。"},
        {"id": "next_steps", "type": "markdown", "body": "## 推荐的下一步\n\n1. 直接把本报告的模型表、逐类区间、高阻检测曲线和阈值敏感性写入论文。\n2. 在 Simulink 章节补齐模型结构图、参数表、故障注入时序和求解器/步长说明。\n3. 若电脑时间允许，优先新增跨参数盲测：母线电压、负载、电感/电容容差、传感器误差组合，而不是新增更复杂网络。\n4. 若后续能恢复 MATLAB 稳定批处理，再把 12 mΩ 附近 9–13 mΩ 做动态功率级阶跃复核。\n5. 硬件实验是加分项而非当前机器学习部分的前置条件；若投稿目标提升，再做少量代表点验证。"},
        {"id": "problems", "type": "markdown", "sourceId": "problem_log", "body": "## 问题记录\n\n完整问题链已更新至 P32，覆盖旧基线、数据泄漏、分布漂移、可观测性、PWM 采样混叠、权限等待、MATLAB 会话故障、置信区间缺失及高阻动态边界。该日志可直接用于后续组会问题汇报。"},
        {"id": "questions", "type": "markdown", "body": "## 仍需回答的论文问题\n\n- Simulink 模型参数是否来自器件手册、实验辨识或文献？\n- 投稿期刊/会议是否要求硬件或 HIL 证据？\n- 12 mΩ 的工程故障界限能否由器件热损耗、导通压降或维护标准解释？\n- 能否取得至少一个跨参数、跨随机种子且完全冻结的最终盲测集？"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "面向 IEEE 一般级论文的模型统计验证、高阻鲁棒性、局限和下一步报告。",
            "generatedAt": "2026-08-04T16:30:00+08:00",
            "filters": [],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "datasets": {
                "headline": headline,
                "main_models": main_rows,
                "class_recall": recalls,
                "mcnemar": mcnemar,
                "highr_stress": stress,
                "highr_focus": stress_focus,
                "threshold_tradeoff": threshold,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(f"datasets={len(artifact['snapshot']['datasets'])}, blocks={len(blocks)}")


if __name__ == "__main__":
    main()
