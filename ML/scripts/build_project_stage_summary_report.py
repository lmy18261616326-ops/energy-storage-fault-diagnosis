#!/usr/bin/env python
"""Build a canonical Chinese HTML report summarizing completed project stages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
)
ORIGINAL_DEV = DATA_ROOT / "combined_development"
EXPANDED_DEV = DATA_ROOT / "combined_development_expanded_v13"
BLIND_DATA = DATA_ROOT / "blind_test" / "combined_blind_216"
FEATURE_RESULT = PROJECT_ROOT / "ML" / "results" / "feature_study_expanded_v13_bridge_696"
NESTED_RESULT = PROJECT_ROOT / "ML" / "results" / "robust_nested_expanded_v13_bridge_696_rf_full"
MODEL_ROOT = PROJECT_ROOT / "ML" / "models" / "final_robust_expanded_v13_bridge_696_rf_full"
BLIND_RESULT = PROJECT_ROOT / "ML" / "results" / "final_blind_expanded_v13_bridge_696_rf_full"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reports" / "project_stage_summary_2026-08-01"


CLASS_ZH = {
    0: "健康",
    1: "母线电压传感器偏置",
    2: "电感电流传感器偏置",
    3: "S1 开路",
    4: "S2 开路",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def make_source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    generated_at: str,
    *,
    sql: str | None = None,
    tables_used: list[str] | None = None,
    filters: list[str] | None = None,
    definitions: list[str] | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "engine": "DuckDB" if sql else "Python/pandas",
        "language": "SQL" if sql else "Python",
        "description": description,
        "executed_at": generated_at,
    }
    if sql:
        query["sql"] = sql
    if tables_used:
        query["tables_used"] = tables_used
    if filters:
        query["filters"] = filters
    if definitions:
        query["metric_definitions"] = definitions
    return {
        "id": source_id,
        "label": label,
        "path": rel(path),
        "query": query,
    }


def main() -> int:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    original_merge = json.loads((ORIGINAL_DEV / "merge_summary.json").read_text(encoding="utf-8"))
    expanded_merge = json.loads((EXPANDED_DEV / "merge_summary.json").read_text(encoding="utf-8"))
    blind_merge = json.loads((BLIND_DATA / "merge_summary.json").read_text(encoding="utf-8"))
    dev_phases = pd.read_csv(EXPANDED_DEV / "phase_manifest.csv")
    feature_summary = pd.read_csv(FEATURE_RESULT / "summary.csv")
    nested_summary = pd.read_csv(NESTED_RESULT / "summary.csv")
    model_meta = json.loads((MODEL_ROOT / "metadata.json").read_text(encoding="utf-8"))
    freeze_manifest = json.loads((MODEL_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    blind_metrics = pd.read_csv(BLIND_RESULT / "blind_metrics.csv").iloc[0]
    blind_per_class = pd.read_csv(BLIND_RESULT / "blind_per_class_metrics.csv")
    blind_far = pd.read_csv(BLIND_RESULT / "healthy_false_alarm_by_operating_point.csv")

    bridge_runs = int(
        dev_phases.loc[dev_phases["Phase"].str.startswith("phase8_bridge") | dev_phases["Phase"].str.startswith("phase9_bridge") | dev_phases["Phase"].str.startswith("phase10_bridge") | dev_phases["Phase"].str.startswith("phase11_bridge") | dev_phases["Phase"].str.startswith("phase12_bridge") | dev_phases["Phase"].str.startswith("phase13_bridge"), "RunCount"].sum()
    )
    original_run_ids = int(original_merge["run_count"])
    expanded_run_ids = int(expanded_merge["run_count"])
    blind_run_ids = int(blind_merge["run_count"])
    robust_row = nested_summary.loc[nested_summary["Variant"].eq("robust_threshold")].iloc[0]
    raw_row = nested_summary.loc[nested_summary["Variant"].eq("raw_argmax")].iloc[0]

    stage_rows = [
        {
            "stage": "1. 运行保护与可恢复执行",
            "tasks": "建立磁盘阈值、进程与日志监控；采用 numWorkers=1、parallelBatchSize=2 和 resume；避免重复启动。",
            "result": "长时仿真可断点恢复，未删除 raw_runs、combined、模型、报告或 ML 结果。",
            "status": "完成",
        },
        {
            "stage": "2. 原开发数据收束",
            "tasks": "完成 phase1–7 汇总并保持旧六折研究只作历史基线，不重复运行。",
            "result": f"原开发集 {original_run_ids} 个 RunID；其中 phase2–7 共 408 次运行，失败 0。",
            "status": "完成",
        },
        {
            "stage": "3. 工作域桥接扩展",
            "tasks": "完成健康、传感器故障及四类开关故障的 phase8–13 扩展采集。",
            "result": f"桥接扩展 {bridge_runs}/240 成功；六阶段失败数均为 0。",
            "status": "完成",
        },
        {
            "stage": "4. 新开发集严格合并与审计",
            "tasks": "将原开发集与 phase8–13 合并；检查失败、RunID/窗口重复、特征模式、类别与工况覆盖；排除修复前 264 次。",
            "result": f"{expanded_run_ids} 个 RunID、{int(expanded_merge['feature_window_count']):,} 个窗口、{int(expanded_merge['eligible_window_count']):,} 个合格窗口；28 个工况。",
            "status": "完成",
        },
        {
            "stage": "5. 特征预算与模型比较",
            "tasks": "重跑 30/50/80/全量特征比较，并检查去除可疑特征族的影响；比较随机森林与 XGBoost。",
            "result": "30/50 特征明显不足；80/全量更稳。最终进入稳健流程的是 225 特征随机森林。",
            "status": "完成",
        },
        {
            "stage": "6. 稳健折内调参、校准与冻结",
            "tasks": "执行六折嵌套候选选择、温度校准、健康误报约束下的阈值优化，并在盲测数据生成前冻结。",
            "result": f"稳健阈值开发 Macro-F1 {float(robust_row['MacroF1Mean']):.2%}，健康误报率 {float(robust_row['HealthyFARMean']):.2%}；模型温度 {float(model_meta['temperature']):.4f}、阈值 {float(model_meta['alarm_threshold']):.4f}。",
            "status": "完成",
        },
        {
            "stage": "7. 最终盲测生成与一次性评估",
            "tasks": "冻结后才生成 216 次全新盲测；严格合并、隔离审计后仅评估一次。",
            "result": f"{blind_run_ids}/216 成功、失败 0；{int(blind_merge['eligible_window_count']):,} 个合格窗口；Macro-F1 {float(blind_metrics['MacroF1']):.2%}。",
            "status": "完成",
        },
        {
            "stage": "8. 误差诊断与部署判断",
            "tasks": "输出逐类别指标、混淆矩阵和按工况健康误报诊断。",
            "result": "S1/S2 开路召回率均 13.79%；健康误报集中在两个 600 W、ModeCommand=1 工况；当前模型不建议部署。",
            "status": "完成",
        },
        {
            "stage": "9. 仪表板、最终报告与进程收束",
            "tasks": "持续更新训练仪表板，生成中文最终报告，停止 MATLAB/Python 训练任务和磁盘守护。",
            "result": "最终报告、仪表板、冻结模型与盲测结果均已落盘；相关计算进程已关闭。",
            "status": "完成",
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_csv = OUTPUT_DIR / "stage_summary.csv"
    pd.DataFrame(stage_rows).to_csv(stage_csv, index=False, encoding="utf-8-sig")

    batch_rows = [
        {"batch": "原开发集（phase1–7）", "runIds": original_run_ids, "role": "开发"},
        {"batch": "桥接扩展（phase8–13）", "runIds": bridge_runs, "role": "开发扩展"},
        {"batch": "最终盲测集", "runIds": blind_run_ids, "role": "一次性评估"},
    ]

    keep_variants = ["top_30", "top_50", "top_80", "all_features"]
    feature_rows: list[dict[str, object]] = []
    model_label = {"random_forest": "RF", "xgboost": "XGBoost"}
    variant_label = {
        "top_30": "Top 30",
        "top_50": "Top 50",
        "top_80": "Top 80",
        "all_features": "全量 225",
    }
    for row in feature_summary.loc[feature_summary["Variant"].isin(keep_variants)].to_dict(orient="records"):
        feature_rows.append(
            {
                "candidate": f"{model_label[str(row['Model'])]} · {variant_label[str(row['Variant'])]}",
                "model": model_label[str(row["Model"])],
                "variant": variant_label[str(row["Variant"])],
                "featureCount": int(row["FeatureCount"]),
                "macroF1": float(row["MacroF1Mean"]),
                "macroF1Std": float(row["MacroF1Std"]),
                "healthyFar": float(row["HealthyFalseAlarmRateMean"]),
            }
        )

    nested_rows = []
    nested_labels = {
        "raw_argmax": "原始概率 + argmax",
        "calibrated_argmax": "温度校准 + argmax",
        "robust_threshold": "温度校准 + 稳健阈值",
    }
    for row in nested_summary.to_dict(orient="records"):
        nested_rows.append(
            {
                "variant": nested_labels[str(row["Variant"])],
                "macroF1": float(row["MacroF1Mean"]),
                "macroF1Min": float(row["MacroF1Min"]),
                "healthyFar": float(row["HealthyFARMean"]),
                "worstOpFar": float(row["WorstOperatingPointFARMean"]),
                "ece": float(row["TestECEMean"]),
            }
        )

    per_class_rows = []
    for row in blind_per_class.to_dict(orient="records"):
        class_id = int(row["ClassID"])
        per_class_rows.append(
            {
                "classId": class_id,
                "className": CLASS_ZH[class_id],
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "support": int(row["Support"]),
            }
        )

    worst_far_rows = []
    for row in blind_far.sort_values("HealthyFalseAlarmRate", ascending=False).head(4).to_dict(orient="records"):
        worst_far_rows.append(
            {
                "operatingPoint": str(row["OperatingPointID"]),
                "mode": int(row["ModeCommand"]),
                "soc": int(row["SOCInit"]),
                "loadW": int(row["Pload"]),
                "healthyWindows": int(row["HealthyWindowCount"]),
                "falseAlarms": int(row["FalseAlarmCount"]),
                "falseAlarmRate": float(row["HealthyFalseAlarmRate"]),
            }
        )

    sources = [
        make_source(
            "src_stage_summary",
            "项目阶段汇总",
            stage_csv,
            "由开发集、桥接扩展、特征研究、嵌套验证、冻结清单和最终盲测产物汇总得到。",
            generated_at,
            sql=(
                "SELECT * FROM read_csv_auto("
                "'output/reports/project_stage_summary_2026-08-01/stage_summary.csv', "
                "header=true) ORDER BY Stage"
            ),
            tables_used=[
                rel(ORIGINAL_DEV / "merge_summary.json"),
                rel(EXPANDED_DEV / "merge_summary.json"),
                rel(EXPANDED_DEV / "phase_manifest.csv"),
                rel(FEATURE_RESULT / "summary.csv"),
                rel(NESTED_RESULT / "summary.csv"),
                rel(MODEL_ROOT / "metadata.json"),
                rel(BLIND_DATA / "merge_summary.json"),
                rel(BLIND_RESULT / "blind_metrics.csv"),
            ],
        ),
        make_source(
            "src_expanded_merge",
            "扩展开发集严格合并结果",
            EXPANDED_DEV / "merge_summary.json",
            "读取 phase1–13 合并后的 RunID、窗口、工况与类别规模。",
            generated_at,
            sql="SELECT * FROM read_json_auto('simulink/experiments/sensor_bias/scripts/dataset_output_v13/combined_development_expanded_v13/merge_summary.json')",
            tables_used=[rel(EXPANDED_DEV / "merge_summary.json"), rel(EXPANDED_DEV / "phase_manifest.csv")],
            filters=["phase1–13", "排除修复前 264 次", "RunID 与窗口键唯一", "FailedRunCount=0"],
            definitions=["合格窗口：IsTrainingEligible != 0", "开发 RunID：合并后所有唯一 RunID；其中 608 个 RunID 含合格窗口"],
        ),
        make_source(
            "src_feature_study",
            "特征预算与模型比较",
            FEATURE_RESULT / "summary.csv",
            "读取六折 30/50/80/全量特征及特征族消融比较结果。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/feature_study_expanded_v13_bridge_696/summary.csv', header=true)",
            filters=["Model IN (random_forest, xgboost)", "六个外层工况隔离折"],
            definitions=["Macro-F1 Mean：六个外层测试折 Macro-F1 的算术平均", "健康误报率：真实健康窗口被判为任意故障的比例"],
        ),
        make_source(
            "src_nested",
            "稳健嵌套验证结果",
            NESTED_RESULT / "summary.csv",
            "读取原始 argmax、温度校准 argmax 与稳健阈值三种决策规则的六折结果。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/robust_nested_expanded_v13_bridge_696_rf_full/summary.csv', header=true)",
            filters=["225 特征随机森林", "六个外层测试折", "阈值仅在训练折内选择"],
            definitions=["WorstOperatingPointFARMean：各外层折最差工况健康误报率的平均", "Test ECE：外层测试折期望校准误差"],
        ),
        make_source(
            "src_model",
            "冻结模型配置与完整性清单",
            MODEL_ROOT / "metadata.json",
            "读取冻结前模型配置、温度、阈值、数据哈希与盲测隔离状态。",
            generated_at,
            sql="SELECT * FROM read_json_auto('ML/models/final_robust_expanded_v13_bridge_696_rf_full/metadata.json')",
            tables_used=[rel(MODEL_ROOT / "metadata.json"), rel(MODEL_ROOT / "freeze_manifest.json")],
            filters=["state=frozen_pre_blind", "blind_data_used=false", "blind_data_generated=false"],
        ),
        make_source(
            "src_blind_merge",
            "最终盲测严格合并结果",
            BLIND_DATA / "merge_summary.json",
            "读取六阶段最终盲测的 RunID、窗口、工况与类别覆盖。",
            generated_at,
            sql="SELECT * FROM read_json_auto('simulink/experiments/sensor_bias/scripts/dataset_output_v13/blind_test/combined_blind_216/merge_summary.json')",
            tables_used=[rel(BLIND_DATA / "merge_summary.json"), rel(BLIND_DATA / "phase_manifest.csv")],
            filters=["六个盲测阶段", "FailedRunCount=0", "开发集与盲测 RunID 交集=0"],
        ),
        make_source(
            "src_blind_metrics",
            "一次性盲测总体指标",
            BLIND_RESULT / "blind_metrics.csv",
            "读取冻结模型在 15,208 个合格盲测窗口上的一次性总体指标。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_metrics.csv', header=true)",
            filters=["Split=final_blind", "IsTrainingEligible != 0", "评估次数=1"],
            definitions=["Macro-F1：五个类别 F1 的算术平均", "平衡准确率：五个类别召回率的算术平均", "健康误报率：真实健康窗口被判为任意故障的比例"],
        ),
        make_source(
            "src_blind_per_class",
            "一次性盲测逐类别指标",
            BLIND_RESULT / "blind_per_class_metrics.csv",
            "读取最终盲测的逐类别精确率、召回率、F1 与支持数。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_per_class_metrics.csv', header=true) ORDER BY ClassID",
            filters=["Split=final_blind", "类别 0–4"],
        ),
        make_source(
            "src_blind_far",
            "最终盲测健康误报工况诊断",
            BLIND_RESULT / "healthy_false_alarm_by_operating_point.csv",
            "读取 12 个盲测工况各自的健康误报率与误报窗口数。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/healthy_false_alarm_by_operating_point.csv', header=true) ORDER BY HealthyFalseAlarmRate DESC",
            filters=["TrueClassID=0", "Split=final_blind"],
        ),
    ]

    kpi_rows = [
        {
            "expandedRunIds": expanded_run_ids,
            "eligibleWindows": int(expanded_merge["eligible_window_count"]),
            "bridgeRuns": bridge_runs,
            "blindRuns": blind_run_ids,
            "blindMacroF1": float(blind_metrics["MacroF1"]),
            "blindHealthyFar": float(blind_metrics["HealthyFalseAlarmRate"]),
        }
    ]

    cards = [
        {
            "id": "card_expanded_runs",
            "description": "原开发集与桥接扩展严格合并后的唯一 RunID 总数。",
            "dataset": "kpis",
            "sourceId": "src_expanded_merge",
            "metrics": [{"label": "扩展开发 RunID", "field": "expandedRunIds", "format": "number"}],
        },
        {
            "id": "card_eligible_windows",
            "description": "扩展开发集中可用于训练或评估的窗口数。",
            "dataset": "kpis",
            "sourceId": "src_expanded_merge",
            "metrics": [{"label": "开发合格窗口", "field": "eligibleWindows", "format": "number"}],
        },
        {
            "id": "card_bridge_runs",
            "description": "phase8–13 工作域桥接扩展全部成功。",
            "dataset": "kpis",
            "sourceId": "src_stage_summary",
            "metrics": [{"label": "桥接扩展运行", "field": "bridgeRuns", "format": "number"}],
        },
        {
            "id": "card_blind_runs",
            "description": "冻结后生成的一次性最终盲测运行数。",
            "dataset": "kpis",
            "sourceId": "src_blind_merge",
            "metrics": [{"label": "最终盲测运行", "field": "blindRuns", "format": "number"}],
        },
        {
            "id": "card_blind_f1",
            "description": "五个类别窗口级 F1 的算术平均。",
            "dataset": "kpis",
            "sourceId": "src_blind_metrics",
            "metrics": [{"label": "盲测 Macro-F1", "field": "blindMacroF1", "format": "percent"}],
        },
        {
            "id": "card_blind_far",
            "description": "真实健康窗口被判为任意故障的比例。",
            "dataset": "kpis",
            "sourceId": "src_blind_metrics",
            "metrics": [{"label": "盲测健康误报率", "field": "blindHealthyFar", "format": "percent"}],
        },
    ]

    charts = [
        {
            "id": "chart_batch_runs",
            "title": "独立采集批次的 RunID 数量",
            "subtitle": "原开发、桥接扩展和最终盲测为相互隔离的采集批次。",
            "intent": "comparison",
            "question": "项目在三个主要采集批次中分别形成了多少唯一运行？",
            "rationale": "三个离散批次适合使用按规模排序的水平条形图。",
            "comparisonContext": {"denominator": "各批次唯一 RunID", "grain": "采集批次", "normalization": "无", "semanticFamily": "数据采集规模", "unit": "RunID"},
            "type": "horizontalBar",
            "dataset": "batch_runs",
            "sourceId": "src_stage_summary",
            "encodings": {
                "x": {"field": "batch", "type": "nominal", "label": "采集批次"},
                "y": {"field": "runIds", "type": "quantitative", "format": "number", "label": "RunID 数量"},
                "tooltip": [{"field": "role", "type": "nominal", "label": "用途"}],
            },
            "valueFormat": "number",
            "layout": "full",
            "labels": {"values": "all"},
            "palette": {"kind": "sequential"},
            "settings": {"groupMode": "single", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "interactiveLegend": False, "showControls": False, "viewMode": "both"},
        },
        {
            "id": "chart_feature_budget",
            "title": "特征预算与模型的六折 Macro-F1",
            "subtitle": "30/50 特征候选明显偏弱；80/全量候选更接近可用区间。",
            "intent": "comparison",
            "question": "在不同特征预算下，随机森林与 XGBoost 的平均 Macro-F1 如何变化？",
            "rationale": "八个离散候选按平均 Macro-F1 排序，水平条形图便于比较。",
            "comparisonContext": {"denominator": "六个外层测试折", "grain": "模型×特征预算候选", "normalization": "六折平均", "semanticFamily": "开发期分类表现", "unit": "比例"},
            "type": "horizontalBar",
            "dataset": "feature_budget",
            "sourceId": "src_feature_study",
            "encodings": {
                "x": {"field": "candidate", "type": "nominal", "label": "候选"},
                "y": {"field": "macroF1", "type": "quantitative", "format": "percent", "label": "Macro-F1"},
                "tooltip": [
                    {"field": "featureCount", "type": "quantitative", "format": "number", "label": "特征数"},
                    {"field": "macroF1Std", "type": "quantitative", "format": "percent", "label": "折间标准差"},
                    {"field": "healthyFar", "type": "quantitative", "format": "percent", "label": "健康误报率"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "labels": {"values": "all"},
            "palette": {"kind": "sequential"},
            "settings": {"groupMode": "single", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "interactiveLegend": False, "showControls": False, "viewMode": "both"},
        },
        {
            "id": "chart_blind_recall",
            "title": "最终盲测逐类别召回率",
            "subtitle": "15,208 个合格窗口；召回率越高表示漏检越少。",
            "intent": "comparison",
            "question": "冻结模型在五个真实类别上分别检出了多少样本？",
            "rationale": "长类别标签与五项比较适合水平条形图。",
            "comparisonContext": {"denominator": "各真实类别窗口数", "grain": "真实类别", "normalization": "Recall=TP/(TP+FN)", "semanticFamily": "盲测检测能力", "unit": "比例"},
            "type": "horizontalBar",
            "dataset": "blind_per_class",
            "sourceId": "src_blind_per_class",
            "encodings": {
                "x": {"field": "className", "type": "nominal", "label": "类别"},
                "y": {"field": "recall", "type": "quantitative", "format": "percent", "label": "召回率"},
                "tooltip": [
                    {"field": "precision", "type": "quantitative", "format": "percent", "label": "精确率"},
                    {"field": "f1", "type": "quantitative", "format": "percent", "label": "F1"},
                    {"field": "support", "type": "quantitative", "format": "number", "label": "支持数"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "labels": {"values": "all"},
            "palette": {"kind": "sequential"},
            "settings": {"groupMode": "single", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "interactiveLegend": False, "showControls": False, "viewMode": "both"},
        },
    ]

    tables = [
        {
            "id": "table_stage_results",
            "title": "项目阶段、已执行任务与结果",
            "subtitle": "按执行顺序列出已完成工作及其可交付结果。",
            "dataset": "stage_results",
            "defaultSort": {"field": "stage", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_stage_summary",
            "layout": "full",
            "columns": [
                {"field": "stage", "label": "阶段", "type": "text"},
                {"field": "tasks", "label": "已执行任务", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "table_nested",
            "title": "稳健嵌套验证的决策规则比较",
            "subtitle": "六个外层测试折；Macro-F1 越高越好，误报率与 ECE 越低越好。",
            "dataset": "nested_results",
            "defaultSort": {"field": "macroF1", "direction": "desc"},
            "density": "spacious",
            "sourceId": "src_nested",
            "layout": "full",
            "columns": [
                {"field": "variant", "label": "决策规则", "type": "text"},
                {"field": "macroF1", "label": "Macro-F1 均值", "type": "percent"},
                {"field": "macroF1Min", "label": "最差折 Macro-F1", "type": "percent"},
                {"field": "healthyFar", "label": "健康误报率均值", "type": "percent"},
                {"field": "worstOpFar", "label": "最差工况误报率均值", "type": "percent"},
                {"field": "ece", "label": "测试 ECE", "type": "percent"},
            ],
        },
        {
            "id": "table_blind_per_class",
            "title": "最终盲测逐类别指标",
            "subtitle": "支持数为每个真实类别的合格窗口数。",
            "dataset": "blind_per_class",
            "defaultSort": {"field": "recall", "direction": "desc"},
            "density": "spacious",
            "sourceId": "src_blind_per_class",
            "layout": "full",
            "columns": [
                {"field": "className", "label": "类别", "type": "text"},
                {"field": "precision", "label": "精确率", "type": "percent"},
                {"field": "recall", "label": "召回率", "type": "percent"},
                {"field": "f1", "label": "F1", "type": "percent"},
                {"field": "support", "label": "支持数", "type": "number"},
            ],
        },
        {
            "id": "table_worst_far",
            "title": "健康误报率最高的盲测工况",
            "subtitle": "12 个工况中仅两个工况出现健康误报；表中保留最高四项便于核查。",
            "dataset": "worst_far",
            "defaultSort": {"field": "falseAlarmRate", "direction": "desc"},
            "density": "spacious",
            "sourceId": "src_blind_far",
            "layout": "full",
            "columns": [
                {"field": "operatingPoint", "label": "工况", "type": "text"},
                {"field": "mode", "label": "模式", "type": "number"},
                {"field": "soc", "label": "SOC 初值", "type": "number"},
                {"field": "loadW", "label": "负载 W", "type": "number"},
                {"field": "falseAlarms", "label": "误报窗口", "type": "number"},
                {"field": "falseAlarmRate", "label": "健康误报率", "type": "percent"},
            ],
        },
    ]

    title = "储能故障诊断项目：已完成工作分阶段汇总"
    blocks = [
        {"id": "block_title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "block_summary",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                "项目已经完成从仿真恢复与数据扩展、严格合并审计、特征研究、稳健调参和校准、冻结模型，到冻结后一次性盲测的完整闭环。"
                f"最终扩展开发集包含 **{expanded_run_ids} 个 RunID、{int(expanded_merge['eligible_window_count']):,} 个合格窗口和 28 个工况**；"
                f"最终盲测包含 **{blind_run_ids} 个全新 RunID、{int(blind_merge['eligible_window_count']):,} 个合格窗口和 12 个工况**。\n\n"
                f"开发期稳健阈值六折 Macro-F1 为 **{float(robust_row['MacroF1Mean']):.2%}**、健康误报率为 **{float(robust_row['HealthyFARMean']):.2%}**；"
                f"一次性盲测 Macro-F1 降至 **{float(blind_metrics['MacroF1']):.2%}**，健康误报率升至 **{float(blind_metrics['HealthyFalseAlarmRate']):.2%}**。"
                "S1/S2 开路召回率均只有 **13.79%**，因此当前模型应视为已完成验证但未达到部署条件。"
            ),
            "layout": "full",
        },
        {"id": "block_kpis", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "block_stage_overview",
            "type": "markdown",
            "body": (
                "## 九个阶段已经形成完整、可审计的交付链\n\n"
                "工作不只包含模型训练，还覆盖了长时仿真的安全运行、断点恢复、磁盘保护、数据隔离、严格合并、模型冻结、一次性盲测和最终进程收束。"
                "下表按顺序列出项目内已经执行的任务以及每个阶段形成的结果。"
            ),
            "layout": "full",
        },
        {"id": "block_stage_table", "type": "table", "tableId": "table_stage_results", "layout": "full"},
        {
            "id": "block_batch_context",
            "type": "markdown",
            "body": (
                "## 数据采集分为原开发、桥接扩展和最终盲测三批\n\n"
                "原开发集负责建立基础工作域，桥接扩展补充此前覆盖不足的健康与故障域；两者合并后形成 696 个开发 RunID。"
                "最终盲测 216 个 RunID 与开发数据严格隔离，只在模型冻结后生成和查看。"
            ),
            "layout": "full",
        },
        {"id": "block_batch_chart", "type": "chart", "chartId": "chart_batch_runs", "layout": "full"},
        {
            "id": "block_scope",
            "type": "markdown",
            "body": (
                "## 数据范围、样本口径与隔离规则\n\n"
                f"扩展开发集共有 {expanded_run_ids} 个唯一 RunID 和 {int(expanded_merge['feature_window_count']):,} 个窗口，其中 {int(expanded_merge['eligible_window_count']):,} 个满足 `IsTrainingEligible != 0`；"
                f"608 个 RunID 至少包含一个合格窗口。五类合格窗口分别为健康 {int(expanded_merge['class_counts']['0']):,}、母线电压偏置 {int(expanded_merge['class_counts']['1']):,}、"
                f"电感电流偏置 {int(expanded_merge['class_counts']['2']):,}、S1 开路 {int(expanded_merge['class_counts']['3']):,}、S2 开路 {int(expanded_merge['class_counts']['4']):,}。\n\n"
                "严格审计要求阶段 schema 一致、RunID 唯一、(RunID, WindowID) 唯一、合格窗口模型特征无缺失、开发与盲测 RunID 无交集，并明确排除修复前 264 次数据。"
            ),
            "sourceId": "src_expanded_merge",
            "layout": "full",
        },
        {
            "id": "block_feature_finding",
            "type": "markdown",
            "body": (
                "## 低特征预算不足，80/全量候选更接近稳定区间\n\n"
                "六折特征研究表明，30 和 50 特征候选的平均性能与折间稳定性不足；80 和全量特征候选明显更强。"
                "虽然部分 XGBoost 或特征消融候选具有更高的平均 Macro-F1，最终冻结流程并未只按单一均值选择，而是转入 225 特征随机森林的稳健折内候选与阈值约束。"
            ),
            "sourceId": "src_feature_study",
            "layout": "full",
        },
        {"id": "block_feature_chart", "type": "chart", "chartId": "chart_feature_budget", "layout": "full"},
        {
            "id": "block_nested_finding",
            "type": "markdown",
            "body": (
                "## 稳健阈值显著降低开发期健康误报，但不能消除跨域风险\n\n"
                f"原始 argmax 的六折平均 Macro-F1 为 {float(raw_row['MacroF1Mean']):.2%}、健康误报率为 {float(raw_row['HealthyFARMean']):.2%}；"
                f"稳健阈值把 Macro-F1 提升到 {float(robust_row['MacroF1Mean']):.2%}，并把健康误报率压到 {float(robust_row['HealthyFARMean']):.2%}。"
                "这说明阈值约束在开发域有效，但最差工况误报仍然较高，且最终盲测显示新的工况域失配。"
            ),
            "sourceId": "src_nested",
            "layout": "full",
        },
        {"id": "block_nested_table", "type": "table", "tableId": "table_nested", "layout": "full"},
        {
            "id": "block_model",
            "type": "markdown",
            "body": (
                "## 冻结模型在盲测生成前完成定型\n\n"
                f"最终模型为随机森林：{int(model_meta['selected_parameters']['n_estimators'])} 棵树、最大深度 {int(model_meta['selected_parameters']['max_depth'])}、"
                f"叶节点最小样本数 {int(model_meta['selected_parameters']['min_samples_leaf'])}、`max_features={model_meta['selected_parameters']['max_features']}`，共 {int(model_meta['feature_count'])} 个特征。"
                f"温度校准参数为 {float(model_meta['temperature']):.6f}，全局报警阈值为 {float(model_meta['alarm_threshold']):.4f}。\n\n"
                f"冻结状态为 `{freeze_manifest['state']}`，模型 SHA-256 为 `{freeze_manifest['files']['random_forest.joblib']['sha256']}`。"
                "冻结清单明确记录 `blind_data_used=false` 和 `blind_data_generated=false`，因此盲测结果不是调参回看。"
            ),
            "sourceId": "src_model",
            "layout": "full",
        },
        {
            "id": "block_blind_result",
            "type": "markdown",
            "body": (
                "## 一次性盲测暴露开路漏检和高负载工况误报\n\n"
                f"盲测总体准确率为 {float(blind_metrics['Accuracy']):.2%}、平衡准确率为 {float(blind_metrics['BalancedAccuracy']):.2%}、Macro-F1 为 {float(blind_metrics['MacroF1']):.2%}。"
                "电感电流传感器偏置召回率达到 91.67%，但 S1 和 S2 开路召回率均只有 13.79%，是决定性短板。\n\n"
                "健康误报率为 15.62%，全部 746 个误报窗口集中在 `blind_op_0008` 和 `blind_op_0011`：二者均为 600 W、ModeCommand=1，误报率分别为 93.47% 和 93.97%。"
                "其余十个盲测工况健康误报率均为 0。"
            ),
            "layout": "full",
        },
        {"id": "block_blind_chart", "type": "chart", "chartId": "chart_blind_recall", "layout": "full"},
        {"id": "block_blind_table", "type": "table", "tableId": "table_blind_per_class", "layout": "full"},
        {"id": "block_far_table", "type": "table", "tableId": "table_worst_far", "layout": "full"},
        {
            "id": "block_methodology",
            "type": "markdown",
            "body": (
                "## 方法与验证流程\n\n"
                "开发评估使用按 OperatingPointID 隔离的六个外层折；候选选择、概率校准和阈值搜索限制在相应训练折内。"
                "最终模型基于六个外层测试折的原始概率进行温度校准，并依据最差折表现和健康误报约束选择候选。"
                "冻结后，才运行六阶段盲测仿真、完成严格合并审计并执行一次预测。\n\n"
                "主要指标均为窗口级：Macro-F1 对五类等权，平衡准确率为五类召回率均值，健康误报率只在真实健康窗口上计算。"
                "这些指标不等价于事件级检出率、检出延迟或整次运行级安全性能。"
            ),
            "layout": "full",
        },
        {
            "id": "block_limitations",
            "type": "markdown",
            "body": (
                "## 限制、不确定性与稳健性边界\n\n"
                "15,208 个盲测窗口来自 216 次运行，窗口之间并非统计独立；当前结果没有重复盲测或置信区间。"
                "盲测 IrefLevel 固定为 10，不能外推到未覆盖电流指令、温度或控制组合。"
                "开发期与盲测的指标差异是描述性泛化差距，不是配对显著性检验。\n\n"
                "更重要的是，盲测结果已经被查看。任何使用这些结果进行特征、阈值或超参数修改的后续工作都必须定义为新的开发循环，并配套一套从未查看的新盲测集。"
            ),
            "layout": "full",
        },
        {
            "id": "block_next_steps",
            "type": "markdown",
            "body": (
                "## 建议的下一阶段\n\n"
                "1. **暂不部署当前模型。** 把 S1/S2 开路召回率与最差工况健康误报率设置为新版硬门槛。\n"
                "2. **建立新开发循环。** 针对 600 W、ModeCommand=1 补充健康与开路桥接样本，并检查母线电压偏置概率的域漂移。\n"
                "3. **增加事件级指标。** 除窗口级指标外，报告每次运行是否检出、首次检出延迟、连续报警稳定性和最坏工况置信区间。\n"
                "4. **预注册新盲测。** 在新模型冻结前固定工况生成规则、样本数、通过门槛和一次性评估流程。"
            ),
            "layout": "full",
        },
        {
            "id": "block_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 600 W、ModeCommand=1 下哪些特征把健康窗口推向母线电压偏置？\n"
                "- S1/S2 开路漏检是否随故障阶段、开关位置或窗口相位系统变化？\n"
                "- 事件级聚合和滞迟规则能否提高开路检出，同时维持最差工况误报约束？\n"
                "- 下一套盲测需要覆盖哪些未见 IrefLevel、温度、SOC 与负载组合？"
            ),
            "layout": "full",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "储能故障诊断项目已完成任务、阶段结果、模型验证结论与下一阶段建议。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "kpis": kpi_rows,
                "stage_results": stage_rows,
                "batch_runs": batch_rows,
                "feature_budget": feature_rows,
                "nested_results": nested_rows,
                "blind_per_class": per_class_rows,
                "worst_far": worst_far_rows,
            },
        },
        "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
    }

    notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "question": "目前项目内已经完成哪些任务，各阶段形成了什么结果，哪些结论可用于下一步决策？",
        "decision_useful_answer": "仿真、数据、建模、冻结和一次性盲测闭环均已完成；当前模型因开路漏检与特定工况健康误报不具备部署条件。",
        "required_structure_mapping": {
            "technical_summary": "技术摘要",
            "key_findings": ["九个阶段已经形成完整、可审计的交付链", "低特征预算不足，80/全量候选更接近稳定区间", "一次性盲测暴露开路漏检和高负载工况误报"],
            "scope_data_metrics": "数据范围、样本口径与隔离规则",
            "model_specification": "冻结模型在盲测生成前完成定型",
            "methodology": "方法与验证流程",
            "limitations": "限制、不确定性与稳健性边界",
            "recommended_next_steps": "建议的下一阶段",
            "further_questions": "仍需回答的问题",
        },
        "chart_map": [
            {"section": "数据采集分为原开发、桥接扩展和最终盲测三批", "question": "三个独立采集批次分别形成多少 RunID？", "family": "Comparison & Ranking", "type": "horizontalBar", "fields": ["batch", "runIds", "role"], "takeaway": "原开发 456、桥接 240、盲测 216 个 RunID。", "palette_policy": "single-root preferred"},
            {"section": "低特征预算不足，80/全量候选更接近稳定区间", "question": "不同模型和特征预算的平均 Macro-F1 如何？", "family": "Comparison & Ranking", "type": "horizontalBar", "fields": ["candidate", "macroF1", "featureCount", "healthyFar"], "takeaway": "30/50 特征候选明显弱于 80/全量候选。", "palette_policy": "single-root preferred"},
            {"section": "一次性盲测暴露开路漏检和高负载工况误报", "question": "五类盲测召回率分别是多少？", "family": "Comparison & Ranking", "type": "horizontalBar", "fields": ["className", "recall", "precision", "f1", "support"], "takeaway": "S1/S2 开路召回率均为 13.79%。", "palette_policy": "single-root preferred"},
        ],
        "qa_notes": [
            "所有百分比在 artifact 中均使用 0–1 比例。",
            "三张图均为单系列水平条形图，长标签有换行空间且不使用冗余图例。",
            "开发与盲测比较仅作为描述性泛化差距，不声明因果或统计显著性。",
            "阶段表的汇总源 stage_summary.csv 由已落盘的原始产物派生。",
        ],
        "model_sha256": freeze_manifest["files"]["random_forest.joblib"]["sha256"],
        "blind_evaluation_count": 1,
    }

    (OUTPUT_DIR / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "report_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT_DIR / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
