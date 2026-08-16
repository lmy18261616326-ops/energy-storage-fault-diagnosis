#!/usr/bin/env python
"""Build the canonical Chinese artifact for the final frozen-model report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ML"
RESULTS_ROOT = ML_ROOT / "results"
MODEL_ROOT = ML_ROOT / "models" / "final_robust_expanded_v13_bridge_696_rf_full"
BLIND_RESULT = RESULTS_ROOT / "final_blind_expanded_v13_bridge_696_rf_full"
NESTED_RESULT = RESULTS_ROOT / "robust_nested_expanded_v13_bridge_696_rf_full"
BLIND_DATA = (
    PROJECT_ROOT
    / "simulink"
    / "experiments"
    / "sensor_bias"
    / "scripts"
    / "dataset_output_v13"
    / "blind_test"
    / "combined_blind_216"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "reports" / "final_energy_storage_ml_report_2026-08-01"


CLASS_NAMES_ZH = {
    0: "健康",
    1: "母线电压传感器偏置",
    2: "电感电流传感器偏置",
    3: "S1 开路",
    4: "S2 开路",
}

PHASE_NAMES_ZH = {
    "health": "健康",
    "sensor_faults": "传感器故障",
    "switch_full_open": "开关完全开路",
    "switch_partial_open": "开关部分开路",
    "switch_intermittent": "开关间歇开路",
    "switch_high_resistance": "开关高阻",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    generated_at: str,
    *,
    sql: str | None = None,
    filters: list[str] | None = None,
    definitions: list[str] | None = None,
    tables_used: list[str] | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "engine": "DuckDB" if sql else "Python/pandas",
        "language": "SQL" if sql else "Python",
        "description": description,
        "executed_at": generated_at,
    }
    if sql:
        query["sql"] = sql
    if filters:
        query["filters"] = filters
    if definitions:
        query["metric_definitions"] = definitions
    if tables_used:
        query["tables_used"] = tables_used
    return {
        "id": source_id,
        "label": label,
        "path": rel(path),
        "query": query,
    }


def main() -> int:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    metrics = pd.read_csv(BLIND_RESULT / "blind_metrics.csv").iloc[0]
    per_class = pd.read_csv(BLIND_RESULT / "blind_per_class_metrics.csv")
    far_by_op = pd.read_csv(BLIND_RESULT / "healthy_false_alarm_by_operating_point.csv")
    blind_meta = json.loads(
        (BLIND_RESULT / "blind_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    merge_summary = json.loads(
        (BLIND_DATA / "merge_summary.json").read_text(encoding="utf-8")
    )
    phase_manifest = pd.read_csv(BLIND_DATA / "phase_manifest.csv")
    model_meta = json.loads((MODEL_ROOT / "metadata.json").read_text(encoding="utf-8"))
    freeze_manifest = json.loads(
        (MODEL_ROOT / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    nested = pd.read_csv(NESTED_RESULT / "summary.csv")
    nested_robust = nested.loc[nested["Variant"].eq("robust_threshold")].iloc[0]

    per_class_rows = []
    for row in per_class.to_dict(orient="records"):
        class_id = int(row["ClassID"])
        per_class_rows.append(
            {
                "classId": class_id,
                "className": str(row["ClassName"]),
                "classNameZh": CLASS_NAMES_ZH[class_id],
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "support": int(row["Support"]),
            }
        )

    far_rows = []
    for row in far_by_op.sort_values("HealthyFalseAlarmRate", ascending=False).to_dict(
        orient="records"
    ):
        far_rows.append(
            {
                "operatingPointId": str(row["OperatingPointID"]),
                "modeCommand": int(row["ModeCommand"]),
                "socInit": int(row["SOCInit"]),
                "pload": float(row["Pload"]),
                "healthyWindowCount": int(row["HealthyWindowCount"]),
                "falseAlarmCount": int(row["FalseAlarmCount"]),
                "healthyFalseAlarmRate": float(row["HealthyFalseAlarmRate"]),
            }
        )

    phase_rows = []
    for row in phase_manifest.to_dict(orient="records"):
        phase = str(row["Phase"])
        phase_rows.append(
            {
                "phase": phase,
                "phaseNameZh": PHASE_NAMES_ZH[phase],
                "runCount": int(row["RunCount"]),
                "featureWindowCount": int(row["FeatureWindowCount"]),
                "eligibleWindowCount": int(row["EligibleWindowCount"]),
                "failedRunCount": int(row["FailedRunCount"]),
            }
        )

    confusion = blind_meta["confusion_matrix"]
    confusion_rows = []
    for true_id, values in enumerate(confusion):
        confusion_rows.append(
            {
                "trueClass": CLASS_NAMES_ZH[true_id],
                "predHealthy": int(values[0]),
                "predVbusBias": int(values[1]),
                "predCurrentBias": int(values[2]),
                "predS1Open": int(values[3]),
                "predS2Open": int(values[4]),
            }
        )

    validation_rows = [
        {
            "stage": "六折嵌套开发评估",
            "macroF1": float(nested_robust["MacroF1Mean"]),
            "healthyFalseAlarmRate": float(nested_robust["HealthyFARMean"]),
            "sampleCount": int(model_meta["development_windows"]),
            "note": "六个外层测试折的均值；阈值在折内选择",
        },
        {
            "stage": "冻结后一次性盲测",
            "macroF1": float(metrics["MacroF1"]),
            "healthyFalseAlarmRate": float(metrics["HealthyFalseAlarmRate"]),
            "sampleCount": int(metrics["SampleCount"]),
            "note": "216 次全新仿真、12 个盲测工况；仅评估一次",
        },
    ]

    audit_rows = [
        {"check": "盲测仿真运行数", "result": "通过", "detail": "216/216，失败 0"},
        {"check": "RunID 唯一性", "result": "通过", "detail": "216 个唯一 RunID"},
        {"check": "窗口键唯一性", "result": "通过", "detail": "42,984 个 (RunID, WindowID)，重复 0"},
        {"check": "开发集隔离", "result": "通过", "detail": "与开发集 RunID 交集为 0"},
        {"check": "特征模式", "result": "通过", "detail": "六阶段 344 列一致；冻结模型 225 列全部存在"},
        {"check": "冻结特征完整性", "result": "通过", "detail": "15,208 个合格窗口的模型特征无缺失"},
        {"check": "模型完整性", "result": "通过", "detail": "SHA-256 与冻结清单一致"},
        {"check": "盲测次数", "result": "通过", "detail": "评估输出首次创建；未重复运行"},
    ]

    source_defs = [
        source(
            "src_blind_metrics",
            "一次性盲测总体指标",
            BLIND_RESULT / "blind_metrics.csv",
            "读取冻结随机森林在最终盲测合格窗口上的总体分类指标。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_metrics.csv', header=true)",
            filters=["IsTrainingEligible != 0", "Split=final_blind"],
            definitions=[
                "Macro-F1=五个类别 F1 的算术平均",
                "健康误报率=真实健康窗口被预测为任意故障的比例",
                "平衡准确率=五个类别召回率的算术平均",
            ],
        ),
        source(
            "src_blind_per_class",
            "一次性盲测逐类别指标",
            BLIND_RESULT / "blind_per_class_metrics.csv",
            "读取最终盲测的逐类别精确率、召回率、F1 和支持数。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_per_class_metrics.csv', header=true) ORDER BY ClassID",
            filters=["Split=final_blind", "类别 0-4"],
            definitions=["逐类 F1=2×Precision×Recall/(Precision+Recall)"],
        ),
        source(
            "src_blind_far_by_op",
            "盲测健康误报按工况",
            BLIND_RESULT / "healthy_false_alarm_by_operating_point.csv",
            "读取 12 个盲测工况各自的健康误报数与误报率。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/healthy_false_alarm_by_operating_point.csv', header=true) ORDER BY HealthyFalseAlarmRate DESC",
            filters=["TrueClassID=0", "Split=final_blind"],
            definitions=["工况健康误报率=FalseAlarmCount/HealthyWindowCount"],
        ),
        source(
            "src_blind_metadata",
            "盲测混淆矩阵与样本元数据",
            BLIND_RESULT / "blind_evaluation_metadata.json",
            "读取一次性盲测样本数、工况数和五分类混淆矩阵。",
            generated_at,
            sql="SELECT * FROM read_json_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_evaluation_metadata.json')",
        ),
        source(
            "src_blind_merge",
            "盲测六阶段严格合并",
            BLIND_DATA / "merge_summary.json",
            "读取盲测六阶段合并后的运行数、窗口数、类别覆盖与工况覆盖。",
            generated_at,
            sql="SELECT * FROM read_json_auto('simulink/experiments/sensor_bias/scripts/dataset_output_v13/blind_test/combined_blind_216/merge_summary.json')",
            tables_used=[
                rel(BLIND_DATA / "merge_summary.json"),
                rel(BLIND_DATA / "phase_manifest.csv"),
            ],
            filters=["六个盲测阶段", "拒绝重复 RunID", "拒绝重复窗口键", "要求特征列顺序完全一致"],
        ),
        source(
            "src_phase_manifest",
            "盲测六阶段清单",
            BLIND_DATA / "phase_manifest.csv",
            "读取每个盲测阶段的运行数、窗口数、合格窗口数和失败数。",
            generated_at,
            sql="SELECT * FROM read_csv_auto('simulink/experiments/sensor_bias/scripts/dataset_output_v13/blind_test/combined_blind_216/phase_manifest.csv', header=true) ORDER BY Phase",
            filters=["六个盲测阶段", "FailedRunCount=0"],
        ),
        source(
            "src_model_metadata",
            "冻结模型配置",
            MODEL_ROOT / "metadata.json",
            "读取冻结随机森林的超参数、特征数、温度校准和报警阈值。",
            generated_at,
            sql="SELECT * FROM read_json_auto('ML/models/final_robust_expanded_v13_bridge_696_rf_full/metadata.json')",
            tables_used=[rel(MODEL_ROOT / "metadata.json"), rel(MODEL_ROOT / "freeze_manifest.json")],
            filters=["state=frozen_pre_blind", "blind_data_used=false at freeze time"],
        ),
        source(
            "src_validation_comparison",
            "开发评估与盲测比较",
            NESTED_RESULT / "summary.csv",
            "将六折稳健阈值开发评估与冻结后一次性盲测置于同一比例尺度比较。",
            generated_at,
            sql=(
                "SELECT '六折嵌套开发评估' AS Stage, MacroF1Mean AS MacroF1, HealthyFARMean AS HealthyFalseAlarmRate "
                "FROM read_csv_auto('ML/results/robust_nested_expanded_v13_bridge_696_rf_full/summary.csv', header=true) "
                "WHERE Variant='robust_threshold' UNION ALL "
                "SELECT '冻结后一次性盲测', MacroF1, HealthyFalseAlarmRate "
                "FROM read_csv_auto('ML/results/final_blind_expanded_v13_bridge_696_rf_full/blind_metrics.csv', header=true)"
            ),
            tables_used=[
                rel(NESTED_RESULT / "summary.csv"),
                rel(BLIND_RESULT / "blind_metrics.csv"),
            ],
            filters=["开发行 Variant=robust_threshold", "盲测行 Split=final_blind"],
            definitions=[
                "开发 Macro-F1 与健康误报率为六个外层测试折均值",
                "盲测指标基于 15,208 个合格窗口的一次性评估",
            ],
        ),
    ]

    cards = [
        {
            "id": "card_blind_runs",
            "description": "六类盲测仿真全部成功，并完成严格合并。",
            "dataset": "kpis",
            "sourceId": "src_blind_merge",
            "metrics": [
                {"label": "盲测运行数", "field": "blindRuns", "format": "number"},
                {"label": "合格窗口", "field": "eligibleWindows", "format": "number"},
            ],
        },
        {
            "id": "card_macro_f1",
            "description": "五个类别 F1 的算术平均；最终盲测仅执行一次。",
            "dataset": "kpis",
            "sourceId": "src_blind_metrics",
            "metrics": [
                {"label": "盲测 Macro-F1", "field": "macroF1", "format": "percent"},
                {"label": "开发六折均值", "field": "developmentMacroF1", "format": "percent"},
            ],
        },
        {
            "id": "card_balanced_accuracy",
            "description": "五类召回率等权平均，避免类别数量差异主导结果。",
            "dataset": "kpis",
            "sourceId": "src_blind_metrics",
            "metrics": [
                {"label": "平衡准确率", "field": "balancedAccuracy", "format": "percent"},
                {"label": "总体准确率", "field": "accuracy", "format": "percent"},
            ],
        },
        {
            "id": "card_healthy_far",
            "description": "真实健康窗口被判为任意故障的比例。",
            "dataset": "kpis",
            "sourceId": "src_blind_metrics",
            "metrics": [
                {"label": "盲测健康误报率", "field": "healthyFar", "format": "percent"},
                {"label": "开发六折均值", "field": "developmentHealthyFar", "format": "percent"},
            ],
        },
    ]

    charts = [
        {
            "id": "chart_class_recall",
            "title": "最终盲测逐类别召回率",
            "subtitle": "15,208 个合格窗口；数值越高越好。",
            "intent": "comparison",
            "question": "冻结模型在五个类别上分别检出了多少真实样本？",
            "rationale": "水平条形图适合比较五个长标签类别，并直接暴露开路故障漏检。",
            "comparisonContext": {
                "denominator": "每个类别的真实窗口数",
                "grain": "故障类别",
                "normalization": "Recall=TP/(TP+FN)",
                "semanticFamily": "盲测类别检出能力",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "per_class",
            "sourceId": "src_blind_per_class",
            "encodings": {
                "x": {"field": "classNameZh", "type": "nominal", "label": "类别"},
                "y": {"field": "recall", "type": "quantitative", "format": "percent", "label": "召回率"},
                "tooltip": [
                    {"field": "precision", "type": "quantitative", "format": "percent", "label": "精确率"},
                    {"field": "f1", "type": "quantitative", "format": "percent", "label": "F1"},
                    {"field": "support", "type": "quantitative", "format": "number", "label": "窗口数"},
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
            "id": "chart_far_by_op",
            "title": "最终盲测健康误报率按工况",
            "subtitle": "每个工况 398 个健康窗口；数值越低越好。",
            "intent": "comparison",
            "question": "健康误报是否集中在少数盲测工况？",
            "rationale": "按误报率排序的水平条形图可直接识别工况集中性。",
            "comparisonContext": {
                "denominator": "每个工况 398 个健康窗口",
                "grain": "OperatingPointID",
                "normalization": "误报窗口数/健康窗口数",
                "semanticFamily": "健康误报",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "far_by_op",
            "sourceId": "src_blind_far_by_op",
            "encodings": {
                "x": {"field": "operatingPointId", "type": "nominal", "label": "盲测工况"},
                "y": {"field": "healthyFalseAlarmRate", "type": "quantitative", "format": "percent", "label": "健康误报率"},
                "tooltip": [
                    {"field": "falseAlarmCount", "type": "quantitative", "format": "number", "label": "误报窗口数"},
                    {"field": "modeCommand", "type": "quantitative", "format": "number", "label": "模式"},
                    {"field": "socInit", "type": "quantitative", "format": "number", "label": "SOC 初值"},
                    {"field": "pload", "type": "quantitative", "format": "number", "label": "负载功率"},
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
            "id": "table_per_class",
            "title": "最终盲测逐类别精确指标",
            "subtitle": "按召回率降序；Support 为真实类别窗口数。",
            "dataset": "per_class",
            "defaultSort": {"field": "recall", "direction": "desc"},
            "density": "spacious",
            "sourceId": "src_blind_per_class",
            "layout": "full",
            "columns": [
                {"field": "classNameZh", "label": "类别", "type": "text"},
                {"field": "precision", "label": "精确率", "format": "percent"},
                {"field": "recall", "label": "召回率", "format": "percent"},
                {"field": "f1", "label": "F1", "format": "percent"},
                {"field": "support", "label": "窗口数", "format": "number"},
            ],
        },
        {
            "id": "table_validation_comparison",
            "title": "开发评估与最终盲测",
            "subtitle": "同为窗口级五分类指标，但样本来源与评估角色不同。",
            "dataset": "validation_comparison",
            "defaultSort": {"field": "stage", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_validation_comparison",
            "layout": "full",
            "columns": [
                {"field": "stage", "label": "评估阶段", "type": "text"},
                {"field": "macroF1", "label": "Macro-F1", "format": "percent"},
                {"field": "healthyFalseAlarmRate", "label": "健康误报率", "format": "percent"},
                {"field": "sampleCount", "label": "窗口数", "format": "number"},
                {"field": "note", "label": "口径", "type": "text"},
            ],
        },
        {
            "id": "table_phase_audit",
            "title": "盲测六阶段数据审计",
            "subtitle": "216 次仿真均成功；窗口数为合并前阶段特征表记录数。",
            "dataset": "phase_audit",
            "defaultSort": {"field": "phaseNameZh", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_phase_manifest",
            "layout": "full",
            "columns": [
                {"field": "phaseNameZh", "label": "阶段", "type": "text"},
                {"field": "runCount", "label": "运行数", "format": "number"},
                {"field": "featureWindowCount", "label": "全部窗口", "format": "number"},
                {"field": "eligibleWindowCount", "label": "合格窗口", "format": "number"},
                {"field": "failedRunCount", "label": "失败", "format": "number"},
            ],
        },
        {
            "id": "table_confusion",
            "title": "最终盲测混淆矩阵",
            "subtitle": "行是真实类别，列是模型预测类别；窗口级计数。",
            "dataset": "confusion_matrix",
            "defaultSort": {"field": "trueClass", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_blind_metadata",
            "layout": "full",
            "columns": [
                {"field": "trueClass", "label": "真实类别", "type": "text"},
                {"field": "predHealthy", "label": "预测健康", "format": "number"},
                {"field": "predVbusBias", "label": "预测母线偏置", "format": "number"},
                {"field": "predCurrentBias", "label": "预测电流偏置", "format": "number"},
                {"field": "predS1Open", "label": "预测 S1 开路", "format": "number"},
                {"field": "predS2Open", "label": "预测 S2 开路", "format": "number"},
            ],
        },
        {
            "id": "table_audit_checks",
            "title": "冻结与盲测隔离检查",
            "subtitle": "评估前执行；所有检查均通过后才创建盲测结果。",
            "dataset": "audit_checks",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_blind_merge",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "检查项", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "detail", "label": "证据", "type": "text"},
            ],
        },
    ]

    title = "储能故障诊断模型：冻结后一次性盲测报告"
    blocks = [
        {"id": "block_title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "block_summary",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                "冻结随机森林已在 216 次全新仿真组成的盲测集上完成唯一一次评估。"
                "窗口级 Macro-F1 为 **43.79%**，平衡准确率 **50.17%**，总体准确率 **42.28%**，"
                "健康误报率 **15.62%**。\n\n"
                "模型对电感电流传感器偏置保持较强识别能力，但 S1/S2 开路召回率均只有 **13.79%**；"
                "大多数开路窗口被判为健康或母线电压偏置。健康误报又高度集中于两个 600 W、模式 1 工况。"
                "因此当前模型不应直接进入保护或告警部署，应把这次盲测视为最终泛化结论，而不是新的调参数据。"
            ),
            "layout": "full",
        },
        {"id": "block_kpis", "type": "metric-strip", "cardIds": [c["id"] for c in cards], "layout": "full"},
        {
            "id": "block_key_findings",
            "type": "markdown",
            "body": (
                "## 开路故障漏检是决定性短板\n\n"
                "盲测逐类别结果差异明显：电感电流传感器偏置召回率为 91.67%，健康类为 84.38%，"
                "而 S1 与 S2 开路召回率都只有 13.79%。两类开路各有 4,352 个窗口，但模型分别只正确识别 600 个。\n\n"
                "这意味着整体准确率并不能代表保护价值。对开路类而言，当前阈值策略虽然压低了开发期误报，"
                "却在新域上产生了大量漏报；下一版本必须把开路召回和最坏工况误报同时作为硬约束。"
            ),
            "sourceId": "src_blind_per_class",
            "layout": "full",
        },
        {"id": "block_class_chart", "type": "chart", "chartId": "chart_class_recall", "layout": "full"},
        {"id": "block_class_table", "type": "table", "tableId": "table_per_class", "layout": "full"},
        {
            "id": "block_far_finding",
            "type": "markdown",
            "body": (
                "## 健康误报集中在两个高负载模式 1 工况\n\n"
                "746 个健康误报全部来自 blind_op_0008 与 blind_op_0011；两者负载均为 600 W、ModeCommand=1，"
                "误报率分别为 93.47% 和 93.97%。其余 10 个盲测工况的健康误报率均为 0。\n\n"
                "总体 15.62% 的健康误报率因此不是均匀退化，而是明显的工况域失配。后续应优先补充这两个"
                "模式/负载邻域的健康与故障桥接样本，并检查母线电压偏置概率在高负载模式 1 下的校准。"
            ),
            "sourceId": "src_blind_far_by_op",
            "layout": "full",
        },
        {"id": "block_far_chart", "type": "chart", "chartId": "chart_far_by_op", "layout": "full"},
        {
            "id": "block_generalization",
            "type": "markdown",
            "body": (
                "## 一次性盲测证实开发估计仍偏乐观\n\n"
                "六折稳健阈值开发评估的平均 Macro-F1 为 62.15%，健康误报率为 5.32%；"
                "最终盲测分别为 43.79% 和 15.62%。由于两者样本域不同，这不是严格配对显著性检验，"
                "但方向一致地表明跨域泛化弱于开发期估计。\n\n"
                "盲测结果已经被查看，后续任何基于它的特征、阈值或超参数修改都必须定义为新开发循环，"
                "并另行生成一套从未查看的新盲测集。"
            ),
            "layout": "full",
        },
        {"id": "block_validation_table", "type": "table", "tableId": "table_validation_comparison", "layout": "full"},
        {
            "id": "block_scope",
            "type": "markdown",
            "body": (
                "## 评估范围与指标口径\n\n"
                "盲测由健康、两类传感器偏置及四种开关故障采集阶段组成，共 216 个唯一 RunID、"
                "42,984 个窗口，其中 15,208 个满足训练/评估资格。合格窗口覆盖五个标签、12 个 OperatingPointID、"
                "ModeCommand 0/1/2 和 SOC 初值 45/60；本批次 IrefLevel 固定为 10。\n\n"
                "Macro-F1 对五类等权，健康误报率只在真实健康窗口上计算；逐类别召回率用于衡量漏检，"
                "精确率用于衡量某类告警的可信度。所有指标均为窗口级，不等价于事件级或整次运行级性能。"
            ),
            "sourceId": "src_blind_merge",
            "layout": "full",
        },
        {"id": "block_phase_table", "type": "table", "tableId": "table_phase_audit", "layout": "full"},
        {
            "id": "block_method",
            "type": "markdown",
            "body": (
                "## 冻结模型与评估方法\n\n"
                "最终模型为随机森林：350 棵树、最大深度 16、叶节点最小样本 3、特征采样 sqrt，"
                "使用 225 个在线可用特征。概率经过温度 1.1587 校准，稳健报警阈值为 0.9125。"
                "模型在生成盲测数据前已冻结，冻结清单记录 `blind_data_used=false`，模型文件 SHA-256 与清单一致。\n\n"
                "评估流程先完成六阶段仿真与阶段内汇总，再严格合并；随后检查失败记录、特征模式、RunID、窗口键、"
                "开发/盲测隔离和冻结特征完整性。全部通过后才首次创建评估目录并执行一次预测。"
            ),
            "sourceId": "src_model_metadata",
            "layout": "full",
        },
        {"id": "block_confusion_table", "type": "table", "tableId": "table_confusion", "layout": "full"},
        {"id": "block_audit_table", "type": "table", "tableId": "table_audit_checks", "layout": "full"},
        {
            "id": "block_limitations",
            "type": "markdown",
            "body": (
                "## 限制与不确定性\n\n"
                "本报告只给出一个冻结模型在一个一次性盲测集上的描述性泛化结果，没有置信区间或重复盲测。"
                "15,208 个窗口来自 216 次运行，窗口之间并非统计独立，因此不能把窗口数直接当作独立样本量。"
                "盲测 IrefLevel 固定为 10，结论不能外推到未覆盖电流指令；事件级检出延迟、连续告警稳定性和控制安全影响也尚未评估。\n\n"
                "健康误报集中在两个工况，说明总体均值对工况组成敏感。开路故障高精确率伴随低召回，"
                "表明模型只在少数非常明显窗口上触发该类，而不是已经可靠区分开路。"
            ),
            "layout": "full",
        },
        {
            "id": "block_next_steps",
            "type": "markdown",
            "body": (
                "## 建议的下一步\n\n"
                "1. **暂不部署当前模型。** 将 S1/S2 开路召回率与最坏工况健康误报率设为下一版本的硬门槛。\n"
                "2. **建立新的开发循环。** 把本次盲测暴露的高负载 ModeCommand=1 域失配作为开发证据，"
                "补充邻域健康、部分开路、高阻和完全开路样本，但不要在现有盲测分数上反复调阈值。\n"
                "3. **重做分层验证。** 在外层按 OperatingPointID 隔离的同时，报告事件级检出率、检出延迟、"
                "最坏工况误报和置信区间。\n"
                "4. **预先登记下一套盲测。** 在新模型冻结前固定工况生成规则、样本数、通过门槛和只评估一次的流程。"
            ),
            "layout": "full",
        },
        {
            "id": "block_questions",
            "type": "markdown",
            "body": (
                "## 后续需要回答的问题\n\n"
                "- 高负载 ModeCommand=1 下，哪些特征导致健康窗口被推向母线电压偏置？\n"
                "- 开路窗口被判为健康与母线电压偏置的比例，是否随故障阶段、开关位置或窗口相位系统变化？\n"
                "- 事件级聚合、迟滞或连续窗口规则能否在不抬高最坏工况误报的前提下改善开路检出？\n"
                "- 下一套盲测应覆盖哪些未见 IrefLevel、SOC、负载与温度组合，才能检验真正的工作域泛化？"
            ),
            "layout": "full",
        },
    ]

    kpi_row = {
        "blindRuns": int(merge_summary["run_count"]),
        "eligibleWindows": int(merge_summary["eligible_window_count"]),
        "macroF1": float(metrics["MacroF1"]),
        "developmentMacroF1": float(nested_robust["MacroF1Mean"]),
        "balancedAccuracy": float(metrics["BalancedAccuracy"]),
        "accuracy": float(metrics["Accuracy"]),
        "healthyFar": float(metrics["HealthyFalseAlarmRate"]),
        "developmentHealthyFar": float(nested_robust["HealthyFARMean"]),
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "冻结随机森林的开发评估、数据审计与一次性盲测技术结论。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": source_defs,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "kpis": [kpi_row],
                "per_class": per_class_rows,
                "far_by_op": far_rows,
                "phase_audit": phase_rows,
                "confusion_matrix": confusion_rows,
                "validation_comparison": validation_rows,
                "audit_checks": audit_rows,
            },
        },
        "sources": [
            {"id": item["id"], "label": item["label"], "path": item["path"]}
            for item in source_defs
        ],
    }

    report_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "question": "冻结储能故障诊断模型在一次性盲测中的泛化表现是否足以支持部署？",
        "decision_useful_answer": "否；开路故障召回率约 13.79%，健康误报集中于两个新工况，需进入新开发循环。",
        "required_structure_mapping": {
            "technical_summary": "技术摘要",
            "key_findings": ["开路故障漏检是决定性短板", "健康误报集中在两个高负载模式 1 工况", "一次性盲测证实开发估计仍偏乐观"],
            "scope_data_metrics": "评估范围与指标口径",
            "methodology": "冻结模型与评估方法",
            "limitations": "限制与不确定性",
            "recommended_next_steps": "建议的下一步",
            "further_questions": "后续需要回答的问题",
        },
        "chart_map": [
            {
                "section": "开路故障漏检是决定性短板",
                "question": "五类真实窗口的检出率分别是多少？",
                "family": "Comparison & Ranking",
                "type": "horizontalBar",
                "fields": ["classNameZh", "recall", "precision", "f1", "support"],
                "takeaway": "S1/S2 开路召回率均为 13.79%，显著低于其他类别。",
                "palette_policy": "single-root preferred",
                "delivery": "report.html native artifact chart",
            },
            {
                "section": "健康误报集中在两个高负载模式 1 工况",
                "question": "健康误报是否集中在少数工况？",
                "family": "Comparison & Ranking",
                "type": "horizontalBar",
                "fields": ["operatingPointId", "healthyFalseAlarmRate", "falseAlarmCount", "modeCommand", "socInit", "pload"],
                "takeaway": "全部 746 个健康误报集中于两个 600 W、ModeCommand=1 工况。",
                "palette_policy": "single-root preferred",
                "delivery": "report.html native artifact chart",
            },
        ],
        "qa_notes": [
            "所有百分比在 artifact 中以 0-1 比例存储。",
            "图表均为单系列，不使用冗余图例；水平条形图为长标签预留空间。",
            "开发评估与盲测只做描述性比较，不声明配对显著性或因果。",
            "窗口相关性与盲测工况覆盖限制已在正文邻近结论处披露。",
        ],
        "freeze_state_at_model_creation": freeze_manifest["state"],
        "blind_evaluation_count_for_this_frozen_model": 1,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "report_notes.json").write_text(
        json.dumps(report_notes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(OUTPUT_DIR / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
