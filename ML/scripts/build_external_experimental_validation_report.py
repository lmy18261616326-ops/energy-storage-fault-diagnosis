#!/usr/bin/env python
"""Build the canonical technical HTML-report artifact for external validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = PROJECT_ROOT / "ML" / "results" / "external_experimental_validation_jh69mxmx99_v1"
OUTPUT = PROJECT_ROOT / "output" / "reports" / "external_experimental_validation_jh69mxmx99_v1"
VALIDATION_SCRIPT = PROJECT_ROOT / "ML" / "scripts" / "validate_external_experimental_dataset.py"
MODEL = PROJECT_ROOT / "ML" / "models" / "reference_hdf5_optimized_v19" / "optimized_controller_fault_model_provisional.joblib"
ZIP_PATH = PROJECT_ROOT / "external_data" / "mendeley_jh69mxmx99_v1" / "jh69mxmx99_v1.zip"


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    generated_at: str,
    *,
    filters: list[str] | None = None,
    definitions: list[str] | None = None,
    tables_used: list[str] | None = None,
    url: str | None = None,
    sql: str | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "engine": "DuckDB" if sql else "Python/pandas",
        "language": "SQL" if sql else "Python",
        "description": description,
        "executed_at": generated_at,
        "id": source_id,
    }
    if sql:
        query["sql"] = sql
    if filters:
        query["filters"] = filters
    if definitions:
        query["metric_definitions"] = definitions
    if tables_used:
        query["tables_used"] = tables_used
    if url:
        query["url"] = url
    return {"id": source_id, "label": label, "path": rel(path), "query": query}


def main() -> int:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    summary = json.loads((RESULT / "validation_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((RESULT / "data_quality.json").read_text(encoding="utf-8"))
    cv = json.loads((RESULT / "c_state_cv_summary.json").read_text(encoding="utf-8"))
    compatibility = json.loads((RESULT / "model_compatibility.json").read_text(encoding="utf-8"))
    domain = json.loads((RESULT / "domain_shift_summary.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(RESULT / "source_inventory.csv")

    state_rows = [
        {"state": f"C{state}", "stateIndex": state, "recall": float(value)}
        for state, value in cv["per_state_recall"].items()
    ]
    fold_rows = [
        {
            "fold": int(row["fold"]),
            "groups": int(row["groups"]),
            "rows": int(row["rows"]),
            "accuracy": float(row["accuracy"]),
            "balancedAccuracy": float(row["balanced_accuracy"]),
            "macroF1": float(row["macro_f1"]),
        }
        for row in cv["folds"]
    ]
    key_name = {
        "vbus_meas_V__mean__mean": "V 均值",
        "vbus_meas_V__std__mean": "V 波动标准差",
        "il_meas_A__mean__mean": "I 均值",
        "il_meas_A__std__mean": "I 波动标准差",
    }
    domain_rows = [
        {
            "feature": key_name.get(row["feature"], row["feature"]),
            "featureId": row["feature"],
            "trainingMin": float(row["training_min"]),
            "trainingMedian": float(row["training_median"]),
            "trainingMax": float(row["training_max"]),
            "externalMin": float(row["external_min"]),
            "externalMedian": float(row["external_median"]),
            "externalMax": float(row["external_max"]),
            "outsideFraction": float(row["external_outside_training_range_fraction"]),
        }
        for row in domain["key_features"]
    ]
    distribution_total = sum(compatibility["prediction_distribution"].values())
    class_rows = [
        {
            "class": name,
            "count": int(count),
            "share": float(count / distribution_total),
            "interpretation": "诊断输出；无外部同类真值",
        }
        for name, count in compatibility["prediction_distribution"].items()
    ]
    coverage_rows = [
        {
            "representation": "原始控制器通道",
            "available": compatibility["assumed_available_raw_channels"],
            "expected": compatibility["expected_raw_channels"],
            "coverage": compatibility["raw_channel_coverage"],
            "status": "假设性语义映射",
        },
        {
            "representation": "冻结模型特征",
            "available": compatibility["available_features"],
            "expected": compatibility["expected_features"],
            "coverage": compatibility["feature_coverage"],
            "status": "其余 850 项训练中位数填补",
        },
    ]
    quality_rows = [
        {"check": "ZIP 字节数", "result": "通过", "evidence": f"{quality['zip']['bytes']} bytes"},
        {"check": "ZIP SHA-256", "result": "通过", "evidence": quality["zip"]["sha256"]},
        {"check": "工作簿/工作表结构", "result": "通过", "evidence": "21 个工作簿、189 个工作表"},
        {"check": "V/I 数值单元", "result": "通过", "evidence": f"{quality['shape']['voltage_current_numeric_cells']} 个"},
        {"check": "缺失、非数或无穷", "result": "通过", "evidence": "0"},
        {"check": "公式单元格", "result": "通过", "evidence": "0"},
        {"check": "重复 V/I 波形对", "result": "通过", "evidence": "0"},
        {"check": "零方差波形", "result": "通过", "evidence": "0"},
        {"check": "R1/R2/C 数值映射", "result": "缺失", "evidence": "工作簿未提供"},
        {"check": "采样间隔/时间单位", "result": "缺失", "evidence": "工作簿仅提供 Index"},
        {"check": "六类故障真值", "result": "不重叠", "evidence": "外部源只给文件级 C 状态"},
    ]
    compatibility_rows = [
        {"item": "冻结模型资格", "value": compatibility["model_qualification_status"], "decision": "保持临时"},
        {"item": "V/I 语义映射", "value": "V→vbus_meas_V；I→il_meas_A", "decision": "仅假设"},
        {"item": "冻结特征覆盖", "value": f"{compatibility['available_features']}/{compatibility['expected_features']}", "decision": "不足"},
        {"item": "外部标签重叠", "value": "0/6", "decision": "无法计算准确率"},
        {"item": "诊断输出", "value": f"{distribution_total}/{distribution_total} → high_resistance", "decision": "单类塌缩"},
        {"item": "最大概率中位数", "value": f"{compatibility['median_max_probability']:.3f}", "decision": "低置信"},
        {"item": "归一化熵中位数", "value": f"{compatibility['median_normalized_entropy']:.3f}", "decision": "高不确定性"},
        {"item": "最终模型晋级", "value": "否", "decision": "真实六类验证未完成"},
    ]
    inventory_rows = [
        {
            "file": str(row["file"]),
            "cState": int(row["c_state"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
            "sheets": int(row["sheets"]),
        }
        for row in inventory.sort_values("c_state").to_dict(orient="records")
    ]

    source_defs = [
        source(
            "src_external_source",
            "Mendeley 实验数据包 jh69mxmx99 v1",
            ZIP_PATH,
            "官方数据包；21 个双向 Buck/Boost 变换器退化状态实验波形工作簿。",
            generated_at,
            filters=["版本 1", "CC BY 4.0", "仅使用 Waveforms_C1.xlsx 至 Waveforms_C21.xlsx"],
            definitions=["trace=一个全局源样本 ID 下的一对 2,000 点 V/I 波形"],
            tables_used=[rel(ZIP_PATH)],
            url="https://data.mendeley.com/datasets/jh69mxmx99/1",
            sql=(
                "SELECT file, c_state, bytes, sha256, sheets FROM read_csv_auto("
                f"'{rel(RESULT / 'source_inventory.csv')}') ORDER BY c_state"
            ),
        ),
        source(
            "src_quality",
            "外部实验数据质量审计",
            RESULT / "data_quality.json",
            "逐工作簿验证表名、表头、维度、Index、有限数、重复和零方差，并核对 ZIP 大小与 SHA-256。",
            generated_at,
            definitions=[
                "完整性门=ZIP 大小匹配且 SHA-256 匹配且所有工作簿结构/数值检查通过",
                "重复波形=V 与 I 的完整二进制序列 SHA-256 同时重复",
            ],
            tables_used=[rel(RESULT / "data_quality.json"), rel(RESULT / "source_inventory.csv")],
            sql=f"SELECT * FROM read_json_auto('{rel(RESULT / 'data_quality.json')}')",
        ),
        source(
            "src_cv",
            "真实 V/I 文件级 C 状态分组交叉验证",
            RESULT / "c_state_cv_summary.json",
            "100 个 V/I 滑窗统计特征的五折 Extra Trees OOF 评估；按文件内网格位置隔离。",
            generated_at,
            filters=["21 个文件级 C 状态", "每折训练/测试的网格位置不重叠", "不解释为六类故障准确率"],
            definitions=[
                "Macro-F1=21 个文件级 C 状态 F1 的算术平均",
                "机会水平=1/21",
                "负对照=每个网格位置内独立打乱 C 标签后使用同一 GroupKFold 口径",
                "95% 区间=以 441 个网格位置为重采样单位的 1,000 次 bootstrap 百分位区间",
            ],
            tables_used=[rel(RESULT / "c_state_oof_predictions.csv"), rel(RESULT / "external_common_channel_features.csv")],
            sql=f"SELECT * FROM read_json_auto('{rel(RESULT / 'c_state_cv_summary.json')}')",
        ),
        source(
            "src_compatibility",
            "冻结六类模型兼容性与诊断压力测试",
            RESULT / "model_compatibility.json",
            "在未提供特征由冻结 SimpleImputer 中位数填补的条件下，执行假设性 V/I 映射压力测试。",
            generated_at,
            filters=["V 假设映射到 vbus_meas_V", "I 假设映射到 il_meas_A", "不计算外部准确率"],
            definitions=[
                "特征覆盖=外部可构造且列名匹配的特征数/冻结模型期望特征数",
                "归一化熵=-Σp·ln(p)/ln(6)",
                "单类塌缩=所有 9,261 条输入具有同一 argmax 类别",
            ],
            tables_used=[rel(RESULT / "diagnostic_frozen_model_predictions.csv"), rel(MODEL)],
            sql=f"SELECT * FROM read_json_auto('{rel(RESULT / 'model_compatibility.json')}')",
        ),
        source(
            "src_domain",
            "实验域与合成训练域范围对比",
            RESULT / "domain_shift_summary.json",
            "对 100 个假设性对齐特征逐项比较外部值与合成训练集最小/最大范围。",
            generated_at,
            filters=["仅比较 vbus_meas_V 与 il_meas_A 的 100 个同构特征", "不把范围越界解释为故障"],
            definitions=["域外单元占比=外部特征值低于训练最小值或高于训练最大值的单元数/比较单元总数"],
            tables_used=[rel(RESULT / "domain_shift_features.csv")],
            sql=f"SELECT * FROM read_json_auto('{rel(RESULT / 'domain_shift_summary.json')}')",
        ),
        source(
            "src_script",
            "可重跑外部验证脚本",
            VALIDATION_SCRIPT,
            "XLSX 解析、特征构造、数据质量门、分组交叉验证、负对照、冻结模型压力测试和图表生成代码。",
            generated_at,
            tables_used=[rel(VALIDATION_SCRIPT)],
        ),
    ]

    kpis = {
        "integrityGate": 1,
        "workbooks": quality["shape"]["workbooks"],
        "traces": quality["shape"]["traces"],
        "numericCells": quality["shape"]["voltage_current_numeric_cells"],
        "stateMacroF1": cv["macro_f1"],
        "chanceMacroF1": cv["chance_macro_f1"],
        "negativeMacroF1": cv["within_group_label_permutation_macro_f1"],
        "featureCoverage": compatibility["feature_coverage"],
        "imputedFeatures": compatibility["imputed_features"],
        "outsideFraction": domain["external_cells_outside_synthetic_training_minmax_fraction"],
        "collapsedShare": compatibility["prediction_distribution"]["high_resistance"] / distribution_total,
        "medianConfidence": compatibility["median_max_probability"],
    }
    cards = [
        {
            "id": "card_integrity",
            "description": "1 表示 ZIP 哈希、工作簿结构和数值完整性检查全部通过。",
            "dataset": "kpis",
            "sourceId": "src_quality",
            "metrics": [
                {"label": "数据完整性门（1=通过）", "field": "integrityGate", "format": "number"},
                {"label": "真实 V/I 波形对", "field": "traces", "format": "number"},
            ],
        },
        {
            "id": "card_state_f1",
            "description": "文件内网格位置隔离的五折 OOF；任务是 21 个文件级 C 状态识别。",
            "dataset": "kpis",
            "sourceId": "src_cv",
            "metrics": [
                {"label": "真实 V/I 状态 Macro-F1", "field": "stateMacroF1", "format": "percent"},
                {"label": "机会水平", "field": "chanceMacroF1", "format": "percent"},
                {"label": "标签负对照", "field": "negativeMacroF1", "format": "percent"},
            ],
        },
        {
            "id": "card_coverage",
            "description": "仅有两个假设性对齐通道；其余冻结特征由训练期中位数填补。",
            "dataset": "kpis",
            "sourceId": "src_compatibility",
            "metrics": [
                {"label": "冻结模型特征覆盖", "field": "featureCoverage", "format": "percent"},
                {"label": "中位数填补特征", "field": "imputedFeatures", "format": "number"},
            ],
        },
        {
            "id": "card_domain",
            "description": "100 个假设性对齐特征中，外部数值越过合成训练最小/最大范围的单元占比。",
            "dataset": "kpis",
            "sourceId": "src_domain",
            "metrics": [{"label": "域外特征单元", "field": "outsideFraction", "format": "percent"}],
        },
        {
            "id": "card_collapse",
            "description": "诊断性压力测试输出，不是有外部六类真值支持的正确率。",
            "dataset": "kpis",
            "sourceId": "src_compatibility",
            "metrics": [
                {"label": "判为 high_resistance", "field": "collapsedShare", "format": "percent"},
                {"label": "最大概率中位数", "field": "medianConfidence", "format": "percent"},
            ],
        },
    ]

    charts = [
        {
            "id": "chart_state_recall",
            "title": "文件级 C 状态逐类 OOF 召回率",
            "subtitle": "按文件内网格位置隔离；C1/C21 最易区分，中间状态仍显著高于 1/21 机会水平。",
            "intent": "comparison",
            "question": "真实 V/I 对各文件级 C 状态的区分能力是否一致？",
            "rationale": "21 个离散状态适合用水平条形图比较并保留完整标签。",
            "comparisonContext": {
                "denominator": "每个真实 C 状态的 441 条波形",
                "grain": "文件级 C 状态",
                "normalization": "Recall=TP/(TP+FN)",
                "semanticFamily": "真实实验状态可区分性",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "state_recall",
            "sourceId": "src_cv",
            "encodings": {
                "x": {"field": "state", "type": "nominal", "label": "文件级 C 状态"},
                "y": {"field": "recall", "type": "quantitative", "format": "percent", "label": "OOF 召回率"},
                "tooltip": [{"field": "stateIndex", "type": "quantitative", "format": "number", "label": "状态序号"}],
            },
            "valueFormat": "percent",
            "layout": "full",
            "labels": {"values": "all"},
            "palette": {"kind": "sequential"},
            "settings": {"groupMode": "single", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "interactiveLegend": False, "showControls": False, "viewMode": "both"},
        },
        {
            "id": "chart_coverage",
            "title": "冻结模型输入覆盖率",
            "subtitle": "两种口径均不足 17%，且可用 V/I 的具体测量位置仍未经来源确认。",
            "intent": "comparison",
            "question": "实验数据覆盖了多少冻结模型输入？",
            "rationale": "两个明确分母适合直接比较覆盖比例。",
            "comparisonContext": {
                "denominator": "冻结模型期望输入",
                "grain": "输入口径",
                "normalization": "可用数/期望数",
                "semanticFamily": "模型兼容性",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "coverage",
            "sourceId": "src_compatibility",
            "encodings": {
                "x": {"field": "representation", "type": "nominal", "label": "口径"},
                "y": {"field": "coverage", "type": "quantitative", "format": "percent", "label": "覆盖率"},
                "tooltip": [
                    {"field": "available", "type": "quantitative", "format": "number", "label": "可用"},
                    {"field": "expected", "type": "quantitative", "format": "number", "label": "期望"},
                    {"field": "status", "type": "nominal", "label": "说明"},
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
            "id": "chart_domain",
            "title": "关键对齐特征的训练范围越界率",
            "subtitle": "V/I 均值全部越界；V 波动标准差几乎全部越界。",
            "intent": "comparison",
            "question": "关键实验特征有多少落在合成训练范围之外？",
            "rationale": "特征越界率是同尺度比例，适合水平条形比较。",
            "comparisonContext": {
                "denominator": "每个特征的 9,261 个外部值",
                "grain": "假设性对齐特征",
                "normalization": "训练范围外数/外部总数",
                "semanticFamily": "域偏移",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "domain_key",
            "sourceId": "src_domain",
            "encodings": {
                "x": {"field": "feature", "type": "nominal", "label": "特征"},
                "y": {"field": "outsideFraction", "type": "quantitative", "format": "percent", "label": "越界率"},
                "tooltip": [
                    {"field": "trainingMedian", "type": "quantitative", "format": "number", "label": "合成中位数"},
                    {"field": "externalMedian", "type": "quantitative", "format": "number", "label": "实验中位数"},
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
            "id": "chart_model_distribution",
            "title": "诊断性冻结模型输出分布",
            "subtitle": "9,261 条均以 high_resistance 为 argmax；没有外部六类真值，不表示 100% 正确。",
            "intent": "comparison",
            "question": "部分输入和强域偏移下，冻结模型输出是否发生塌缩？",
            "rationale": "六类输出份额用水平条形图可直接暴露单类塌缩。",
            "comparisonContext": {
                "denominator": "9,261 条实验波形",
                "grain": "冻结模型输出类",
                "normalization": "argmax 类计数/总波形数",
                "semanticFamily": "诊断压力测试",
                "unit": "比例",
            },
            "type": "horizontalBar",
            "dataset": "model_distribution",
            "sourceId": "src_compatibility",
            "encodings": {
                "x": {"field": "class", "type": "nominal", "label": "冻结类"},
                "y": {"field": "share", "type": "quantitative", "format": "percent", "label": "输出份额"},
                "tooltip": [
                    {"field": "count", "type": "quantitative", "format": "number", "label": "计数"},
                    {"field": "interpretation", "type": "nominal", "label": "口径"},
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
            "id": "table_quality",
            "title": "数据质量与语义完整性门",
            "subtitle": "结构完整性通过；物理参数映射、时间尺度和冻结故障真值缺失。",
            "dataset": "quality_checks",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_quality",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "检查项", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "evidence", "label": "证据", "type": "text"},
            ],
        },
        {
            "id": "table_folds",
            "title": "五折网格位置隔离结果",
            "subtitle": "每折包含 88–89 个未见网格位置；各折 Macro-F1 接近。",
            "dataset": "cv_folds",
            "defaultSort": {"field": "fold", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_cv",
            "layout": "full",
            "columns": [
                {"field": "fold", "label": "折", "format": "number"},
                {"field": "groups", "label": "测试网格位置", "format": "number"},
                {"field": "rows", "label": "测试波形", "format": "number"},
                {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                {"field": "balancedAccuracy", "label": "Balanced accuracy", "format": "percent"},
                {"field": "macroF1", "label": "Macro-F1", "format": "percent"},
            ],
        },
        {
            "id": "table_domain",
            "title": "关键特征训练域与实验域范围",
            "subtitle": "V/I 具体测量位置未经数据源确认，因此数值只作域偏移证据。",
            "dataset": "domain_key",
            "defaultSort": {"field": "outsideFraction", "direction": "desc"},
            "density": "spacious",
            "sourceId": "src_domain",
            "layout": "full",
            "columns": [
                {"field": "feature", "label": "特征", "type": "text"},
                {"field": "trainingMin", "label": "合成最小", "format": "number"},
                {"field": "trainingMedian", "label": "合成中位数", "format": "number"},
                {"field": "trainingMax", "label": "合成最大", "format": "number"},
                {"field": "externalMin", "label": "实验最小", "format": "number"},
                {"field": "externalMedian", "label": "实验中位数", "format": "number"},
                {"field": "externalMax", "label": "实验最大", "format": "number"},
                {"field": "outsideFraction", "label": "越界率", "format": "percent"},
            ],
        },
        {
            "id": "table_compatibility",
            "title": "冻结模型兼容性判定",
            "subtitle": "兼容性门失败，因此没有生成六类准确率、召回率或误报率。",
            "dataset": "compatibility",
            "defaultSort": {"field": "item", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_compatibility",
            "layout": "full",
            "columns": [
                {"field": "item", "label": "项目", "type": "text"},
                {"field": "value", "label": "观测", "type": "text"},
                {"field": "decision", "label": "判定", "type": "text"},
            ],
        },
        {
            "id": "table_inventory",
            "title": "21 个外部实验工作簿清单",
            "subtitle": "所有文件均单独计算 SHA-256；ZIP 总哈希另见质量表。",
            "dataset": "inventory",
            "defaultSort": {"field": "cState", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_quality",
            "layout": "full",
            "columns": [
                {"field": "cState", "label": "C 状态序号", "format": "number"},
                {"field": "file", "label": "文件", "type": "text"},
                {"field": "bytes", "label": "字节数", "format": "number"},
                {"field": "sheets", "label": "工作表", "format": "number"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
    ]

    title = "双向 DC-DC 冻结模型：真实实验数据外部验证"
    blocks = [
        {"id": "block_title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "block_summary_quality",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                "Mendeley 数据包的 **21 个工作簿、189 个工作表、9,261 对实验 V/I 波形和 37,044,000 个数值单元**"
                "通过结构与数值完整性门；ZIP 字节数和 SHA-256 均与已锁定下载一致。"
            ),
            "sourceId": "src_quality",
            "layout": "full",
        },
        {
            "id": "block_summary_cv",
            "type": "markdown",
            "body": (
                "只用真实 V/I 的 21 状态外部基准取得 **Macro-F1 50.44%**，网格位置 bootstrap 95% 区间为 "
                "**49.09%–51.75%**；机会水平为 **4.76%**，标签负对照为 **4.23%**。"
                "这证明数据含有可复现的文件级状态信号，但不是冻结六类故障准确率。"
            ),
            "sourceId": "src_cv",
            "layout": "full",
        },
        {
            "id": "block_summary_compat",
            "type": "markdown",
            "body": (
                "冻结六类模型的直接外部准确率被兼容性门阻断：仅能假设性构造 **100/950（10.53%）** 特征，"
                "850 项必须中位数填补，且外部标签与六类真值没有重叠。诊断压力测试把 **9,261/9,261** 条波形"
                "全部判为 high_resistance，最大概率中位数仅 **35.92%**。当前模型必须保持 "
                "`provisional_synthetic_only`，不得晋级。"
            ),
            "sourceId": "src_compatibility",
            "layout": "full",
        },
        {"id": "block_kpis", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "block_finding_state",
            "type": "markdown",
            "body": (
                "## 真实 V/I 确实携带稳定的文件级状态信息\n\n"
                "五个外层折的 Macro-F1 为 49.09%–51.98%，波动较小。C1 与 C21 的召回率分别为 80.73% 和 74.83%，"
                "中间状态更容易相互混淆。由于测试折按 1–441 的文件内网格位置隔离，结果不是同一位置跨折泄漏造成。"
            ),
            "sourceId": "src_cv",
            "layout": "full",
        },
        {"id": "block_state_chart", "type": "chart", "chartId": "chart_state_recall", "layout": "full"},
        {"id": "block_folds_table", "type": "table", "tableId": "table_folds", "layout": "full"},
        {
            "id": "block_finding_coverage",
            "type": "markdown",
            "body": (
                "## 输入与标签不兼容，六类准确率没有可定义的分母\n\n"
                "外部工作簿只有通用 `V`/`I`，而冻结管线依赖 12 个控制器原始通道以及 7 个派生通道。"
                "即使暂时假设 V 是母线测量电压、I 是电感测量电流，原始通道覆盖率也只有 16.67%，"
                "工程特征覆盖率只有 10.53%。外部的 C 文件状态也不能映射到 healthy、两类传感器偏置、"
                "S1/S2 开路和 high_resistance 六类，因此没有报告准确率、召回率或误报率。"
            ),
            "sourceId": "src_compatibility",
            "layout": "full",
        },
        {"id": "block_coverage_chart", "type": "chart", "chartId": "chart_coverage", "layout": "full"},
        {"id": "block_compat_table", "type": "table", "tableId": "table_compatibility", "layout": "full"},
        {
            "id": "block_finding_domain",
            "type": "markdown",
            "body": (
                "## 强域偏移使冻结模型的单类输出不可解释为检测成功\n\n"
                "在 926,100 个可对齐特征单元中，**82.61%** 超出合成训练最小–最大范围，100 个特征中有 85 个"
                "至少出现一次越界。V 均值和 I 均值的所有外部值都越界；V 波动标准差的越界率为 99.67%。"
                "这是冻结模型输入域不匹配的直接证据。"
            ),
            "sourceId": "src_domain",
            "layout": "full",
        },
        {"id": "block_domain_chart", "type": "chart", "chartId": "chart_domain", "layout": "full"},
        {"id": "block_domain_table", "type": "table", "tableId": "table_domain", "layout": "full"},
        {
            "id": "block_finding_collapse",
            "type": "markdown",
            "body": (
                "## 诊断压力测试暴露了单类塌缩\n\n"
                "9,261 条波形全部以 high_resistance 为最大概率类，但最大概率中位数只有 35.92%，归一化熵中位数为 89.88%。"
                "高熵与单类 argmax 同时出现，说明模型并未形成有把握的外部判别；该图只用于故障分析。"
            ),
            "sourceId": "src_compatibility",
            "layout": "full",
        },
        {"id": "block_collapse_chart", "type": "chart", "chartId": "chart_model_distribution", "layout": "full"},
        {
            "id": "block_scope",
            "type": "markdown",
            "body": (
                "## 范围、数据与指标定义\n\n"
                "数据源页面将这些文件描述为双向 Buck/Boost 变换器在 R1、R2、C 组合退化状态下的稳态实验电压/电流波形。"
                "本地文件包含 C1–C21、全局源样本 ID 1–9,261、通用 V/I 列和 Index；没有 R1/R2/C 的数值映射、"
                "采样间隔或六类故障标签。\n\n"
                "本报告的 Macro-F1 只衡量 21 个文件级 C 状态；“域外单元”只表示数值越过合成训练范围；"
                "“诊断输出份额”只表示冻结模型 argmax 分布。这三者都不等于真实六类故障性能。"
            ),
            "sourceId": "src_external_source",
            "layout": "full",
        },
        {"id": "block_quality_table", "type": "table", "tableId": "table_quality", "layout": "full"},
        {"id": "block_inventory", "type": "table", "tableId": "table_inventory", "layout": "full"},
        {
            "id": "block_method",
            "type": "markdown",
            "body": (
                "## 方法\n\n"
                "验证脚本逐个流式解析 21 个 XLSX，核对官方全局列号、工作表维度、1–2,000 的 Index、有限数、公式、"
                "重复和零方差。每条 V/I 波形按冻结管线的 1,000 点窗口、500 点步长提取 100 个统计特征。"
                "外部基准采用 5 折 GroupKFold，分组键是每文件内 1–441 的网格位置；Extra Trees 的折外预测用于"
                "Macro-F1，区间按网格位置 bootstrap。负对照在每个网格位置内打乱 21 个 C 标签。\n\n"
                "冻结压力测试把可对齐特征写入原 950 维顺序，其余值保持缺失并交给冻结 SimpleImputer。"
                "所有生成步骤、数据口径和图表均由同一脚本产生。"
            ),
            "sourceId": "src_script",
            "layout": "full",
        },
        {
            "id": "block_limitations",
            "type": "markdown",
            "body": (
                "## 限制、不确定性与稳健性\n\n"
                "数据源描述称存在 R1/R2/C 对应参数，但下载的 21 个工作簿没有给出其数值映射；441 个文件内位置只能"
                "作为稳定分组键，不能被声明为已知 R1/R2 组合。Index 也没有时间单位，因此不能计算物理频率、上升时间或延迟。"
                "通用 V/I 的具体电路测点未标明，对齐到 `vbus_meas_V`/`il_meas_A` 是显式假设。\n\n"
                "状态基准的 95% 区间按 441 个网格位置重采样，五折结果接近且标签负对照回到机会水平附近；这支持"
                "“文件级状态信号可复现”，但不支持具体物理参数、严重度单调性或当前六类诊断能力。"
            ),
            "sourceId": "src_quality",
            "layout": "full",
        },
        {
            "id": "block_next_steps",
            "type": "markdown",
            "body": (
                "## 建议的下一步\n\n"
                "1. **保持冻结模型不晋级。** 把本次结果登记为外部兼容性失败，而不是低分或高分的准确率试验。\n"
                "2. **向数据作者索取参数字典。** 需要 C1–C21 的实际电容值、9,261 个源样本与 R1/R2 的映射、采样间隔和 V/I 测点。\n"
                "3. **建立真实六类采集协议。** 同步记录 12 个控制器原始通道，并以运行级真值覆盖 healthy、vbus_bias、il_bias、S1_open、S2_open、high_resistance。\n"
                "4. **先做域适配再冻结新模型。** 电压约 100 V、当前约 26 A 的实验域不能直接套用原 400 V、约 ±15 A 训练范围；"
                "应预注册工况归一化、分组策略和一次性盲测门槛。\n"
                "5. **保留本数据作为辅助外部基准。** 在参数字典补齐后，可用于 R1/R2/C 回归、状态排序或只含 V/I 的轻量诊断器，而不是冒充六类模型验证集。"
            ),
            "layout": "full",
        },
        {
            "id": "block_questions",
            "type": "markdown",
            "body": (
                "## 后续需要回答的问题\n\n"
                "- C1–C21 是否按实际电容值单调排序，还是仅为任意文件编号？\n"
                "- 每个文件的 441 个位置是否严格对应相同的 21×21 R1/R2 网格？\n"
                "- V 与 I 分别测量在哪个端口/器件，极性与探头带宽是什么？\n"
                "- 采样频率、开关频率、负载、控制模式和稳态截取规则是什么？\n"
                "- 能否在相同实验台架上注入现有六类故障并同步导出控制器通道？"
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
            "description": "真实实验 V/I 数据质量、文件级状态可区分性、冻结六类模型兼容性与域偏移技术审计。",
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
                "kpis": [kpis],
                "state_recall": state_rows,
                "cv_folds": fold_rows,
                "domain_key": domain_rows,
                "model_distribution": class_rows,
                "coverage": coverage_rows,
                "quality_checks": quality_rows,
                "compatibility": compatibility_rows,
                "inventory": inventory_rows,
            },
        },
        "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in source_defs],
    }
    report_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "question": "可公开获得的真实双向 Buck/Boost 实验数据能否验证冻结六类合成模型？",
        "decision_useful_answer": "数据可验证实验状态信号，但因标签、通道和工作域不兼容，不能计算冻结六类准确率；模型不得晋级。",
        "required_structure_mapping": {
            "technical_summary": "技术摘要",
            "key_findings": ["真实 V/I 确实携带稳定的文件级状态信息", "输入与标签不兼容", "强域偏移", "诊断压力测试单类塌缩"],
            "scope_data_metrics": "范围、数据与指标定义",
            "methodology": "方法",
            "limitations": "限制、不确定性与稳健性",
            "recommended_next_steps": "建议的下一步",
            "further_questions": "后续需要回答的问题",
        },
        "chart_map": [
            {"chart": "chart_state_recall", "question": "各 C 状态是否同等可区分？", "takeaway": "总体明显高于机会水平，端点状态更易区分。"},
            {"chart": "chart_coverage", "question": "外部数据覆盖多少冻结输入？", "takeaway": "原始通道与工程特征覆盖均低于 17%。"},
            {"chart": "chart_domain", "question": "关键输入是否处于训练范围？", "takeaway": "V/I 均值全部越界，存在强域偏移。"},
            {"chart": "chart_model_distribution", "question": "压力测试输出是否塌缩？", "takeaway": "全部 argmax 为 high_resistance，但置信度低、熵高。"},
        ],
        "qa_notes": [
            "所有 percent 字段以 0–1 比例存储。",
            "六类准确率未定义且未计算。",
            "文件级 C 状态不解释为物理严重度顺序。",
            "报告只嵌入聚合快照；9,261 行预测保留在结果目录供审计。",
        ],
        "validation_decision": summary["decision"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "report_notes.json").write_text(json.dumps(report_notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
