"""Patch the complete group-report artifact with the v06 high-R resolution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "ML" / "reports" / "model_selection_group_report_2026-08-04"
ARTIFACT_PATH = REPORT_ROOT / "artifact.json"
RESULT_ROOT = ROOT / "ML" / "results" / "switch_observability_specialist"
GENERATED_AT = "2026-08-04T08:55:00+08:00"


def item_by_id(items: list[dict], item_id: str) -> dict:
    return next(item for item in items if item["id"] == item_id)


def upsert(items: list[dict], value: dict) -> None:
    for index, item in enumerate(items):
        if item.get("id") == value["id"]:
            items[index] = value
            return
    items.append(value)


def main() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    manifest["generatedAt"] = GENERATED_AT
    snapshot["generatedAt"] = GENERATED_AT
    manifest["description"] = (
        "面向组会的模型对比、数据分布、验证边界、问题复盘与高阻可观测性修复报告。"
    )

    high_r_card = item_by_id(manifest["cards"], "high_r_card")
    high_r_card.update(
        {
            "description": "v06 同步开关压降/电流测量下，冻结物理阈值在未见 0.05 Ω 独立集上的结果。",
            "dataset": "headline_metrics",
            "sourceId": "hr_direct_comparison",
            "metrics": [
                {
                    "label": "高阻盲测 Macro-F1",
                    "field": "high_r_macro_f1",
                    "format": "percent",
                },
                {
                    "label": "健康 FAR",
                    "field": "high_r_healthy_far",
                    "format": "percent",
                },
            ],
        }
    )

    hr_table = {
        "id": "hr_direct_comparison_table",
        "title": "高阻专用输入上的模型对比",
        "subtitle": "32 Run 分组 OOF 开发集与 16 Run 未见 0.05 Ω 独立集；表中展示独立验证",
        "dataset": "hr_direct_comparison",
        "sourceId": "hr_direct_comparison",
        "density": "comfortable",
        "defaultSort": {"field": "train_seconds", "direction": "asc"},
        "columns": [
            {"field": "model", "label": "模型", "type": "text"},
            {"field": "macro_f1", "label": "窗口 Macro-F1", "format": "percent", "type": "percent"},
            {"field": "run_macro_f1", "label": "Run Macro-F1", "format": "percent", "type": "percent"},
            {"field": "fault_recall", "label": "故障召回", "format": "percent", "type": "percent"},
            {"field": "healthy_far", "label": "健康 FAR", "format": "percent", "type": "percent"},
            {"field": "train_seconds", "label": "四折训练秒数", "format": "number", "type": "number"},
            {"field": "role", "label": "结论", "type": "text"},
        ],
    }
    upsert(manifest["tables"], hr_table)

    for source in [
        {
            "id": "hr_direct_summary",
            "label": "v06 开关直接电阻专用模型摘要",
            "path": "ML/results/switch_observability_specialist/summary.json",
        },
        {
            "id": "hr_direct_comparison",
            "label": "v06 高阻专用模型对比与独立验证",
            "path": "ML/results/switch_observability_specialist/model_comparison.csv",
        },
    ]:
        upsert(manifest["sources"], source)
    item_by_id(manifest["sources"], "problem_log")["label"] = "完整问题日志 P01–P29"

    blocks = manifest["blocks"]
    item_by_id(blocks, "technical_summary")["body"] = (
        "## 技术结论\n\n"
        "- **已有合格主基线。** 在主动可观测范围内，逻辑回归主判与 ExtraTrees 校验器的六折分组 OOF Macro-F1、最低 S1/S2 召回均为 100%，健康 FAR 为 0。\n"
        "- **高阻可观测性问题已在 v06 解决。** 旧通用控制信号的 28 个候选仍是失败证据；新增 1 µs 同步开关电流/压降后，冻结 0.0105 Ω 阈值在 16 Run、1,904 窗口的未见 0.05 Ω 独立集上 Macro-F1/召回均为 100%，健康 FAR 为 0。\n"
        "- **分布决定模型。** 主任务的 660 维具名统计表适合正则化线性/树模型；高阻直接测量则是一维物理阈值问题。两种输入表示下都没有使用复杂融合的必要。\n"
        "- **结论仍有硬件边界。** 高阻通过只适用于 `main_model_fd_v06_switchobservability` 及同步器件测量，尚不代表 v05、HIL 或真实硬件已经通过。"
    )

    new_blocks = [
        {
            "id": "hr_direct_resolution",
            "type": "markdown",
            "sourceId": "hr_direct_summary",
            "body": (
                "## 直接测量把高阻从统计弱信号变成物理可分问题\n\n"
                "v06 复用 IGBT/Diode 已有测量端口，分别记录 S1/S2 器件电流与压降。开发集含 32 Run、3,808 个合格窗口：健康导通电阻最大为 0.001 Ω，0.02/0.10 Ω 故障最小为 0.02 Ω；据此冻结阈值 0.0105 Ω。随后使用新随机种子和未见 `Ron=0.05 Ω` 的 16 Run、1,904 窗口做独立验证，健康上界仍为 0.001 Ω、故障下界为 0.05 Ω。该修复解决的是观测设计，不是靠扩大模型容量拟合旧信号。"
            ),
        },
        {
            "id": "hr_direct_model_finding",
            "type": "markdown",
            "sourceId": "hr_direct_comparison",
            "body": (
                "### 六类学习模型都通过，但物理阈值最合适\n\n"
                "冻结物理阈值、逻辑回归、ExtraTrees、随机森林、KNN、MLP 和 1D-CNN 在分组 OOF及独立验证上均为 1.00 Macro-F1、1.00 故障召回和 0 健康 FAR。结果说明新特征本身已经充分可分；物理阈值训练成本为 0、可解释且直接对应器件参数，因此选为专用主判，其余模型仅保留为对照。"
            ),
        },
        {"id": "hr_direct_table_block", "type": "table", "tableId": "hr_direct_comparison_table"},
        {
            "id": "hr_history_context",
            "type": "markdown",
            "body": (
                "### 旧高阻失败结果仍保留为对照，而不是被删除\n\n"
                "下面的 25% 健康误报和 28 个候选全失败，针对的是没有开关同步压降/电流的 v05 通用控制信号。它们证明继续调旧模型无效，也直接促成 v06 观测设计；不能与新增测量后的通过结果混为同一输入条件。"
            ),
        },
    ]
    blocks[:] = [block for block in blocks if block["id"] not in {b["id"] for b in new_blocks}]
    insert_at = next(i for i, block in enumerate(blocks) if block["id"] == "hr_independent_finding")
    blocks[insert_at:insert_at] = new_blocks
    item_by_id(blocks, "hr_independent_finding")["body"] = (
        "## 旧通用信号的独立验证否定了 Pilot 乐观结果\n\n"
        "在 v05 通用控制信号上，冻结的健康指纹 ExtraTrees 与 0.60 阈值，在 16 个新随机种子、未见 `Ron=0.05 Ω` 事件上只有 62.5% 召回，健康 FAR 为 25%、最差工况 FAR 为 100%。该历史候选未通过，阈值没有根据独立集回调；其失败结论仍对无新增传感器的方案有效。"
    )
    item_by_id(blocks, "hr_transfer_finding")["body"] = (
        "### 旧信号的 28 个替代候选仍无法同时满足误报与召回\n\n"
        "失败后对原始事件、负载阶跃、物理子集、健康残差、相对残差与四类模型做一次性诊断，共 28 个候选。没有候选进入 FAR≤5%、召回≥60% 的资格区域；最佳零误报候选召回仅 50%。这条负结果支持增加观测量，而非继续融合旧模型。"
    )
    item_by_id(blocks, "scope_definitions")["body"] = (
        "## 两套模型的范围与指标定义\n\n"
        "- **主五类事件模型：** 416 Run、28 工况，按 `OperatingPointID` 做 6 折 OOF；支持健康、两类传感器偏置及 S1/Mode1、S2/Mode2 的主动位置故障。\n"
        "- **高阻专用模型：** 只在活动开关导通样本中计算 `median(|V/I|)`；开发集 32 Run，独立集 16 Run，独立阻值为未见 0.05 Ω。\n"
        "- **硬件要求：** v06 的 S1/S2 器件电流和压降须以 1 µs 同步记录，再聚合到 50 µs；无此观测时仍适用旧失败结论。\n"
        "- **核心指标：** Macro-F1 为类别 F1 等权平均；健康 FAR 为健康事件被报为故障的比例；高阻故障召回分别在 Mode1/S1 和 Mode2/S2 上计算。\n"
        "- **仍排除状态：** 非导通开关的位置/高阻故障需等待模式切换或主动诊断激励。"
    )
    item_by_id(blocks, "methodology")["body"] = (
        "## 实验设计、采样修复与模型规格\n\n"
        "1. 主任务采用 Run 级事件聚合和工况分组验证，真实量与标签列不进入训练；统一比较 RF、ExtraTrees、KNN、逻辑回归、XGBoost、MLP 和分段 NumPy 1D-CNN。\n"
        "2. v06 只在模型副本中接出 IGBT/Diode 已有测量向量，不改变 v05 基线。首轮发现 50 µs 与 PWM 周期同频造成 S1 固定相位混叠，随后仅四个器件量保留 1 µs，其余 52 个日志仍为 50 µs。\n"
        "3. 每个 50 µs 窗内联合筛选 `|I|≥0.5 A` 的样本，计算导通比例与 `median(|V/I|)`，避免把关断电压与另一相位电流相除。\n"
        "4. 高阻阈值只由 0.02/0.10 Ω 开发集冻结为 0.0105 Ω；0.05 Ω 新种子独立集只用于资格判定，不回调阈值。\n"
        "5. 最终分别打包主五类逻辑回归 + ExtraTrees 和高阻物理阈值专用产物。"
    )
    item_by_id(blocks, "problem_intro")["body"] = (
        "## 问题链已从权限阻塞推进到采样混叠并完成闭环\n\n"
        "完整日志记录 P01–P29。旧模型问题来自工况漂移、标签权重、窗口泄漏和物理不可观测性；本轮又发现统一表漏列与 PWM 同频采样混叠。两者修复后，高阻在新增同步器件观测下通过独立验证。下表保留历史失败与当前边界，便于组会说明为何必须改观测而不是继续调模型。"
    )
    item_by_id(blocks, "problem_intro")["sourceId"] = "problem_log"
    item_by_id(blocks, "limitations")["body"] = (
        "## 仿真 100% 仍不等同于真实硬件认证\n\n"
        "- 主模型与高阻专用模型的 100% 指标都来自 Simulink；尚未覆盖温度、老化、跨器件、测量带宽、同步误差、ADC 量化和 HIL 实时调度。\n"
        "- 高阻专用结果近乎确定性，是因为仿真直接输出器件模型内部电流/电压；真实传感器噪声与带宽可能缩小 0.001–0.02 Ω 的安全间隔。\n"
        "- 高阻 Ron 从 t=0 生效，仍缺少同一运行内动态劣化过程；独立验证只证明新工况/新随机种子/新阻值下的静态泛化。\n"
        "- 无 v06 同步器件测量时，高阻仍不支持；不得把专用产物应用到 v05 通用信号。\n\n"
        "**分享评级：带硬件边界可分享。** 可在组会上声明“仿真直接测量方案已通过”，但不能声明真实硬件、安全认证或无新增传感器方案已通过。"
    )
    item_by_id(blocks, "next_steps")["body"] = (
        "## 下一步转向传感器实现与跨设备盲测\n\n"
        "1. 保持主五类逻辑回归 + ExtraTrees 不变，将高阻阈值专用模型作为有明确传感器前置条件的并行分支。\n"
        "2. 在 HIL/实测中验证 S1/S2 压降与电流的同步精度、带宽、量化、温漂与电磁噪声，重新估计健康上界和最小故障下界。\n"
        "3. 生成动态 Ron 劣化或分段状态衔接数据，测量检测延迟和过渡期误报。\n"
        "4. 采用跨器件/跨温度留一验证并保留新的未见阻值盲测；阈值不得在盲测后回调。\n"
        "5. 若硬件无法增加同步测量，回退为“高阻不支持”，不要再融合旧控制信号模型。"
    )
    item_by_id(blocks, "further_questions")["body"] = (
        "## 组会建议讨论的问题\n\n"
        "- 真实硬件能否提供每个开关的同步压降/电流，1 µs 是可实现采样还是仅仿真上限？\n"
        "- 可接受的最小高阻幅值、检测延迟和正式 FAR 门槛分别是多少？\n"
        "- 动态劣化应通过在线参数变化、状态衔接，还是 HIL 故障注入实现？\n"
        "- 非导通开关应等待自然模式切换，还是设计低扰动主动诊断脉冲？\n"
        "- HIL/实测至少需要多少器件、温度点、SOC、功率与老化等级才能重新声明资格？"
    )

    problem_table = item_by_id(manifest["tables"], "problem_summary_table")
    problem_table["subtitle"] = "从完整 P01–P29 日志中提炼的组会优先问题"
    item_by_id(manifest["sources"], "problem_summary")["label"] = "组会关键问题摘要（P01–P29）"

    headline = snapshot["datasets"]["headline_metrics"][0]
    headline.pop("high_r_passed", None)
    headline.pop("high_r_candidates", None)
    headline["high_r_macro_f1"] = 1.0
    headline["high_r_healthy_far"] = 0.0

    comparison = pd.read_csv(RESULT_ROOT / "model_comparison.csv")
    comparison = comparison.loc[comparison["split"] == "independent_validation"].copy()
    role = {
        "physics_threshold": "首选：零训练、可解释",
        "logistic_regression": "对照",
        "extra_trees": "对照",
        "random_forest": "对照",
        "knn": "对照",
        "mlp": "对照",
        "1d_cnn": "对照；无额外收益",
    }
    snapshot["datasets"]["hr_direct_comparison"] = [
        {
            "model": row.model,
            "macro_f1": float(row.macro_f1),
            "run_macro_f1": float(row.run_macro_f1),
            "fault_recall": float(row.fault_recall),
            "healthy_far": float(row.healthy_false_alarm_rate),
            "train_seconds": float(row.train_seconds),
            "role": role[row.model],
        }
        for row in comparison.itertuples(index=False)
    ]

    problems = snapshot["datasets"]["problem_summary"]
    for row in problems:
        if row["id"] in {"P23/P25", "P26"}:
            row["status"] = "旧通用信号失败证据"
    additions = [
        {"id": "P27", "priority": 7, "problem": "新增开关信号已读取但首版未加入 rawTable，仿真成功却统一表漏列。", "impact": "静默丢字段会误判观测端失败。", "resolution": "补齐表构造并提升架构版本，resume 自动重跑旧结构。", "status": "已修复"},
        {"id": "P28", "priority": 8, "problem": "50 µs 日志与 PWM 同频，固定采到 S1 关断相位，电流看似恒为 0。", "impact": "采样混叠造成假不可观测。", "resolution": "四个器件量改为 1 µs，并在 50 µs 窗内联合估计 |V/I|。", "status": "已修复"},
        {"id": "P29", "priority": 9, "problem": "v06 32 Run 开发 + 16 Run 未见阻值独立验证均完全分离。", "impact": "高阻结论从旧信号不支持变为新增测量下支持。", "resolution": "冻结 0.0105 Ω 物理阈值并打包专用产物。", "status": "仿真通过，待 HIL"},
    ]
    problems[:] = [row for row in problems if row["id"] not in {a["id"] for a in additions}]
    problems.extend(additions)
    priority_order = [
        "P01/P07", "P05/P06", "P08", "P15/P16", "P17/P18", "P19",
        "P23/P25", "P26", "P27", "P28", "P29", "P10/P11", "P12",
        "P09", "P14",
    ]
    priority = {problem_id: index + 1 for index, problem_id in enumerate(priority_order)}
    for row in problems:
        row["priority"] = priority[row["id"]]
    problems.sort(key=lambda row: (row["priority"], row["id"]))

    rich_sources = artifact["sources"]
    for source in [
        {
            "id": "hr_direct_summary",
            "label": "v06 开关直接电阻专用模型摘要",
            "path": "ML/results/switch_observability_specialist/summary.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('ML/results/switch_observability_specialist/summary.json')",
                "description": "读取冻结阈值、开发/独立集规模及健康/故障电阻边界。",
                "executed_at": GENERATED_AT,
                "tables_used": ["ML/results/switch_observability_specialist/summary.json"],
                "filters": ["窗口起点不早于 0.4 s", "IsTrainingEligible=1", "独立 Ron=0.05 Ω"],
                "metric_definitions": ["活动开关 Ron 为每个 50 µs 窗内 |I|≥0.5 A 样本的 |V/I| 中位数。", "冻结阈值 0.0105 Ω 仅由开发集健康上界与故障下界确定。"],
            },
        },
        {
            "id": "hr_direct_comparison",
            "label": "v06 高阻专用模型对比与独立验证",
            "path": "ML/results/switch_observability_specialist/model_comparison.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('ML/results/switch_observability_specialist/model_comparison.csv') WHERE split = 'independent_validation'",
                "description": "比较冻结物理阈值及六类学习模型在未见 0.05 Ω 独立集上的窗口级和 Run 级指标。",
                "executed_at": GENERATED_AT,
                "tables_used": ["ML/results/switch_observability_specialist/model_comparison.csv"],
                "filters": ["split=independent_validation", "16 Run", "1,904 合格窗口"],
                "metric_definitions": ["健康 FAR 为健康窗口或 Run 被报为高阻的比例。", "Run 分数为所属窗口分数的中位数。"],
            },
        },
    ]:
        upsert(rich_sources, source)
    problem_log_source = item_by_id(rich_sources, "problem_log")
    problem_log_source["label"] = "完整问题日志 P01–P29"
    problem_log_source["description"] = (
        "记录旧基线、分布漂移、权限阻塞、可观测性、表字段遗漏、"
        "PWM 采样混叠及 v06 高阻闭环验证。"
    )
    problem_summary_source = item_by_id(rich_sources, "problem_summary")
    problem_summary_source["label"] = "组会关键问题摘要（P01–P29）"
    problem_summary_source["query"]["description"] = (
        "读取从 P01–P29 中提炼的组会优先问题。"
    )
    problem_summary_source["query"]["executed_at"] = GENERATED_AT

    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (REPORT_ROOT / "problem_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "priority", "problem", "impact", "resolution", "status"],
        )
        writer.writeheader()
        writer.writerows(problems)


if __name__ == "__main__":
    main()
