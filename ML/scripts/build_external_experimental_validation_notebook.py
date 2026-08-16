"""Build and execute the reader-facing external experimental validation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "ML" / "notebooks" / "external_experimental_validation_jh69mxmx99_v1.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> int:
    cells = [
        markdown(
            "# 真实实验数据外部验证：Mendeley jh69mxmx99 v1\n\n"
            "本笔记本复核下载完整性、实验波形质量、真实 V/I 对文件级 C 状态的可区分性，"
            "并对冻结六类合成模型执行受限的诊断性压力测试。数据来源："
            "[Mendeley Data, DOI 10.17632/jh69mxmx99.1]"
            "(https://data.mendeley.com/datasets/jh69mxmx99/1)。"
        ),
        markdown(
            "## tl;dr\n\n"
            "- 数据包与全部工作簿结构通过完整性门。\n"
            "- 真实 V/I 在文件内网格位置隔离的 5 折验证中，可以显著区分 21 个文件级状态。\n"
            "- 冻结模型只有 100/950 个特征可做假设性对齐，且没有重叠故障标签；因此不能计算真实六类准确率。\n"
            "- 压力测试中全部样本被判为 high_resistance，同时存在强域偏移和高预测熵，模型仍不得晋级。"
        ),
        code(
            "from pathlib import Path\n"
            "import json, subprocess\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'ML':\n"
            "    ROOT = ROOT.parent\n"
            "RESULT = ROOT / 'ML' / 'results' / 'external_experimental_validation_jh69mxmx99_v1'\n"
            "assert RESULT.is_dir(), RESULT\n"
            "summary = json.loads((RESULT / 'validation_summary.json').read_text(encoding='utf-8'))\n"
            "quality = json.loads((RESULT / 'data_quality.json').read_text(encoding='utf-8'))\n"
            "compatibility = json.loads((RESULT / 'model_compatibility.json').read_text(encoding='utf-8'))\n"
            "domain = json.loads((RESULT / 'domain_shift_summary.json').read_text(encoding='utf-8'))\n"
            "cv = json.loads((RESULT / 'c_state_cv_summary.json').read_text(encoding='utf-8'))\n"
            "print(f\"Loaded audited snapshot: {RESULT}\")"
        ),
        code(
            "pd.Series({\n"
            "    '完整性门': summary['data_integrity_passed'],\n"
            "    '真实波形数': quality['shape']['traces'],\n"
            "    'V/I 数值单元': quality['shape']['voltage_current_numeric_cells'],\n"
            "    'C 状态 OOF Macro-F1': cv['macro_f1'],\n"
            "    '机会水平': cv['chance_macro_f1'],\n"
            "    '负对照 Macro-F1': cv['within_group_label_permutation_macro_f1'],\n"
            "    '冻结模型特征覆盖': compatibility['feature_coverage'],\n"
            "    '域外数值占比': domain['external_cells_outside_synthetic_training_minmax_fraction'],\n"
            "}, name='核心结论').to_frame()"
        ),
        markdown(
            "## Context & Methods\n\n"
            "冻结模型是 950 特征的 Extra Trees 六分类器，训练域为合成参考 HDF5。外部数据是 21 个 XLSX，"
            "每个文件含 441 对 V/I 波形，每条 2,000 点。审计将列名中的 1–9,261 识别为全局源样本 ID，"
            "并从每个文件内顺序派生 1–441 的网格位置。\n\n"
            "外部基准只使用两条可观测波形生成 100 个与冻结管线同构的滑窗统计特征。"
            "五折 GroupKFold 按文件内网格位置隔离，使同一位置在 21 个 C 文件中的记录不会跨越训练/测试。"
            "负对照在每个网格位置内独立打乱 C 标签。\n\n"
            "冻结模型压力测试假设 V 对应 `vbus_meas_V`、I 对应 `il_meas_A`，未提供的 850 个特征由冻结管线"
            "按训练中位数填补。由于信号位置语义未经数据源确认、标签不重叠，该输出只用于发现域外行为。"
        ),
        code(
            "REGENERATE = False\n"
            "if REGENERATE:\n"
            "    command = [\n"
            "        str(ROOT / 'ML' / '.venv' / 'Scripts' / 'python.exe'),\n"
            "        str(ROOT / 'ML' / 'scripts' / 'validate_external_experimental_dataset.py'),\n"
            "        '--dataset-dir', str(ROOT / 'external_data' / 'mendeley_jh69mxmx99_v1' / 'raw'),\n"
            "        '--zip-path', str(ROOT / 'external_data' / 'mendeley_jh69mxmx99_v1' / 'jh69mxmx99_v1.zip'),\n"
            "        '--model', str(ROOT / 'ML' / 'models' / 'reference_hdf5_optimized_v19' / 'optimized_controller_fault_model_provisional.joblib'),\n"
            "        '--training-features', str(ROOT / 'ML' / 'results' / 'reference_hdf5_model_evaluation_v18' / 'controller_engineered_features.csv'),\n"
            "        '--output-dir', str(RESULT),\n"
            "    ]\n"
            "    subprocess.run(command, check=True)\n"
            "else:\n"
            "    print('Using the frozen audited result snapshot; set REGENERATE=True to re-parse all 21 XLSX files.')"
        ),
        markdown("## Data\n\n下面展示数据包、工作簿和语义完整性。"),
        code(
            "pd.DataFrame([\n"
            "    {'检查': 'ZIP 字节数', '结果': quality['zip']['size_match'], '证据': quality['zip']['bytes']},\n"
            "    {'检查': 'ZIP SHA-256', '结果': quality['zip']['sha256_match'], '证据': quality['zip']['sha256']},\n"
            "    {'检查': '工作簿/工作表结构', '结果': quality['integrity']['structure_pass'], '证据': '21 workbooks / 189 sheets'},\n"
            "    {'检查': '缺失或非数', '结果': quality['integrity']['nonnumeric_or_missing_cells'] == 0, '证据': quality['integrity']['nonnumeric_or_missing_cells']},\n"
            "    {'检查': '重复 V/I 波形对', '结果': quality['integrity']['duplicate_trace_pairs'] == 0, '证据': quality['integrity']['duplicate_trace_pairs']},\n"
            "    {'检查': '采样间隔元数据', '结果': quality['label_and_semantic_completeness']['time_or_sampling_interval_available'], '证据': '未提供'},\n"
            "    {'检查': 'R1/R2/C 数值映射', '结果': quality['label_and_semantic_completeness']['c_numeric_values_available'], '证据': '未提供'},\n"
"] )"
        ),
        code("display(Image(filename=str(RESULT / 'experimental_waveform_examples.png')))"),
        markdown(
            "图中仅使用源文件的 Index 作为横轴；数据源没有给出采样间隔，因此不能把横轴换算为秒，也不能报告 Hz。"
        ),
        markdown("## Results\n\n### 真实 V/I 可区分文件级 C 状态，但这不是六类故障准确率"),
        code("display(Image(filename=str(RESULT / 'c_state_oof_recall.png')))"),
        code(
            "folds = pd.DataFrame(cv['folds'])\n"
            "for column in ['accuracy', 'balanced_accuracy', 'macro_f1']:\n"
            "    folds[column] = folds[column].round(3)\n"
            "folds"
        ),
        markdown(
            "Macro-F1 的网格位置分组 bootstrap 95% 区间为 "
            "0.491–0.517；五折结果接近，且负对照回落到机会水平附近。"
        ),
        markdown("### 冻结模型输入覆盖不足，并发生显著域偏移"),
        code("display(Image(filename=str(RESULT / 'input_coverage.png')))"),
        code("display(Image(filename=str(RESULT / 'domain_shift_key_features.png')))"),
        code(
            "pd.DataFrame(domain['key_features'])[[\n"
            "    'feature', 'training_min', 'training_median', 'training_max',\n"
            "    'external_min', 'external_median', 'external_max',\n"
            "    'external_outside_training_range_fraction'\n"
            "]]"
        ),
        markdown("### 诊断性压力测试显示单类塌缩，不构成真实验证通过"),
        code("display(Image(filename=str(RESULT / 'diagnostic_model_probability_heatmap.png')))"),
        code(
            "pd.Series({\n"
            "    '全部预测 high_resistance': compatibility['prediction_distribution']['high_resistance'],\n"
            "    '预测总数': sum(compatibility['prediction_distribution'].values()),\n"
            "    '最大概率中位数': compatibility['median_max_probability'],\n"
            "    '归一化熵中位数': compatibility['median_normalized_entropy'],\n"
            "    '计算六类准确率': compatibility['accuracy_computed'],\n"
            "})"
        ),
        markdown(
            "## Takeaways\n\n"
            "1. 数据源是真实、结构完好的实验波形，可用于外部信号质量、状态可区分性和后续域适配研究。\n"
            "2. 它不能直接验证当前六类冻结模型：标签体系不重叠，12 条原始控制器通道仅假设性覆盖 2 条。\n"
            "3. 全部预测塌缩到 high_resistance、低置信度和强域偏移共同表明当前模型对该实验域不可用。\n"
            "4. 若要完成真实六类验证，应采集同一控制器信号清单，并提供 healthy/vbus_bias/il_bias/S1_open/S2_open/high_resistance 的运行级真值。\n\n"
            "当前决策：保留模型的 `provisional_synthetic_only` 状态，不晋级。"
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "energy-storage-ml",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
