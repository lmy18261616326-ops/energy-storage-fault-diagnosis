# 储能双向 DC/DC 故障诊断与安全控制

> MATLAB/Simulink + Python 的可审计故障诊断研究工程，覆盖物理建模、故障注入、批量仿真、数据质量、分组验证、冻结盲测与域偏移审计。

[![Python tests](https://github.com/lmy18261616326-ops/energy-storage-fault-diagnosis/actions/workflows/python-tests.yml/badge.svg)](https://github.com/lmy18261616326-ops/energy-storage-fault-diagnosis/actions/workflows/python-tests.yml)

## 项目定位

本项目研究储能电池与直流母线之间的双向 Buck-Boost 变换器，目标是在充电、放电和待机工况下：

- 建立双闭环控制、模式切换与能量保护模型；
- 注入传感器偏置、S1/S2 开路和开关高阻故障；
- 构建可断点恢复、可追溯的仿真数据采集流水线；
- 使用按工况隔离的机器学习评估，防止窗口泄漏和调参泄漏；
- 用冻结后盲测、故障前负对照和外部数据域偏移决定模型是否可以晋级。

```mermaid
flowchart LR
    C["工况与故障配置"] --> S["Simulink v05 / v06"]
    S --> R["原始测量与真值校验"]
    R --> F["滑窗特征与数据质量"]
    F --> G["按 OperatingPointID 分组"]
    G --> M["嵌套选择 / 校准 / 阈值"]
    M --> B["冻结后盲测"]
    B --> A["负对照与域偏移审计"]
    A --> D{"部署门禁"}
```

## 模型版本的明确角色

| 模型 | 角色 | 当前用途 |
|---|---|---|
| `main_model_fd_v05_energyprotect.slx` | 主基线 | 五类故障数据采集、能量保护和主要复现入口 |
| `main_model_fd_v06_switchobservability.slx` | 可观测性扩展 | 器件电压/电流直接观测与开关高阻专项研究 |
| v03/v04 | 历史对照 | 仅用于版本演进与回归对照 |

默认数据采集配置以 v05 为基线；只有开关可观测性专项脚本才切换到 v06。

## 已验证的结果与证据边界

| 评估轨道 | 主要结果 | 可以说明 | 不能声称 |
|---|---|---|---|
| v13 五类冻结盲测 | 216/216 Run 成功；Macro-F1 43.79%；S1/S2 开路召回率均为 13.79% | 盲测暴露了开关故障可观测性和高负载误报问题 | 该版本可部署 |
| v17 仿真主任务 | 416 个独立 Run、28 个工况；多个结构化模型 Run 级 OOF 达到 100% | 在当前主动可观测仿真范围内类别可分 | 跨设备、跨拓扑或实物泛化为 100% |
| v19 合成参考域 | 固定 Extra Trees Macro-F1 94.93%；嵌套 OOF 90.72%；健康 FAR 0%；高阻召回 90.48% | 负对照接近随机水平，评估程序能拦截可疑晋级 | 合成数据等同实测 |
| 外部实验波形审计 | 9,261 行、441 组、21 状态；共享特征覆盖 10.5%；82.6% 单元超训练范围 | 存在显著域偏移，外部波形可用于状态可分性研究 | 外部六类故障准确率已验证 |

详细口径见 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)。项目的原则是：失败的盲测和被阻止的模型晋级也是正式结果，不通过更换口径隐藏。

## 快速复现

### 1. Python 测试

```powershell
py -3.13 -m venv ML\.venv
& 'ML\.venv\Scripts\python.exe' -m pip install -r 'ML\requirements.txt'
& 'ML\.venv\Scripts\python.exe' -m pytest 'ML\tests' -q
```

当前审计环境的精确版本位于 `ML/requirements-lock.txt`。

### 2. 无大数据端到端演示

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_portfolio_demo.py'
```

演示会生成小型合成数据，执行分组划分、特征排除、Random Forest 训练和 GroupKFold，产物写入被 Git 忽略的 `ML/results/portfolio_demo/`。它只验证管线，不是研究成绩。

### 3. MATLAB/Simulink

```matlab
openProject("EnergyStorageFaultDiagnosis.prj");
run("audit_main_model.m");
```

默认审计 v05。如需检查 v06，在启动 MATLAB 前设置环境变量 `ENERGY_STORAGE_MODEL=main_model_fd_v06_switchobservability`。完整复现说明见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## 仓库结构

```text
.
├── simulink/models/                 # 主模型与版本对照
├── simulink/experiments/            # 仿真、故障注入和数据处理脚本
├── ML/src/energy_fault_ml/          # 分组划分、特征、训练、评估和诊断
├── ML/scripts/                      # 实验入口与报告构建
├── ML/tests/                        # Python 自动化测试
├── docs/specs/                      # 控制架构、Stateflow 与测试规格
└── docs/                            # 状态、复现、学习与面试资料
```

## 数据与开源边界

仓库不包含大型仿真数据集、训练模型、第三方论文、外部压缩包和 MATLAB 缓存。这些内容可能很大，也可能受到许可或数据来源限制。完整数据需按 `docs/REPRODUCIBILITY.md` 重新生成或由合法数据源单独获取。

## 当前局限

- 最终晋级模型仍需要一套消除 seed/phase 标签相关的新冻结盲测。
- 外部实验数据的信号语义和六类故障标签不匹配，不能计算主模型外部故障准确率。
- 高阻压力测试不等同于完整动态功率级仿真、HIL 或硬件试验。
- 项目是仿真和研究原型，不是经过功能安全认证的产品。

## 学习与面试

系统化学习路线、核心概念、高频追问、STAR 叙事和模拟面试答案见 [`docs/学习与面试应答攻略.md`](docs/%E5%AD%A6%E4%B9%A0%E4%B8%8E%E9%9D%A2%E8%AF%95%E5%BA%94%E7%AD%94%E6%94%BB%E7%95%A5.md)。
