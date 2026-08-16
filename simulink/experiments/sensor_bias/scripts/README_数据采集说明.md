# 故障诊断数据采集程序

## 目录结构

```text
scripts/
├─ collect_fault_dataset.m       主程序
├─ build_simulation_cases.m      生成每行一次仿真的配置表
├─ extract_raw_signals.m         信号读取、时间对齐、标签生成
├─ extract_window_features.m     滑动窗口和特征提取
├─ validate_dataset.m            数据质量检查、报告和绘图
├─ signal_config.m               唯一集中配置入口
├─ split_dataset_by_run.m        默认按 OperatingPointID 分组划分数据集
├─ MCP_数据处理修复报告.md       本轮模型核查、脚本修复和模型待改项
├─ README_数据采集说明.md
├─ _legacy/                      旧版脚本（保留但不再参与运行）
└─ dataset_output/               首次正式运行时自动建立
   ├─ raw_runs/
   │  ├─ run_00001.mat
   │  └─ ...
   ├─ combined/
   │  ├─ raw_dataset.mat/csv
   │  ├─ feature_dataset.mat/csv
   │  ├─ simulation_cases.mat/csv
   │  ├─ failed_runs.mat/csv
   │  ├─ dataset_field_groups.mat
   │  └─ dataset_report.txt
   ├─ figures/
   │  └─ *.png
   ├─ simulink_cache/            Simulink 缓存
   ├─ simulink_codegen/          代码生成中间文件
   ├─ parallel_jobs/             并行池 JobStorageLocation
   └─ temp/                      批量运行期间的临时文件
```

## 运行方法

在 MATLAB 中把当前文件夹设为 `scripts`，运行：

```matlab
result = collect_fault_dataset();
```

主程序会在运行期间同时重定向 `TEMP`、`TMP`、MATLAB `tempdir`、
Java `java.io.tmpdir`、Simulink 缓存和并行池任务目录。由于部分 MATLAB
组件会在主程序开始前创建临时文件，如需保证 MATLAB 启动后全程不使用
C 盘临时目录，请先关闭当前 MATLAB，再双击项目根目录下的
`start_project_matlab.cmd` 启动。启动阶段临时目录为项目内的
`work\temp`，采集阶段临时目录仍为
`dataset_output\temp`；这样可避免临时目录位于 Simulink 默认
CacheFolder 内而产生冲突。

批量采集默认关闭全模型状态 `xout`、To Workspace 到数据检查器的流式
传输、无用的 Scope/输出/时间/数据存储记录。52 个 To Workspace 模块
直接按 `cfg.sampleTime=50e-6` 写入结果，避免 1 us 原始步长产生数十 GB
临时 `.dmr`。因此可选字段 `PIiIntegral` 默认是 NaN，并会从特征列中
明确排除；只有确实需要调试全部模型状态时，才设置
`overrides.execution.saveModelStates=true`。

默认使用 2 个 worker 分批并行，每批 16 个 Run；每批结束后立即保存
`raw_runs` 并释放 SimulationOutput，适合当前 16 GB 内存机器。若需改回
串行：

```matlab
overrides.execution.useParallel = false;
result = collect_fault_dataset(overrides);
```

建议保持默认 `numWorkers=2`、`parallelBatchSize=16`；不要直接开启 8 个
worker，否则多个电力电子模型同时编译容易触发内存交换，反而更慢。

扩大正式数据集时只改 `signal_config.m` 中的 `cfg.cases` 数组，或在调用
时传入覆盖结构体。建议先保持 `overwritePolicy="resume"`，中断后可以从
已保存的单次 MAT 文件继续。

数据集划分示例：

```matlab
content = load(fullfile(result.Config.output.combined, ...
    "feature_dataset.mat"), "featureDataset");
splitFolder = fullfile(result.Config.output.root, "split");
[trainSet, validationSet, testSet, summary] = split_dataset_by_run( ...
    content.featureDataset, [0.70 0.15 0.15], 240727, splitFolder);
```

## 故障标签与模型内部代号

| 数据集 `ScenarioFaultID` | 故障 | 模型内部代号 | 模型变量 |
|---:|---|---:|---|
| 0 | 健康 | 0 | 全部故障入口为 0 |
| 1 | 母线电压传感器偏移 | 2 | `FD_VBUS_BIAS` |
| 2 | 电感电流传感器偏移 | 3 | `FD_IL_BIAS` |
| 3 | S1 开路 | 6 | `FD_S1_OPEN` |
| 4 | S2 开路 | 7 | `FD_S2_OPEN` |

模型内部代号不会直接作为机器学习标签。故障发生前 `ActiveFaultID=0`；
故障激活后才等于 `ScenarioFaultID`。传感器偏移窗口由
`FaultActiveRatio` 标注；S1/S2 开路窗口还必须满足对应门极本应导通，
由 `FaultObservableRatio` 标注。故障场景的故障前窗口、不可观测的开路
窗口和过渡窗口仍保留，但 `IsTrainingEligible=0`，不会被当作健康样本训练。

## 当前模型适配说明

目标模型：

```text
simulink/models/main_model_fd_v05_energyprotect.slx
```

当前模型虽然启用了 `logsout`，主要信号实际上由 `To Workspace` 保存到
`SimulationOutput`。读取器同时支持两种来源，不要求重新接线。

以下接口已经可由 `SimulationInput` 设置：

- `ModeCommand`：`FD_MODE_OVERRIDE_ENABLE`、`FD_MODE_COMMAND`
- `SOCInit`：Battery 模块的 `SOC` 参数
- `VbusRef`：`V_ref` 模块的 `Value`
- `Rload`：`R_H_Bus` 模块的 `Resistance`
- `Pload`：转换为 `FD_LOAD_STEP_A=Pload/VbusRef`
- `Rbat`、`Cbus`、`CbusESR`：通过现有 `FD_RBAT`、`FD_CBUS`、
  `FD_CBUS_ESR` 做按随机种子可复现的域随机化
- 故障类型、幅值、开始和结束时间：`FD_*` 模型工作区变量
- `IrefLevel`：`FD_IREF_LEVEL`，最终仍受充/放电电流限值约束
- 传感器非理想：每个 Run 写入噪声标准差、量化步长、采样时间和随机种子
- 功率平衡：写入三路功率方向系数、滤波系数和初始储能

`VbatInit` 仍只作为元数据；当前电池初始端电压由 SOC 和电池参数决定，
不能独立指定。`PIiIntegral` 从状态数据集 `xout` 读取；`SOC_est` 来自独立
库仑计。旧的 `PowerResidual=Pbat-Pbus` 仍默认排除，新的
`PowerBalanceResidual` 包含源、电池、负载和母线/电感储能变化，可进入
`featureColumns`。

## Simulink 信号日志配置清单

现有模型已经能提供大部分信号，不要求新增接线。若以后统一改为
Signal Logging，建议给信号线设置以下名称并启用日志：

| 统一字段 | 当前来源/建议日志名 | 必要性 |
|---|---|---|
| `ModeCommand` | `log_mode_command` | 必要 |
| `ConverterEnable` | 当前由 `mode_id` 派生 | 可选 |
| `Iref` | `log_Iref` | 必要 |
| `VbusRef` | 当前由仿真配置填充 | 可选 |
| `IL_meas` | `log_I_L` | 必要 |
| `Ibat_meas` | `log_Ibat` | 必要 |
| `Vbus_meas` | `log_Vbus` | 必要 |
| `Vbat_meas` | `log_Vbat` | 必要 |
| `Iload_meas` | `load_current`；禁止回退到支路量 `I_Rh` | 必要 |
| `SOC_est` | `SOC_est` | 必要 |
| `CurrentError` | `log_Ierr` | 必要 |
| `VoltageError` | `log_V_err` | 可选 |
| `DutyRaw` | `log_Duty_cmd` | 可选 |
| `DutyApplied` | `log_Duty_applied` | 必要 |
| `PIiOut` | `Duty_raw`（当前实际为电流 PI 输出） | 可选 |
| `PIiIntegral` | 状态数据集 `xout/PIiIntegral` | 可选 |
| `PIvOut` | `PI_voltage` | 可选 |
| `SatFlag` | `sat_flag_I OR sat_flag_V` | 可选 |
| `S1GateCmd` / `S1GateDuty` | logsout 的 `gate_buck_cmd`；后者按 50 μs 区间求平均 | 必要 |
| `S2GateCmd` / `S2GateDuty` | logsout 的 `gate_boost_cmd`；后者按 50 μs 区间求平均 | 必要 |
| `IL_true` | `log_IL_true` | 必要，仅验证 |
| `Ibat_true` | `log_Ibat_true` | 必要，仅验证 |
| `Vbus_true` | `log_Vbus_true` | 必要，仅验证 |
| `Vbat_true` | `log_Vbat_true` | 必要，仅验证 |
| `SOC_true` | 当前暂用 `log_SOC` | 可选，仅验证 |
| `Ibus_source` | `source_current` | 必要 |
| `Psource_meas` | `Psource_meas` | 必要 |
| `Pload_meas` | `Pload_meas` | 必要 |
| `Pstored_meas` | `Pstored_meas` | 必要 |
| `PowerBalanceResidual` | `PowerBalanceResidual` | 必要 |
| `FaultActive` | `log_fault_active` | 用于核验 |
| `TransitionWindow` | MATLAB 根据故障时刻生成 | 标签 |

若增加独立 `SOC_true` 或 `ConverterEnable` 日志，优先从现有信号使用
Goto/From 接入日志区，不要把主电路和控制分析电路直接拉长线连接。
