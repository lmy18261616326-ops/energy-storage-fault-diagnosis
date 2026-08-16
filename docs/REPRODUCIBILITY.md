# 复现指南

## 1. 支持环境

- Windows 10/11
- MATLAB/Simulink R2025a（电力电子模型所需产品以本机安装为准）
- Python 3.10+；已审计环境为 Python 3.13
- 普通测试不需要仓库外的大型数据

## 2. Python 环境

```powershell
Set-Location '<repository-root>'
py -3.13 -m venv 'ML\.venv'
& 'ML\.venv\Scripts\python.exe' -m pip install --upgrade pip
& 'ML\.venv\Scripts\python.exe' -m pip install -r 'ML\requirements.txt'
```

- `ML/requirements.txt` 保留兼容的最低版本范围。
- `ML/requirements-lock.txt` 记录本次审计实际通过的主要包版本。

## 3. 自动化测试

```powershell
& 'ML\.venv\Scripts\python.exe' -m pytest 'ML\tests' -q
```

测试覆盖分组划分、特征泄漏排除、校准、诊断、事件特征、高阻物理量和参考 HDF5 评估等关键路径。

## 4. 端到端轻量演示

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_portfolio_demo.py'
```

该命令会：

1. 在 `ML/results/portfolio_demo/` 下生成小型五类合成窗口数据；
2. 按 `OperatingPointID` 分组划分；
3. 排除真值、标签派生字段和常量特征；
4. 训练 Random Forest 并执行三折 GroupKFold；
5. 检查产物和返回码，打印验证/测试 Macro-F1。

该合成演示只用于证明软件流程可运行，不能作为论文或简历成绩。

## 5. MATLAB Project 与模型审计

打开项目：

```matlab
proj = openProject("EnergyStorageFaultDiagnosis.prj");
```

默认审计 v05：

```matlab
run("audit_main_model.m");
```

检查 v06 时，在 PowerShell 中设置：

```powershell
$env:ENERGY_STORAGE_MODEL = 'main_model_fd_v06_switchobservability'
matlab -batch "run('audit_main_model.m')"
```

审计脚本会检查模型加载、更新/编译、未连接线和端口，并尝试 20 ms 短仿真。

## 6. 正式仿真数据

完整数据不进入 Git。在 MATLAB 中运行：

```matlab
cd(fullfile("simulink", "experiments", "sensor_bias", "scripts"));
result = collect_fault_dataset();
```

运行前必须确认：

- `signal_config.m` 中的模型和输出路径；
- `overwritePolicy="resume"`；
- 剩余磁盘空间和预期 Run 数；
- 不在冻结前查看或调参盲测数据。

## 7. 可复现性检查清单

- 记录 Git commit、MATLAB/Python 版本和配置文件。
- 记录数据文件大小、修改时间和 SHA-256。
- 分组单元必须是 `OperatingPointID`/Run，不允许相关窗口跨训练与测试集。
- 仅在训练折内做特征筛选、超参选择、校准和阈值搜索。
- 盲测开始前写入模型哈希和冻结状态。
- 对故障前时段执行负对照，检查随机种子、相位、文件顺序等标签捷径。
