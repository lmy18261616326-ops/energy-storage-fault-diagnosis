# 储能双向 DC/DC 可审计机器学习故障诊断

本目录提供统一的 Random Forest、RBF-SVM、XGBoost 五分类基准。三个模型
共用同一份窗口数据、同一次分组划分和同一组输入特征，主指标为
Macro-F1。

> 项目的官方结果口径、证据边界和晋级状态以
> `../docs/PROJECT_STATUS.md` 为准。本 README 主要描述通用训练管线。

标签定义：

| `WindowFaultID` | 类别 |
|---:|---|
| 0 | 健康 |
| 1 | 母线电压传感器偏置 |
| 2 | 电感电流传感器偏置 |
| 3 | S1 开路 |
| 4 | S2 开路 |

## 1. 环境安装（Windows PowerShell）

以下命令均可在项目根目录执行：

```powershell
Set-Location '<repository-root>'

# 本机没有 Python 3.11 时，使用已安装的 Python 3.10+。
# 当前工程初始化时检测到的是 Python 3.13。
py -3.13 -m venv 'ML\.venv'

# 若当前会话禁止运行 Activate.ps1，可只为当前 PowerShell 进程放行。
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& 'ML\.venv\Scripts\Activate.ps1'

python -m pip install --upgrade pip
python -m pip install -r 'ML\requirements.txt'
```

也可以不激活环境，直接使用：

```powershell
& 'ML\.venv\Scripts\python.exe' -m pytest 'ML\tests'
```

## 2. 生成正式数据

默认数据位置为：

```text
simulink/experiments/sensor_bias/scripts/dataset_output/combined/feature_dataset.csv
```

如果该文件不存在，在 MATLAB 中运行：

```matlab
projectRoot = fileparts(which('EnergyStorageFaultDiagnosis.prj'));
cd(fullfile(projectRoot, 'simulink', 'experiments', 'sensor_bias', 'scripts'));
result = collect_fault_dataset();
```

生成后应先检查同目录下的 `dataset_report.txt` 和失败运行清单。机器学习
程序只保留 `IsTrainingEligible != 0` 的窗口。

## 3. 运行测试

```powershell
Set-Location '<repository-root>'
& 'ML\.venv\Scripts\python.exe' -m pytest 'ML\tests'
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' --help
```

## 4. 运行三模型基准

从项目根目录执行：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --data 'simulink\experiments\sensor_bias\scripts\dataset_output\combined\feature_dataset.csv' `
  --config 'ML\configs\baseline.yaml' `
  --output 'ML\results'
```

脚本默认把模型保存到 `ML\models`。指定随机种子：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --seed 240727
```

只运行一个模型：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --models svm
```

启用 YAML 中定义的轻量验证集调参：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' --tune
```

从 `ML` 目录启动也可以：

```powershell
Set-Location '<repository-root>\ML'
& '.\.venv\Scripts\python.exe' '.\scripts\run_benchmark.py' `
  --data '..\simulink\experiments\sensor_bias\scripts\dataset_output\combined\feature_dataset.csv' `
  --config '.\configs\baseline.yaml' `
  --output '.\results'
```

## 5. 特征消融

`configs/baseline.yaml` 的默认值为：

```yaml
features:
  feature_set: physics_enhanced
  include_mode_command: true
```

可在命令行覆盖：

```powershell
# 只使用测量信号的基础统计特征
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --feature-set statistical

# 检查模型是否过度依赖工作模式
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --exclude-mode-command
```

`statistical` 只保留电流、电压、SOC、参考量等基础统计特征；
`physics_enhanced` 在此基础上允许控制量、功率平衡、电流一致性和残差类
在线特征。两种配置都会强制排除：

- 标签、运行编号、工况配置和故障元数据；
- `Validation_` 开头的验证专用字段；
- `IL_true`、`Ibat_true`、`Vbus_true`、`Vbat_true`、`SOC_true`；
- 当前 MATLAB 策略禁止使用的不完整 `PowerResidual`。

全缺失和常数特征只根据训练集删除。SVM 的中位数填补和标准化也只在
训练集拟合。

## 6. 分组与泄漏防护

默认按 `OperatingPointID` 整组划分 70%/15%/15%。程序还会检查：

- 同一个 `RunID` 只能属于一个 `OperatingPointID`；
- `RunID` 和 `OperatingPointID` 在三个集合之间均无交集；
- 每个类别在训练、验证、测试中均存在；
- 每类至少覆盖三个独立工况组，否则直接报错。

测试集不参与调参或 XGBoost 早停。XGBoost 早停只使用验证集。

## 7. 输出文件

`ML/results` 中包括：

- `model_comparison.csv`：验证集和测试集的汇总指标及耗时；
- `per_class_metrics.csv`：每类 Precision、Recall、F1、Support；
- `feature_list.txt`、`removed_features.csv`；
- 各模型验证集和测试集混淆矩阵；
- `split_group_assignments.csv`、`run_split_assignments.csv`；
- `split_summary.csv`、`config_used.yaml`、`run_metadata.json`；
- `tuning_trials.csv`。

序列化模型位于 `ML/models/*.joblib`。

## 8. 无正式数据时的端到端冒烟验证

下面的合成数据只用于检查工程是否能运行，不代表真实研究结果：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\tests\make_synthetic_dataset.py' `
  'ML\results\synthetic_feature_dataset.csv'

& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --data 'ML\results\synthetic_feature_dataset.csv' `
  --config 'ML\configs\smoke.yaml' `
  --output 'ML\results\smoke' `
  --model-dir 'ML\models\smoke'
```

合成数据上的指标只能证明代码、环境和输出链路可用，不能作为论文结果。

## 9. 健康误报诊断与分组交叉验证

在原有基准命令末尾增加 `--group-cv`：

```powershell
& 'ML\.venv\Scripts\python.exe' 'ML\scripts\run_benchmark.py' `
  --data 'simulink\experiments\sensor_bias\scripts\dataset_output\combined\feature_dataset.csv' `
  --config 'ML\configs\baseline.yaml' `
  --output 'ML\results\group_cv_analysis' `
  --model-dir 'ML\models\group_cv_analysis' `
  --group-cv
```

程序会继续生成原来的验证集和测试集结果，同时增加：

- `window_predictions.csv`：每个验证/测试窗口的真实类别和预测类别；
- `healthy_false_alarm_by_operating_point.csv`：每个工况的健康误报率；
- `healthy_false_alarm_by_dimension.csv`：按模式、SOC、负载和工况汇总误报；
- `healthy_false_alarm_destinations.csv`：健康窗口被错判成了哪种故障；
- `group_cv/summary.csv`：六折分组交叉验证的平均值、标准差、最小值和最大值；
- `group_cv/fold_metrics.csv`：每一折的完整指标；
- `group_cv/pooled_metrics.csv`：合并全部折外预测后的总体指标；
- `group_cv/healthy_false_alarm_*.csv`：交叉验证范围内的健康误报诊断。

六折交叉验证每折使用两个完整 `OperatingPointID` 作为测试组、两个作为验证组，其余八个用于训练。同一个工况和同一个 `RunID` 不会跨集合泄漏。程序还要求每一折都包含五个故障类别；如果当前数据无法满足，会停止并提示补充数据，而不是输出不可比较的分数。
