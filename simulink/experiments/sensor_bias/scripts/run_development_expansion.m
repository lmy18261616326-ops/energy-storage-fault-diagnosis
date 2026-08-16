%% 修复后开发数据扩充（不包含最终盲测集）
% 依次生成全工况健康、传感器故障、连续开路、部分开路、间歇开路和
% 高导通电阻数据。每个阶段使用独立目录和 RunID 前缀，可安全断点续跑。

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v13");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", "development");
base = signal_config();

common = struct();
common.output.overwritePolicy = "resume";
common.output.saveRunCSV = false;
common.output.saveCombinedCSV = true;
% Keep generated compilation paths short enough for the Windows 260
% character limit. Dataset outputs remain in their descriptive folders.
common.output.temp = fullfile(shortWorkRoot, "tmp");
common.output.cache = fullfile(shortWorkRoot, "cache");
common.output.codegen = fullfile(shortWorkRoot, "codegen");
common.output.parallelJobs = fullfile(shortWorkRoot, "jobs");
common.execution.useParallel = true;
common.execution.numWorkers = 2;
common.execution.parallelBatchSize = 6;
common.execution.showSimulationManager = false;
common.execution.fixedStep = 1e-6;
common.cases.randomSeedBase = 520001;
common.cases.domainRandomization.enabled = true;
common.cases.domainRandomization.RbatRelativeRange = 0.20;
common.cases.domainRandomization.CbusRelativeRange = 0.20;
common.cases.domainRandomization.CbusESRRelativeRange = 0.50;
common.cases.maxRunCount = 500;

% 1) 全部 12 个开发工况的健康覆盖。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase2_core_health");
phase.cases.runIDPrefix = "health_core";
phase.cases.repetitions = 4;
phase.faultList = base.faultList(base.faultList.FaultID == 0,:);
collect_fault_dataset(phase);

% 2) 重新生成修复后的两类传感器故障。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase3_sensor_faults");
phase.cases.runIDPrefix = "sensor";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.25 0.75];
phase.cases.faultDurations = 0.20;
phase.faultList = base.faultList( ...
    ismember(base.faultList.FaultID, [1 2]), :);
collect_fault_dataset(phase);

% 3) 连续完全开路：3 种持续时间带来不同发生/恢复时刻。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase4_switch_full_open");
phase.cases.runIDPrefix = "switch_full";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.25 0.75];
phase.cases.faultDurations = [0.05 0.15 0.30];
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 487e-6;
phase.faultList = switchFaultRows(base, "full_open", {1; 1});
collect_fault_dataset(phase);

% 4) 部分开路：每 0.5 ms 周期屏蔽 25%、50% 或 75% 的门极命令。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase5_switch_partial_open");
phase.cases.runIDPrefix = "switch_partial";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.25 0.75];
phase.cases.faultDurations = 0.20;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 487e-6;
phase.faultList = switchFaultRows( ...
    base, "partial_open", {[0.25 0.50 0.75]; [0.25 0.50 0.75]});
collect_fault_dataset(phase);

% 5) 间歇开路：40 ms 周期内形成较长的故障/恢复突发段。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase6_switch_intermittent");
phase.cases.runIDPrefix = "switch_intermittent";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.25 0.60];
phase.cases.faultDurations = 0.30;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 0.040;
phase.faultList = switchFaultRows( ...
    base, "intermittent", {[0.35 0.65]; [0.35 0.65]});
collect_fault_dataset(phase);

% 6) 高导通电阻：器件 Ron 为编译期参数，因此从 t=0 起施加。
phase = common;
phase.output.root = fullfile(datasetRoot, "phase7_switch_high_resistance");
phase.cases.runIDPrefix = "switch_high_r";
phase.cases.repetitions = 1;
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
phase.faultList = switchFaultRows( ...
    base, "high_resistance", {[0.01 0.03 0.10]; [0.01 0.03 0.10]});
collect_fault_dataset(phase);

function rows = switchFaultRows(base, suffix, magnitudes)
rows = base.faultList(ismember(base.faultList.FaultID, [3 4]), :);
rows.FaultName = ["switch_S1_" + suffix; "switch_S2_" + suffix];
rows.Magnitudes = magnitudes;
end
