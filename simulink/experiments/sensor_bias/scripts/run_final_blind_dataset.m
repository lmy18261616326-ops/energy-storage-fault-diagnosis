%% 最终盲测数据：不得参与特征选择、调参、校准或阈值选择
% 使用全新的 SOC、负载、母线参考和电阻负载组合，并采用开发集中未出现
% 的故障严重度。仅在开发方案冻结后运行和评估。

scriptFolder = string(fileparts(mfilename("fullpath")));
blindRoot = fullfile(scriptFolder, "dataset_output_v13", "blind_test");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", "blind");
base = signal_config();

common = struct();
common.output.overwritePolicy = "resume";
common.output.saveRunCSV = false;
common.output.saveCombinedCSV = true;
common.output.temp = fullfile(shortWorkRoot, "tmp");
common.output.cache = fullfile(shortWorkRoot, "cache");
common.output.codegen = fullfile(shortWorkRoot, "codegen");
common.output.parallelJobs = fullfile(shortWorkRoot, "jobs");
common.execution.useParallel = true;
common.execution.numWorkers = 2;
common.execution.parallelBatchSize = 6;
common.execution.showSimulationManager = false;
common.execution.fixedStep = 1e-6;
common.cases.modeCommands = [0 1 2];
common.cases.socInit = [45 60];
common.cases.irefLevels = 10;
common.cases.vbusRef = 410;
common.cases.rload = 160;
common.cases.pload = [200 600];
common.cases.operatingPointPrefix = "blind_op";
common.cases.randomSeedBase = 810001;
common.cases.domainRandomization.enabled = true;
common.cases.domainRandomization.RbatRelativeRange = 0.15;
common.cases.domainRandomization.CbusRelativeRange = 0.15;
common.cases.domainRandomization.CbusESRRelativeRange = 0.35;
common.cases.maxRunCount = 500;

phase = common;
phase.output.root = fullfile(blindRoot, "health");
phase.cases.runIDPrefix = "blind_health";
phase.cases.repetitions = 2;
phase.faultList = base.faultList(base.faultList.FaultID == 0,:);
collect_fault_dataset(phase);

phase = common;
phase.output.root = fullfile(blindRoot, "sensor_faults");
phase.cases.runIDPrefix = "blind_sensor";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.faultList = base.faultList( ...
    ismember(base.faultList.FaultID, [1 2]), :);
phase.faultList.Magnitudes = {[-7 7]; [-0.75 0.75]};
collect_fault_dataset(phase);

phase = common;
phase.output.root = fullfile(blindRoot, "switch_full_open");
phase.cases.runIDPrefix = "blind_switch_full";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 757e-6;
phase.faultList = switchFaultRows(base, "full_open", {1; 1});
collect_fault_dataset(phase);

phase = common;
phase.output.root = fullfile(blindRoot, "switch_partial_open");
phase.cases.runIDPrefix = "blind_switch_partial";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 757e-6;
phase.faultList = switchFaultRows( ...
    base, "partial_open", {[0.40 0.60]; [0.40 0.60]});
collect_fault_dataset(phase);

phase = common;
phase.output.root = fullfile(blindRoot, "switch_intermittent");
phase.cases.runIDPrefix = "blind_switch_intermittent";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.55];
phase.cases.faultDurations = 0.35;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 0.055;
phase.faultList = switchFaultRows(base, "intermittent", {0.50; 0.50});
collect_fault_dataset(phase);

phase = common;
phase.output.root = fullfile(blindRoot, "switch_high_resistance");
phase.cases.runIDPrefix = "blind_switch_high_r";
phase.cases.repetitions = 1;
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
phase.faultList = switchFaultRows( ...
    base, "high_resistance", {[0.02 0.06]; [0.02 0.06]});
collect_fault_dataset(phase);

function rows = switchFaultRows(base, suffix, magnitudes)
rows = base.faultList(ismember(base.faultList.FaultID, [3 4]), :);
rows.FaultName = ["switch_S1_" + suffix; "switch_S2_" + suffix];
rows.Magnitudes = magnitudes;
end
