%% 修复后目标工况健康样本扩充
% 为历史误报最严重的三个工况生成独立健康 RunID。每个 Run 使用不同
% 随机种子，并对电池内阻、母线电容和电容 ESR 做更宽的域随机化。

scriptFolder = string(fileparts(mfilename("fullpath")));
outputRoot = fullfile(scriptFolder, "dataset_output_v13", ...
    "phase1_target_health");

base = signal_config();
healthyFault = base.faultList(base.faultList.FaultID == 0, :);

overrides = struct();
overrides.output.root = outputRoot;
overrides.output.overwritePolicy = "resume";
overrides.output.saveRunCSV = false;
overrides.output.saveCombinedCSV = true;
overrides.execution.useParallel = true;
overrides.execution.numWorkers = 2;
overrides.execution.parallelBatchSize = 12;
overrides.execution.showSimulationManager = false;
overrides.cases.operatingPointIDs = ["op_0002"; "op_0006"; "op_0009"];
overrides.cases.runIDPrefix = "health_target";
overrides.cases.repetitions = 16;
overrides.cases.randomSeedBase = 510001;
overrides.cases.maxRunCount = 100;
overrides.cases.domainRandomization.enabled = true;
overrides.cases.domainRandomization.RbatRelativeRange = 0.20;
overrides.cases.domainRandomization.CbusRelativeRange = 0.20;
overrides.cases.domainRandomization.CbusESRRelativeRange = 0.50;
overrides.faultList = healthyFault;

result = collect_fault_dataset(overrides); %#ok<NASGU>
