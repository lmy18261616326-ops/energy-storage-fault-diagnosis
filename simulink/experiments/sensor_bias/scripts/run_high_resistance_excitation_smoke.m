%% Active-switch high-resistance excitation smoke dataset.
% The physical switch Ron remains a compile-time parameter.  A controlled
% 600 W load step at 0.35 s provides a repeatable excitation so the healthy
% and high-R conduction responses can be compared without editing the model.

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v15", ...
    "high_resistance_excitation_smoke");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", "high_r_excitation_smoke");
base = signal_config();

common = struct();
common.output.overwritePolicy = "resume";
common.output.saveRunCSV = false;
common.output.saveCombinedCSV = true;
common.output.temp = fullfile(shortWorkRoot, "tmp");
common.output.cache = fullfile(shortWorkRoot, "cache");
common.output.codegen = fullfile(shortWorkRoot, "codegen");
common.output.parallelJobs = fullfile(shortWorkRoot, "jobs");
common.execution.useParallel = false;
common.execution.fixedStep = 1e-6;
common.cases.socInit = 50;
common.cases.irefLevels = 20;
common.cases.vbusRef = 400;
common.cases.vbatInit = NaN;
common.cases.rload = 200;
common.cases.pload = 600;
common.cases.repetitions = 1;
common.cases.stopTime = 1.0;
common.cases.maxRunCount = 10;
common.cases.randomSeedBase = 815001;
common.cases.domainRandomization.enabled = false;
common.adapter.loadStepTime = 0.35;

runHealthy(common, base, datasetRoot, 1, "mode1_health", "high_r_x_health_m1");
runHighResistance(common, base, datasetRoot, 1, 3, ...
    "switch_S1_high_resistance", "mode1_s1_high_r", "high_r_x_s1", 0.10);
runHealthy(common, base, datasetRoot, 2, "mode2_health", "high_r_x_health_m2");
runHighResistance(common, base, datasetRoot, 2, 4, ...
    "switch_S2_high_resistance", "mode2_s2_high_r", "high_r_x_s2", 0.10);

fprintf("High-resistance excitation smoke data written to %s\n", datasetRoot);

function runHealthy(common, base, datasetRoot, modeCommand, folder, prefix)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("high_r_x_m%d_op", modeCommand);
phase.faultList = base.faultList(base.faultList.FaultID == 0, :);
collect_fault_dataset(phase);
end

function runHighResistance(common, base, datasetRoot, modeCommand, faultID, ...
        faultName, folder, prefix, magnitude)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("high_r_x_m%d_op", modeCommand);
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
rows = base.faultList(base.faultList.FaultID == faultID, :);
rows.FaultName = string(faultName);
rows.Magnitudes = {magnitude};
phase.faultList = rows;
collect_fault_dataset(phase);
end
