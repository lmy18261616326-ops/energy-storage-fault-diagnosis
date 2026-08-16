%% Two-run cross-mapping smoke test for switch high resistance.
% Checks whether Ron is observable on the opposite switch/mode pairing.

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v16", ...
    "switch_observability_cross_smoke");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", ...
    "switch_observability_cross_smoke");
base = signal_config();

common = struct();
common.modelName = "main_model_fd_v06_switchobservability";
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
common.cases.maxRunCount = 2;
common.cases.randomSeedBase = 816101;
common.cases.domainRandomization.enabled = false;
common.cases.randomizeFaultStart = false;
common.cases.faultStartTimes = 0;
common.cases.faultDurations = Inf;
common.adapter.loadStepTime = 0.35;
common.adapter.switchFaultMechanism = "high_resistance";

runHighResistance(common, base, datasetRoot, 1, 4, ...
    "switch_S2_high_resistance", "mode1_s2_high_r", ...
    "switch_cross_s2", 0.05);
runHighResistance(common, base, datasetRoot, 2, 3, ...
    "switch_S1_high_resistance", "mode2_s1_high_r", ...
    "switch_cross_s1", 0.05);

fprintf("Switch-observability cross smoke written to %s\n", datasetRoot);

function runHighResistance(common, base, datasetRoot, modeCommand, faultID, ...
        faultName, folder, prefix, magnitude)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("switch_cross_m%d_op", modeCommand);
rows = base.faultList(base.faultList.FaultID == faultID, :);
rows.FaultName = string(faultName);
rows.Magnitudes = {magnitude};
phase.faultList = rows;
collect_fault_dataset(phase);
end
