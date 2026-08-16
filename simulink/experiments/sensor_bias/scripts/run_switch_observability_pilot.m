%% Multi-operating-point development data for the v06 high-R specialist.
% 32 runs: four operating points per active mode, matched healthy repeats,
% and two fault magnitudes (0.02 and 0.10 ohm).

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v16", ...
    "switch_observability_pilot");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", ...
    "switch_observability_pilot");
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
common.execution.useParallel = true;
common.execution.numWorkers = 2;
common.execution.parallelBatchSize = 4;
common.execution.fixedStep = 1e-6;
common.execution.showSimulationManager = false;
common.cases.socInit = 50;
common.cases.irefLevels = 20;
common.cases.vbusRef = [400 410];
common.cases.vbatInit = NaN;
common.cases.rload = [160 200];
common.cases.pload = 600;
common.cases.repetitions = 1;
common.cases.stopTime = 1.0;
common.cases.maxRunCount = 20;
common.cases.randomSeedBase = 816501;
common.cases.domainRandomization.enabled = true;
common.cases.domainRandomization.RbatRelativeRange = 0.10;
common.cases.domainRandomization.CbusRelativeRange = 0.10;
common.cases.domainRandomization.CbusESRRelativeRange = 0.20;
common.adapter.loadStepTime = 0.35;

runHealthy(common, base, datasetRoot, 1, ...
    "mode1_health", "switch_obs_p_health_m1", 2);
runHighResistance(common, base, datasetRoot, 1, 3, ...
    "switch_S1_high_resistance", "mode1_s1_high_r", ...
    "switch_obs_p_s1");
runHealthy(common, base, datasetRoot, 2, ...
    "mode2_health", "switch_obs_p_health_m2", 2);
runHighResistance(common, base, datasetRoot, 2, 4, ...
    "switch_S2_high_resistance", "mode2_s2_high_r", ...
    "switch_obs_p_s2");

fprintf("Switch-observability pilot written to %s\n", datasetRoot);

function runHealthy(common, base, datasetRoot, modeCommand, ...
        folder, prefix, repetitions)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("switch_obs_p_m%d_op", modeCommand);
phase.cases.repetitions = repetitions;
phase.faultList = base.faultList(base.faultList.FaultID == 0, :);
collect_fault_dataset(phase);
end

function runHighResistance(common, base, datasetRoot, modeCommand, faultID, ...
        faultName, folder, prefix)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("switch_obs_p_m%d_op", modeCommand);
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
rows = base.faultList(base.faultList.FaultID == faultID, :);
rows.FaultName = string(faultName);
rows.Magnitudes = {[0.02 0.10]};
phase.faultList = rows;
collect_fault_dataset(phase);
end
