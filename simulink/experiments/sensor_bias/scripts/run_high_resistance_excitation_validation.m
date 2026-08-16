%% Independent validation for the commissioned high-resistance specialist.
% Threshold 0.60 is frozen from the pilot OOF frontier.  These runs use new
% plant-randomization seeds and an unseen Ron=0.05 ohm magnitude.

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v15", ...
    "high_resistance_excitation_validation");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", "high_r_excitation_validation");
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
common.cases.maxRunCount = 10;
common.cases.randomSeedBase = 817001;
common.cases.domainRandomization.enabled = true;
common.cases.domainRandomization.RbatRelativeRange = 0.10;
common.cases.domainRandomization.CbusRelativeRange = 0.10;
common.cases.domainRandomization.CbusESRRelativeRange = 0.20;
common.adapter.loadStepTime = 0.35;

runHealthy(common, base, datasetRoot, 1, "mode1_health", "high_r_val_health_m1");
runHighResistance(common, base, datasetRoot, 1, 3, ...
    "switch_S1_high_resistance", "mode1_s1_high_r", "high_r_val_s1");
runHealthy(common, base, datasetRoot, 2, "mode2_health", "high_r_val_health_m2");
runHighResistance(common, base, datasetRoot, 2, 4, ...
    "switch_S2_high_resistance", "mode2_s2_high_r", "high_r_val_s2");

fprintf("High-resistance independent validation written to %s\n", datasetRoot);

function runHealthy(common, base, datasetRoot, modeCommand, folder, prefix)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("high_r_p_m%d_op", modeCommand);
phase.faultList = base.faultList(base.faultList.FaultID == 0, :);
collect_fault_dataset(phase);
end

function runHighResistance(common, base, datasetRoot, modeCommand, faultID, ...
        faultName, folder, prefix)
phase = common;
phase.output.root = fullfile(datasetRoot, folder);
phase.cases.runIDPrefix = prefix;
phase.cases.modeCommands = modeCommand;
phase.cases.operatingPointPrefix = compose("high_r_p_m%d_op", modeCommand);
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
rows = base.faultList(base.faultList.FaultID == faultID, :);
rows.FaultName = string(faultName);
rows.Magnitudes = {0.05};
phase.faultList = rows;
collect_fault_dataset(phase);
end
