%% Development-domain bridge dataset (kept separate from the final blind set)
% Adds intermediate operating conditions between the original 12 corner
% points. The blind configuration is neither read nor modified here.

scriptFolder = string(fileparts(mfilename("fullpath")));
datasetRoot = fullfile(scriptFolder, "dataset_output_v13");
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
shortWorkRoot = fullfile(projectRoot, ".simtmp", "domain_bridge");
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
% Use a single worker for the bridge expansion.  A larger local pool caused
% memory-allocation and transient MEX-load failures on this computer.
common.execution.numWorkers = 1;
common.execution.parallelBatchSize = 2;
common.execution.showSimulationManager = false;
common.execution.fixedStep = 1e-6;
common.cases.randomSeedBase = 730001;
common.cases.domainRandomization.enabled = true;
common.cases.domainRandomization.RbatRelativeRange = 0.25;
common.cases.domainRandomization.CbusRelativeRange = 0.25;
common.cases.domainRandomization.CbusESRRelativeRange = 0.60;
common.cases.maxRunCount = 300;

% Sixteen intermediate charge/discharge operating points:
% 2 modes x 2 SOCs x 2 current commands x 2 load levels.
common.cases.modeCommands = [1 2];
common.cases.socInit = [43 57];
common.cases.irefLevels = [8.5 11.5];
common.cases.vbusRef = 400;
common.cases.vbatInit = NaN;
common.cases.rload = 200;
common.cases.pload = [140 260];
common.cases.operatingPointPrefix = "bridge_op";

% 1) Healthy bridge coverage: three independent randomized plants per OP.
phase = common;
phase.output.root = fullfile(datasetRoot, "phase8_bridge_health");
phase.cases.runIDPrefix = "bridge_health";
phase.cases.repetitions = 3;
phase.faultList = base.faultList(base.faultList.FaultID == 0,:);
collect_fault_dataset(phase);

% 2) Sensor faults across all bridge operating points.
phase = common;
phase.output.root = fullfile(datasetRoot, "phase9_bridge_sensor_faults");
phase.cases.runIDPrefix = "bridge_sensor";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.faultList = base.faultList( ...
    ismember(base.faultList.FaultID, [1 2]), :);
phase.faultList.Magnitudes = {[-5 5]; [-0.5 0.5]};
collect_fault_dataset(phase);

% 3) Full-open S1/S2 faults at randomized occurrence times.
phase = common;
phase.output.root = fullfile(datasetRoot, "phase10_bridge_switch_full_open");
phase.cases.runIDPrefix = "bridge_full";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 487e-6;
phase.faultList = switchFaultRows(base, "full_open", {1; 1});
collect_fault_dataset(phase);

% 4) Fifty-percent partial-open S1/S2 faults.
phase = common;
phase.output.root = fullfile(datasetRoot, "phase11_bridge_switch_partial_open");
phase.cases.runIDPrefix = "bridge_partial";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.70];
phase.cases.faultDurations = 0.20;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 487e-6;
phase.faultList = switchFaultRows(base, "partial_open", {0.50; 0.50});
collect_fault_dataset(phase);

% 5) Intermittent S1/S2 faults.
phase = common;
phase.output.root = fullfile(datasetRoot, ...
    "phase12_bridge_switch_intermittent");
phase.cases.runIDPrefix = "bridge_intermittent";
phase.cases.repetitions = 1;
phase.cases.faultStartRange = [0.30 0.60];
phase.cases.faultDurations = 0.30;
phase.labeling.observableRatioThreshold = 0.20;
phase.adapter.switchFaultMechanism = "gate_blocking";
phase.adapter.switchFaultPeriod = 0.040;
phase.faultList = switchFaultRows(base, "intermittent", {0.50; 0.50});
collect_fault_dataset(phase);

% 6) High-resistance S1/S2 faults, active from simulation start.
phase = common;
phase.output.root = fullfile(datasetRoot, ...
    "phase13_bridge_switch_high_resistance");
phase.cases.runIDPrefix = "bridge_high_r";
phase.cases.repetitions = 1;
phase.cases.randomizeFaultStart = false;
phase.cases.faultStartTimes = 0;
phase.cases.faultDurations = Inf;
phase.adapter.switchFaultMechanism = "high_resistance";
phase.faultList = switchFaultRows( ...
    base, "high_resistance", {0.05; 0.05});
collect_fault_dataset(phase);

function rows = switchFaultRows(base, suffix, magnitudes)
rows = base.faultList(ismember(base.faultList.FaultID, [3 4]), :);
rows.FaultName = ["switch_S1_" + suffix; "switch_S2_" + suffix];
rows.Magnitudes = magnitudes;
end
