%% 部分/连续开路与高导通电阻的最小冒烟测试

scriptFolder = string(fileparts(mfilename("fullpath")));
root = fullfile(scriptFolder, "dataset_output_v13", "switch_fault_smoke");
base = signal_config();
s1 = base.faultList(base.faultList.FaultID == 3,:);
s2 = base.faultList(base.faultList.FaultID == 4,:);

common = struct();
common.output.overwritePolicy = "resume";
common.output.saveCombinedCSV = true;
common.execution.useParallel = false;
common.execution.fixedStep = 1e-6;
common.cases.operatingPointIDs = "op_0009";
common.cases.repetitions = 1;
common.cases.stopTime = 0.40;
common.cases.randomizeFaultStart = false;
common.cases.faultStartTimes = 0.20;
common.cases.faultDurations = 0.15;
common.cases.randomSeedBase = 910001;
common.cases.domainRandomization.enabled = false;
common.labeling.observableRatioThreshold = 0.20;

gate = common;
gate.output.root = fullfile(root, "gate_blocking");
gate.cases.runIDPrefix = "smoke_gate";
gate.adapter.switchFaultMechanism = "gate_blocking";
gate.adapter.switchFaultPeriod = 487e-6;
gate.faultList = [s1; s2];
gate.faultList.FaultName = [ ...
    "switch_S1_gate_smoke"; "switch_S2_gate_smoke"];
gate.faultList.Magnitudes = { ...
    [0.25 0.50 1.00]; [0.25 0.50 1.00]};
collect_fault_dataset(gate);

highR = common;
highR.output.root = fullfile(root, "high_resistance");
highR.cases.runIDPrefix = "smoke_high_r";
highR.cases.faultStartTimes = 0;
highR.cases.faultDurations = Inf;
highR.adapter.switchFaultMechanism = "high_resistance";
highR.faultList = [s1; s2];
highR.faultList.FaultName = [ ...
    "switch_S1_high_resistance_smoke"; ...
    "switch_S2_high_resistance_smoke"];
highR.faultList.Magnitudes = {0.03; 0.03};
collect_fault_dataset(highR);
