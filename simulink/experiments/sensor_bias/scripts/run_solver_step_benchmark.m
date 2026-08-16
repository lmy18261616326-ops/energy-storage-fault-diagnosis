%% 1 us 与 5 us 固定步长精度/速度基准
% 分别运行健康与 S1 完全开路；结果用于决定大规模扩充是否可用 5 us。

scriptFolder = string(fileparts(mfilename("fullpath")));
benchmarkRoot = fullfile(scriptFolder, "dataset_output_v13", ...
    "solver_step_benchmark");
base = signal_config();
steps = [1e-6 5e-6];
labels = ["1us" "5us"];

for stepIndex = 1:numel(steps)
    common = struct();
    common.output.overwritePolicy = "resume";
    common.output.saveRunCSV = false;
    common.output.saveCombinedCSV = true;
    common.execution.useParallel = false;
    common.execution.fixedStep = steps(stepIndex);
    common.cases.operatingPointIDs = "op_0009";
    common.cases.repetitions = 1;
    common.cases.stopTime = 0.40;
    common.cases.randomSeedBase = 900001;
    common.cases.runIDPrefix = "benchmark_" + labels(stepIndex);
    common.cases.domainRandomization.enabled = false;

    health = common;
    health.output.root = fullfile(benchmarkRoot, ...
        "health_" + labels(stepIndex));
    health.faultList = base.faultList(base.faultList.FaultID == 0,:);
    collect_fault_dataset(health);

    fault = common;
    fault.output.root = fullfile(benchmarkRoot, ...
        "s1_open_" + labels(stepIndex));
    fault.cases.randomizeFaultStart = false;
    fault.cases.faultStartTimes = 0.20;
    fault.cases.faultDurations = 0.15;
    fault.adapter.switchFaultMechanism = "gate_blocking";
    fault.adapter.switchFaultPeriod = 5e-4;
    fault.faultList = base.faultList(base.faultList.FaultID == 3,:);
    collect_fault_dataset(fault);
end

summaryRows = struct([]);
caseNames = ["health" "s1_open"];
signals = ["IL_meas" "Ibat_meas" "Vbus_meas" "Vbat_meas" ...
    "CurrentError" "DutyApplied"];
for caseIndex = 1:numel(caseNames)
    reference = loadFirstRaw(benchmarkRoot, caseNames(caseIndex) + "_1us");
    candidate = loadFirstRaw(benchmarkRoot, caseNames(caseIndex) + "_5us");
    row = struct();
    row.Case = caseNames(caseIndex);
    for signal = signals
        truth = reference.(signal);
        estimate = candidate.(signal);
        rmse = sqrt(mean((estimate - truth).^2, "omitnan"));
        signalRMS = sqrt(mean(truth.^2, "omitnan"));
        scale = max(max(truth) - min(truth), signalRMS);
        row.(signal + "_NRMSE") = rmse / max(scale, eps);
    end
    row.FaultLabelAgreement = mean( ...
        reference.ObservableFaultID == candidate.ObservableFaultID);
    summaryRows(caseIndex,1) = row; %#ok<SAGROW>
end
summary = struct2table(summaryRows, "AsArray", true);
writetable(summary, fullfile(benchmarkRoot, "solver_step_comparison.csv"));
disp(summary);

function rawTable = loadFirstRaw(root, phase)
files = dir(fullfile(root, phase, "raw_runs", "*.mat"));
if numel(files) ~= 1
    error("faultdataset:BenchmarkRunCount", ...
        "%s 应恰好包含一个 raw run，实际为 %d。", phase, numel(files));
end
content = load(fullfile(files(1).folder, files(1).name), "rawTable");
rawTable = content.rawTable;
end
