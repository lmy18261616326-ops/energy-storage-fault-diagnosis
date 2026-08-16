function result = collect_fault_dataset(overrides)
%COLLECT_FAULT_DATASET 批量运行 Simulink 并建立故障诊断数据集。
%   RESULT = COLLECT_FAULT_DATASET() 生成工况、构造
%   Simulink.SimulationInput、逐次仿真、立即保存单次原始数据，最后合并
%   原始表、提取窗口特征、执行质量检查并生成图形。
%
%   RESULT = COLLECT_FAULT_DATASET(OVERRIDES) 用结构体覆盖 signal_config
%   的默认设置。例如：
%       overrides.cases.socInit = [20 50 80];
%       overrides.execution.useParallel = true;
%       result = collect_fault_dataset(overrides);
%
%   输入:
%       overrides - 可选配置覆盖结构体。
%   输出:
%       result    - 输出路径、成功/失败次数和质量报告摘要。

arguments
    overrides (1,1) struct = struct()
end

cfg = signal_config(overrides);
addpath(fileparts(cfg.modelFile));
ensureOutputFolders(cfg);
fileGenerationCleanup = configureFileGeneration(cfg); %#ok<NASGU>
checkOverwritePolicy(cfg);
[toWorkspaceBlocks, scopeBlocks] = findLoggingBlocks(cfg);

cases = build_simulation_cases(cfg);
totalRuns = height(cases);
writetable(cases, fullfile(cfg.output.combined, "simulation_cases.csv"));
save(fullfile(cfg.output.combined, "simulation_cases.mat"), ...
    "cases", "cfg", "-v7.3");

fprintf("模型: %s\n", cfg.modelName);
fprintf("计划运行: %d 次；执行方式: %s\n", totalRuns, ...
    chooseText(cfg.execution.useParallel, "parsim", "serial sim"));
fprintf("输出目录: %s\n", cfg.output.root);

failedRows = struct([]);
failedCount = 0;
successRunIDs = strings(0,1);

if cfg.execution.useParallel
    batchSize = cfg.execution.parallelBatchSize;
    if ~isscalar(batchSize) || batchSize < 1 || fix(batchSize) ~= batchSize
        error("faultdataset:InvalidParallelBatchSize", ...
            "execution.parallelBatchSize 必须是正整数。");
    end

    pendingIndices = zeros(1, 0);
    for runIndex = 1:totalRuns
        runFile = fullfile(cfg.output.rawRuns, ...
            cases.RunID(runIndex) + ".mat");
        if cfg.output.overwritePolicy == "resume" && ...
                isReusableRunFile(runFile, cfg)
            fprintf("[%d/%d] RunID=%s 已存在，按 resume 跳过仿真。\n", ...
                runIndex, totalRuns, cases.RunID(runIndex));
            successRunIDs(end+1,1) = cases.RunID(runIndex); %#ok<AGROW>
        else
            if cfg.output.overwritePolicy == "resume" && isfile(runFile)
                warning("faultdataset:StaleRunFile", ...
                    "%s 使用旧数据结构，将重新仿真并覆盖该单次运行。", ...
                    runFile);
            end
            pendingIndices(end+1) = runIndex; %#ok<AGROW>
        end
    end

    if ~isempty(pendingIndices)
        configureParallelPool(cfg);
    end
    for batchStart = 1:batchSize:numel(pendingIndices)
        batchIndices = pendingIndices( ...
            batchStart:min(batchStart + batchSize - 1, ...
            numel(pendingIndices)));
        batchCount = numel(batchIndices);
        inputs(1,batchCount) = Simulink.SimulationInput( ...
            cfg.modelName); %#ok<AGROW>
        for localIndex = 1:batchCount
            runIndex = batchIndices(localIndex);
            inputs(localIndex) = buildSimulationInput( ...
                cases(runIndex,:), cfg, toWorkspaceBlocks, scopeBlocks);
        end
        outputs = parsim(inputs, ...
            "ShowProgress", "on", ...
            "ShowSimulationManager", ...
            chooseText(cfg.execution.showSimulationManager, "on", "off"));
        for localIndex = 1:batchCount
            runIndex = batchIndices(localIndex);
            [success, failure] = processOneOutput( ...
                outputs(localIndex), cases(runIndex,:), ...
                runIndex, totalRuns, cfg);
            if success
                successRunIDs(end+1,1) = cases.RunID(runIndex); %#ok<AGROW>
            else
                failedCount = failedCount + 1;
                failedRows = appendFailure(failedRows, failure);
            end
        end
        clear inputs outputs
    end
else
    for runIndex = 1:totalRuns
        runCase = cases(runIndex,:);
        runFile = fullfile(cfg.output.rawRuns, runCase.RunID + ".mat");
        if cfg.output.overwritePolicy == "resume" && ...
                isReusableRunFile(runFile, cfg)
            fprintf("[%d/%d] RunID=%s 已存在，按 resume 跳过仿真。\n", ...
                runIndex, totalRuns, runCase.RunID);
            successRunIDs(end+1,1) = runCase.RunID; %#ok<AGROW>
            continue;
        elseif cfg.output.overwritePolicy == "resume" && isfile(runFile)
            warning("faultdataset:StaleRunFile", ...
                "%s 使用旧数据结构，将重新仿真并覆盖该单次运行。", runFile);
        end
        try
            input = buildSimulationInput( ...
                runCase, cfg, toWorkspaceBlocks, scopeBlocks);
            output = sim(input);
            [success, failure] = processOneOutput( ...
                output, runCase, runIndex, totalRuns, cfg);
        catch exception
            success = false;
            failure = makeFailureRow(runCase, exception);
            printProgress(runCase, runIndex, totalRuns, false);
            fprintf("  错误:\n%s\n", getReport( ...
                exception, "extended", "hyperlinks", "off"));
        end
        if success
            successRunIDs(end+1,1) = runCase.RunID; %#ok<AGROW>
        else
            failedCount = failedCount + 1;
            failedRows = appendFailure(failedRows, failure);
        end
    end
end

failedRuns = failureStructToTable(failedRows);
writetable(failedRuns, fullfile(cfg.output.combined, "failed_runs.csv"));
save(fullfile(cfg.output.combined, "failed_runs.mat"), "failedRuns");

[rawDataset, loadedRunIDs] = combineRawRuns(cases, cfg);
if isempty(rawDataset)
    error("faultdataset:NoSuccessfulRuns", ...
        "没有可合并的成功运行，请检查 failed_runs.csv。");
end

rawMat = fullfile(cfg.output.combined, "raw_dataset.mat");
save(rawMat, "rawDataset", "-v7.3");
if cfg.output.saveCombinedCSV
    writetable(rawDataset, ...
        fullfile(cfg.output.combined, "raw_dataset.csv"));
end

[featureDataset, featureColumns, labelColumn, ...
    metadataColumns, excludedColumns] = ...
    extract_window_features(rawDataset, cfg);
featureMat = fullfile(cfg.output.combined, "feature_dataset.mat");
save(featureMat, "featureDataset", "-v7.3");
if cfg.output.saveCombinedCSV
    writetable(featureDataset, ...
        fullfile(cfg.output.combined, "feature_dataset.csv"));
end
save(fullfile(cfg.output.combined, "dataset_field_groups.mat"), ...
    "featureColumns", "labelColumn", "metadataColumns", ...
    "excludedColumns");

report = validate_dataset( ...
    rawDataset, featureDataset, cases, failedRuns, cfg, featureColumns);

result = struct();
result.Config = cfg;
result.TotalRuns = totalRuns;
result.SuccessCount = numel(unique(loadedRunIDs));
result.FailureCount = height(failedRuns);
result.RawDatasetFile = rawMat;
result.FeatureDatasetFile = featureMat;
result.ReportFile = fullfile(cfg.output.combined, "dataset_report.txt");
result.FeatureColumns = featureColumns;
result.LabelColumn = labelColumn;
result.Report = report;

fprintf("\n采集完成：成功 %d，失败 %d。\n", ...
    result.SuccessCount, result.FailureCount);
fprintf("原始数据: %s\n特征数据: %s\n质量报告: %s\n", ...
    result.RawDatasetFile, result.FeatureDatasetFile, result.ReportFile);
end

function input = buildSimulationInput( ...
        runCase, cfg, toWorkspaceBlocks, scopeBlocks)
input = Simulink.SimulationInput(cfg.modelName);
input = input.setModelParameter( ...
    "StopTime", string(runCase.SimulationStopTime), ...
    "ReturnWorkspaceOutputs", "on", ...
    "SignalLogging", "on", ...
    "SaveState", chooseText(cfg.execution.saveModelStates, "on", "off"), ...
    "StateSaveName", "xout", ...
    "SaveOutput", "off", ...
    "SaveTime", "off", ...
    "DSMLogging", "off", ...
    "StreamToWks", "off", ...
    "InspectSignalLogs", "off", ...
    "VisualizeSimOutput", "off", ...
    "SaveFormat", "Dataset");
if isfinite(cfg.execution.fixedStep)
    input = input.setModelParameter( ...
        "SolverType", "Fixed-step", ...
        "FixedStep", string(cfg.execution.fixedStep));
    input = input.setBlockParameter( ...
        Simulink.ID.getFullName( ...
            cfg.modelName + ":" + cfg.adapter.powerguiSID), ...
        "SampleTime", string(cfg.execution.fixedStep));
    input = setModelVariable( ...
        input, cfg, "FD_TS", cfg.execution.fixedStep);
end

% 模型有大量继承 1 us 步长的 To Workspace 模块。最终数据只需要
% cfg.sampleTime，因此直接按该采样周期写入 SimulationOutput，避免先
% 生成约 50 倍的高频数据再由 extract_raw_signals 降采样。
for block = toWorkspaceBlocks
    loggingSampleTime = cfg.loggingSampleTime;
    variableName = string(get_param(block, "VariableName"));
    if ismember(variableName, cfg.switchMeasurement.variableNames)
        loggingSampleTime = cfg.switchMeasurement.loggingSampleTime;
    end
    input = input.setBlockParameter( ...
        block, "SampleTime", string(loggingSampleTime));
end

% 这些 Scope 仅用于交互查看，不参与数据集字段提取。
for block = scopeBlocks
    input = input.setBlockParameter(block, "DataLogging", "off");
end

if cfg.execution.useFastRestart
    input = input.setModelParameter("FastRestart", "on");
end

% 同时传入统一的通用变量名，便于模型后续逐步改用标准接口。
genericNames = [ ...
    "ModeCommand","SOCInit","IrefLevel","VbusRef","VbatInit", ...
    "Rload","Pload","Rbat","Cbus","CbusESR", ...
    "FaultID","FaultMagnitude","FaultParameter1", ...
    "FaultParameter2","FaultStartTime","FaultEndTime","RandomSeed"];
for name = genericNames
    input = setModelVariable(input, cfg, name, runCase.(name));
end

% 工作模式和故障时序。
input = setModelVariable(input, cfg, "FD_MODE_OVERRIDE_ENABLE", 1);
input = setModelVariable(input, cfg, "FD_MODE_COMMAND", runCase.ModeCommand);
input = setModelVariable(input, cfg, "FD_PROTECTION_MODE", ...
    cfg.adapter.protectionMode);
input = setModelVariable(input, cfg, "FD_FAULT_TIME", ...
    runCase.FaultStartTime);
input = setModelVariable(input, cfg, "FD_FAULT_END_TIME", ...
    runCase.FaultEndTime);
input = setModelVariable(input, cfg, "FD_LOAD_STEP_TIME", ...
    cfg.adapter.loadStepTime);
input = setModelVariable(input, cfg, "FD_RBAT", runCase.Rbat);
input = setModelVariable(input, cfg, "FD_CBUS", runCase.Cbus);
input = setModelVariable(input, cfg, "FD_CBUS_ESR", runCase.CbusESR);
input = setModelVariable(input, cfg, "FD_SWITCH_FAULT_PERIOD", ...
    cfg.adapter.switchFaultPeriod);
input = setModelVariable(input, cfg, "FD_IREF_OVERRIDE_ENABLE", 1);
input = setModelVariable(input, cfg, "FD_IREF_LEVEL", ...
    abs(runCase.IrefLevel));
input = setModelVariable(input, cfg, "I_charge_max", ...
    cfg.control.chargeCurrentLimit);
input = setModelVariable(input, cfg, "I_discharge_cmd", ...
    cfg.control.dischargeCurrentLimit);

% 每个 Run 使用相同的传感器结构，只改变确定性随机种子。
input = setModelVariable(input, cfg, "FD_SENSOR_TS", ...
    cfg.sensor.sampleTime);
input = setModelVariable(input, cfg, "FD_VBUS_NOISE_STD", ...
    cfg.sensor.vbusNoiseStd);
input = setModelVariable(input, cfg, "FD_VBAT_NOISE_STD", ...
    cfg.sensor.vbatNoiseStd);
input = setModelVariable(input, cfg, "FD_IBAT_NOISE_STD", ...
    cfg.sensor.ibatNoiseStd);
input = setModelVariable(input, cfg, "FD_IL_NOISE_STD", ...
    cfg.sensor.ilNoiseStd);
input = setModelVariable(input, cfg, "FD_VBUS_QUANT_STEP", ...
    cfg.sensor.vbusQuantStep);
input = setModelVariable(input, cfg, "FD_VBAT_QUANT_STEP", ...
    cfg.sensor.vbatQuantStep);
input = setModelVariable(input, cfg, "FD_CURRENT_QUANT_STEP", ...
    cfg.sensor.currentQuantStep);

% 功率平衡残差的符号约定和初始储能由同一配置集中控制。
input = setModelVariable(input, cfg, "EP_PSOURCE_SIGN", ...
    cfg.power.sourceDirectionFactor);
input = setModelVariable(input, cfg, "EP_PBAT_SIGN", ...
    cfg.power.batteryDirectionFactor);
input = setModelVariable(input, cfg, "EP_PLOAD_SIGN", ...
    cfg.power.loadDirectionFactor);
input = setModelVariable(input, cfg, "EP_POWER_BALANCE_ALPHA", ...
    cfg.power.balanceFilterAlpha);
input = setModelVariable(input, cfg, "EP_ENERGY_INITIAL", ...
    0.5 .* runCase.Cbus .* runCase.VbusRef.^2);

% 所有故障入口先归零，健康和四种故障拥有相同输入结构。
faultVariables = cfg.faultList.FaultVariable;
faultVariables = unique(faultVariables(faultVariables ~= ""), "stable");
for variable = faultVariables'
    input = setModelVariable(input, cfg, variable, 0);
end
input = setModelVariable(input, cfg, "FD_VBAT_BIAS", 0);
input = setModelVariable(input, cfg, "FD_IBAT_BIAS", 0);
input = setModelVariable(input, cfg, "FD_DUTY_BIAS", 0);
input = setModelVariable(input, cfg, "FD_DUTY_STUCK_ENABLE", 0);
input = setModelVariable(input, cfg, "FD_DUTY_STUCK_VALUE", 0.5);

faultRow = cfg.faultList(cfg.faultList.FaultID == runCase.FaultID,:);
if height(faultRow) ~= 1
    error("faultdataset:UnknownFaultID", ...
        "FaultID=%g 未在 cfg.faultList 中唯一配置。", runCase.FaultID);
end
input = setModelVariable(input, cfg, "FD_FAULT_ID", ...
    faultRow.ConfiguredFaultID);
isSwitchFault = ismember(runCase.FaultID, cfg.labeling.switchFaultIDs);
isHighResistance = isSwitchFault && ...
    cfg.adapter.switchFaultMechanism == "high_resistance";
if faultRow.FaultVariable ~= "" && ~isHighResistance
    input = setModelVariable(input, cfg, faultRow.FaultVariable, ...
        runCase.FaultMagnitude);
end

% 高导通电阻使用器件真实 Ron 参数。该参数是编译期参数，因此当前实现
% 从 t=0 起生效；对应采集阶段会把 FaultStartTime 固定为 0。
s1Ron = cfg.adapter.switchRonNominal;
s2Ron = cfg.adapter.switchRonNominal;
if isHighResistance && runCase.FaultID == cfg.labeling.s1OpenFaultID
    s1Ron = runCase.FaultMagnitude;
elseif isHighResistance && runCase.FaultID == cfg.labeling.s2OpenFaultID
    s2Ron = runCase.FaultMagnitude;
end
input = input.setBlockParameter( ...
    Simulink.ID.getFullName( ...
        cfg.modelName + ":" + cfg.adapter.switchS1SID), ...
    cfg.adapter.switchRonParameter, ...
    num2str(s1Ron, 17));
input = input.setBlockParameter( ...
    Simulink.ID.getFullName( ...
        cfg.modelName + ":" + cfg.adapter.switchS2SID), ...
    cfg.adapter.switchRonParameter, ...
    num2str(s2Ron, 17));

% Pload 通过模型现有受控电流负载施加；Rload 使用母线电阻模块。
loadCurrent = runCase.Pload / max(abs(runCase.VbusRef), eps);
input = setModelVariable(input, cfg, "FD_LOAD_STEP_A", loadCurrent);
batterySOCParameter = runCase.SOCInit;
if cfg.adapter.correctBatterySOCDefinition
    [batterySOCParameter, batteryOpenCircuitVoltage] = ...
        requestedSOCToBatteryInitialConditions( ...
        runCase.SOCInit, cfg.adapter.batteryBlock);
else
    [~, batteryOpenCircuitVoltage] = ...
        requestedSOCToBatteryInitialConditions( ...
        runCase.SOCInit, cfg.adapter.batteryBlock, false);
end
input = input.setBlockParameter( ...
    cfg.adapter.batteryBlock, cfg.adapter.batterySOCParameter, ...
    num2str(batterySOCParameter, 17));
if cfg.adapter.synchronizeBatterySideCapacitor
    input = input.setBlockParameter( ...
        cfg.adapter.batterySideCapacitorBlock, ...
        cfg.adapter.batterySideCapacitorVoltageParameter, ...
        num2str(batteryOpenCircuitVoltage, 17));
end
input = input.setBlockParameter( ...
    cfg.adapter.loadResistanceBlock, ...
    cfg.adapter.loadResistanceParameter, num2str(runCase.Rload, 17));
input = input.setBlockParameter( ...
    cfg.adapter.vbusReferenceBlock, ...
    cfg.adapter.vbusReferenceParameter, num2str(runCase.VbusRef, 17));
end

function [parameterSOC, openCircuitVoltage] = ...
        requestedSOCToBatteryInitialConditions( ...
        requestedSOC, batteryBlock, correctDefinition)
% Battery 预设模型用额定容量初始化电荷量，却用有效容量计算测量口 SOC。
% 同时按相同电荷状态计算开路电压，供电池侧电容做一致性初始化。
arguments
    requestedSOC (1,1) double {mustBeFinite,mustBeInRange(requestedSOC,0,100)}
    batteryBlock (1,1) string
    correctDefinition (1,1) logical = true
end
persistent cachedBlock cachedBattery
if isempty(cachedBlock) || cachedBlock ~= string(batteryBlock)
    maskVariables = get_param(batteryBlock, "MaskWSVariables");
    names = string({maskVariables.Name});
    batteryIndex = find(names == "Batt", 1, "first");
    if isempty(batteryIndex) || ...
            ~isfield(maskVariables(batteryIndex).Value, "lambda")
        error("faultdataset:BatteryCapacityFactorUnavailable", ...
            "无法从 %s 读取 Battery 有效容量系数 Batt.lambda。", ...
            batteryBlock);
    end
    cachedBlock = string(batteryBlock);
    cachedBattery = maskVariables(batteryIndex).Value;
end

if correctDefinition
    parameterSOC = 100 .* ...
        (1 - cachedBattery.lambda .* (1 - requestedSOC ./ 100));
else
    parameterSOC = requestedSOC;
end
validateattributes(parameterSOC, {'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',100}, ...
    mfilename, "mapped Battery SOC");

initialCoulombCount = (1 - parameterSOC ./ 100) .* ...
    cachedBattery.Q ./ cachedBattery.lambda;
remainingCapacity = cachedBattery.Q - initialCoulombCount;
openCircuitVoltage = cachedBattery.E0 ...
    - cachedBattery.K .* cachedBattery.Q ./ remainingCapacity .* ...
    initialCoulombCount ...
    + cachedBattery.A .* exp( ...
    -cachedBattery.B .* initialCoulombCount);
validateattributes(openCircuitVoltage, {'numeric'}, ...
    {'scalar','real','finite','positive'}, ...
    mfilename, "Battery open-circuit voltage");
end

function [toWorkspaceBlocks, scopeBlocks] = findLoggingBlocks(cfg)
load_system(cfg.modelFile);
toWorkspaceBlocks = string(find_system(cfg.modelName, ...
    "SearchDepth", 1, "BlockType", "ToWorkspace"));
scopeBlocks = string(find_system(cfg.modelName, ...
    "SearchDepth", 1, "BlockType", "Scope"));
toWorkspaceBlocks = reshape(toWorkspaceBlocks, 1, []);
scopeBlocks = reshape(scopeBlocks, 1, []);
fprintf("精简记录: %d 个 To Workspace 按 %.6g s 采样，" + ...
    "%d 个 Scope 关闭数据记录。\n", ...
    numel(toWorkspaceBlocks), cfg.loggingSampleTime, numel(scopeBlocks));
end

function input = setModelVariable(input, cfg, name, value)
input = input.setVariable(name, value, ...
    "Workspace", cfg.adapter.modelWorkspace);
end

function [success, failure] = processOneOutput( ...
    output, runCase, runIndex, totalRuns, cfg)
success = false;
failure = struct([]);
try
    if strlength(string(output.ErrorMessage)) > 0
        error("faultdataset:SimulationFailed", "%s", output.ErrorMessage);
    end
    rawTable = extract_raw_signals(output, runCase, cfg);
    runFile = fullfile(cfg.output.rawRuns, runCase.RunID + ".mat");
    datasetSchemaVersion = cfg.version;
    save(runFile, "rawTable", "runCase", "datasetSchemaVersion", "-v7.3");
    if cfg.output.saveRunCSV
        writetable(rawTable, ...
            fullfile(cfg.output.rawRuns, runCase.RunID + ".csv"));
    end
    success = true;
    printProgress(runCase, runIndex, totalRuns, true);
catch exception
    failure = makeFailureRow(runCase, exception);
    printProgress(runCase, runIndex, totalRuns, false);
    fprintf("  错误: %s\n", exception.message);
end
end

function reusable = isReusableRunFile(runFile, cfg)
reusable = false;
try
    content = load(runFile, "datasetSchemaVersion", "rawTable");
    if ~isfield(content, "datasetSchemaVersion") || ...
            string(content.datasetSchemaVersion) ~= cfg.version || ...
            ~isfield(content, "rawTable")
        return;
    end
    required = [cfg.requiredRawFields, "FaultObservable", ...
        "ObservableFaultID", "IsTrainingEligible"];
    reusable = isempty(setdiff(required, ...
        string(content.rawTable.Properties.VariableNames)));
catch
    reusable = false;
end
end

function printProgress(runCase, current, total, success)
fprintf(['[%d/%d] RunID=%s FaultID=%g ModeCommand=%g ' ...
    'SOCInit=%g IrefLevel=%g 成功=%s\n'], ...
    current, total, runCase.RunID, runCase.FaultID, ...
    runCase.ModeCommand, runCase.SOCInit, runCase.IrefLevel, ...
    chooseText(success, "是", "否"));
end

function row = makeFailureRow(runCase, exception)
row = struct( ...
    "RunID", string(runCase.RunID), ...
    "OperatingPointID", string(runCase.OperatingPointID), ...
    "FaultID", runCase.FaultID, ...
    "FaultName", string(runCase.FaultName), ...
    "ModeCommand", runCase.ModeCommand, ...
    "SOCInit", runCase.SOCInit, ...
    "IrefLevel", runCase.IrefLevel, ...
    "ErrorIdentifier", string(exception.identifier), ...
    "ErrorMessage", string(getReport( ...
        exception, "extended", "hyperlinks", "off")), ...
    "ErrorTime", datetime("now"));
end

function rows = appendFailure(rows, failure)
if isempty(rows)
    rows = failure;
else
    failure = orderfields(failure, rows);
    rows(end+1,1) = failure;
end
end

function tableOut = failureStructToTable(rows)
if isempty(rows)
    tableOut = table( ...
        strings(0,1), strings(0,1), zeros(0,1), strings(0,1), ...
        zeros(0,1), zeros(0,1), zeros(0,1), strings(0,1), ...
        strings(0,1), NaT(0,1), ...
        'VariableNames', {'RunID','OperatingPointID','FaultID', ...
        'FaultName','ModeCommand','SOCInit','IrefLevel', ...
        'ErrorIdentifier','ErrorMessage','ErrorTime'});
else
    tableOut = struct2table(rows, "AsArray", true);
end
end

function [rawDataset, loadedRunIDs] = combineRawRuns(cases, cfg)
rawDataset = table();
loadedRunIDs = strings(0,1);
for k = 1:height(cases)
    runFile = fullfile(cfg.output.rawRuns, cases.RunID(k) + ".mat");
    if ~isfile(runFile)
        continue;
    end
    content = load(runFile, "rawTable");
    if ~isfield(content, "rawTable") || isempty(content.rawTable)
        warning("faultdataset:InvalidRunFile", ...
            "%s 不含有效 rawTable，已跳过。", runFile);
        continue;
    end
    if isempty(rawDataset)
        rawDataset = content.rawTable;
    else
        rawDataset = [rawDataset; content.rawTable]; %#ok<AGROW>
    end
    loadedRunIDs(end+1,1) = cases.RunID(k); %#ok<AGROW>
end
end

function ensureOutputFolders(cfg)
folders = [cfg.output.root, cfg.output.rawRuns, ...
    cfg.output.combined, cfg.output.figures, ...
    cfg.output.cache, cfg.output.codegen, cfg.output.temp, ...
    cfg.output.parallelJobs];
for folder = folders
    if ~isfolder(folder)
        mkdir(folder);
    end
end
if ~isfile(cfg.modelFile)
    error("faultdataset:ModelNotFound", ...
        "找不到模型文件: %s", cfg.modelFile);
end
end

function cleanup = configureFileGeneration(cfg)
% 把 MATLAB/Java 临时文件、slprj、slxc 和代码生成文件集中到 D 盘。
previous = Simulink.fileGenControl("getConfig");
previousTemp = string(getenv("TEMP"));
previousTmp = string(getenv("TMP"));
hasJvm = usejava("jvm");
previousJavaTemp = "";
if hasJvm
    previousJavaTemp = string( ...
        java.lang.System.getProperty("java.io.tmpdir"));
end
cleanup = onCleanup(@() restoreFileGeneration( ...
    previous, previousTemp, previousTmp, hasJvm, previousJavaTemp));

setenv("TEMP", char(cfg.output.temp));
setenv("TMP", char(cfg.output.temp));

% tempdir 会永久缓存 MATLAB 启动时的 TEMP/TMP。仅 setenv 不会更新该
% 缓存，必须显式清除，否则 tempname 等仍会写入 C 盘。
clear tempdir
actualTemp = string(tempdir);
if ~isPathInside(actualTemp, cfg.output.temp)
    error("faultdataset:TempRedirectFailed", ...
        "MATLAB 临时目录重定向失败：期望 %s，实际 %s。", ...
        cfg.output.temp, actualTemp);
end

if hasJvm
    java.lang.System.setProperty("java.io.tmpdir", char(cfg.output.temp));
end

Simulink.fileGenControl("set", ...
    "CacheFolder", char(cfg.output.cache), ...
    "CodeGenFolder", char(cfg.output.codegen), ...
    "createDir", true);

fprintf("MATLAB 临时目录: %s\n", actualTemp);
fprintf("Simulink 缓存目录: %s\n", cfg.output.cache);
end

function restoreFileGeneration( ...
        previous, previousTemp, previousTmp, hasJvm, previousJavaTemp)
setenv("TEMP", char(previousTemp));
setenv("TMP", char(previousTmp));
clear tempdir
if hasJvm
    if strlength(previousJavaTemp) == 0
        java.lang.System.clearProperty("java.io.tmpdir");
    else
        java.lang.System.setProperty( ...
            "java.io.tmpdir", char(previousJavaTemp));
    end
end
try
    Simulink.fileGenControl("setConfig", "config", previous);
catch exception
    % 使用项目启动器时，旧 CacheFolder 可能是项目根目录，而旧 TEMP
    % 也位于项目内；Simulink 不允许这种父子关系。此时保留当前项目的
    % cache/codegen 设置，比在 onCleanup 中抛错更安全。
    warning("faultdataset:FileGenerationRestoreSkipped", ...
        "旧 Simulink 文件生成目录与当前临时目录冲突，" + ...
        "已保留本项目的 cache/codegen 设置：%s", exception.message);
end
end

function pool = configureParallelPool(cfg)
% parsim 的 worker 及 JobStorageLocation 也必须在项目的 D 盘目录中。
existingPool = gcp("nocreate");
if ~isempty(existingPool)
    fprintf("正在重建并行池，以应用 D 盘临时目录设置。\n");
    delete(existingPool);
end

cluster = parcluster("local");
cluster.JobStorageLocation = char(cfg.output.parallelJobs);
if ~isscalar(cfg.execution.numWorkers) || ...
        cfg.execution.numWorkers < 1 || ...
        fix(cfg.execution.numWorkers) ~= cfg.execution.numWorkers
    error("faultdataset:InvalidWorkerCount", ...
        "execution.numWorkers 必须是正整数。");
end
pool = parpool(cluster, cfg.execution.numWorkers);

if ~isPathInside(string(pool.Cluster.JobStorageLocation), ...
        cfg.output.parallelJobs)
    delete(pool);
    error("faultdataset:ParallelStorageRedirectFailed", ...
        "并行池 JobStorageLocation 未能重定向到 %s。", ...
        cfg.output.parallelJobs);
end
fprintf("并行任务目录: %s\n", pool.Cluster.JobStorageLocation);
end

function tf = isPathInside(pathToCheck, expectedRoot)
pathToCheck = replace(string(pathToCheck), "/", "\");
expectedRoot = replace(string(expectedRoot), "/", "\");
pathToCheck = strip(pathToCheck, "right", "\");
expectedRoot = strip(expectedRoot, "right", "\");
tf = strcmpi(pathToCheck, expectedRoot) || ...
    startsWith(pathToCheck, expectedRoot + "\", "IgnoreCase", true);
end

function checkOverwritePolicy(cfg)
allowed = ["resume","overwrite","error"];
if ~ismember(cfg.output.overwritePolicy, allowed)
    error("faultdataset:InvalidOverwritePolicy", ...
        "overwritePolicy 必须是 resume、overwrite 或 error。");
end
existing = dir(fullfile(cfg.output.rawRuns, "run_*.mat"));
if ~isempty(existing) && cfg.output.overwritePolicy == "error"
    error("faultdataset:OutputExists", ...
        "raw_runs 已有运行文件。请改用 resume/overwrite 或更换输出目录。");
end
if ~isempty(existing) && cfg.output.overwritePolicy == "overwrite"
    warning("faultdataset:OverwriteEnabled", ...
        "overwrite 已启用：本轮同名运行文件将被覆盖，其他文件不会删除。");
end
end

function text = chooseText(condition, trueText, falseText)
if condition, text = trueText; else, text = falseText; end
end
