function rawTable = extract_raw_signals(simOut, runCase, cfg)
%EXTRACT_RAW_SIGNALS 读取、校验并对齐单次仿真的原始信号。
%   RAWTABLE = EXTRACT_RAW_SIGNALS(SIMOUT, RUNCASE, CFG) 优先从 logsout
%   读取信号；若当前模型使用 To Workspace，则从 SimulationOutput 读取。
%   连续信号线性插值，离散信号前值保持。缺少必要信号时抛出错误，缺少
%   可选信号时使用 NaN 占位，保证所有故障拥有完全一致的列结构。
%
%   输入:
%       simOut  - Simulink.SimulationOutput。
%       runCase - build_simulation_cases 生成的单行 table。
%       cfg     - signal_config 返回的配置。
%   输出:
%       rawTable - 每个统一采样时刻一行的原始数据表。

arguments
    simOut (1,1) Simulink.SimulationOutput
    runCase (1,:) table
    cfg (1,1) struct = signal_config()
end

if height(runCase) ~= 1
    error("faultdataset:CaseMustBeScalar", "runCase 必须只有一行。");
end

stopTime = runCase.SimulationStopTime;
time = makeTimeAxis(stopTime, cfg.sampleTime);
n = numel(time);

spec = [ ...
    makeSpec("ModeCommand", "ModeCommand", "previous")
    makeSpec("Iref", "Iref", "linear")
    makeSpec("IL_meas", "ILMeas", "linear")
    makeSpec("Ibat_meas", "IbatMeas", "linear")
    makeSpec("Vbus_meas", "VbusMeas", "linear")
    makeSpec("Vbat_meas", "VbatMeas", "linear")
    makeSpec("Iload_meas", "IloadMeas", "linear")
    makeSpec("Ibus_source", "IbusSource", "linear")
    makeSpec("SOC_est", "SOCEst", "linear")
    makeSpec("CurrentError", "CurrentError", "linear")
    makeSpec("VoltageError", "VoltageError", "linear")
    makeSpec("DutyRaw", "DutyRaw", "linear")
    makeSpec("DutyApplied", "DutyApplied", "linear")
    makeSpec("PIiOut", "PIiOut", "linear")
    makeSpec("PIiIntegral", "PIiIntegral", "linear")
    makeSpec("PIvOut", "PIvOut", "linear")
    makeSpec("S1GateCmd", "S1GateCmd", "previous")
    makeSpec("S2GateCmd", "S2GateCmd", "previous")
    makeSpec("Validation_S1GateActual", "S1GateActual", "binmean")
    makeSpec("Validation_S2GateActual", "S2GateActual", "binmean")
    makeSpec("Validation_S1GateMismatch", "S1GateMismatch", "binmean")
    makeSpec("Validation_S2GateMismatch", "S2GateMismatch", "binmean")
    makeSpec("S1_device_current", "S1DeviceCurrent", "binmean")
    makeSpec("S1_device_voltage", "S1DeviceVoltage", "binmean")
    makeSpec("S2_device_current", "S2DeviceCurrent", "binmean")
    makeSpec("S2_device_voltage", "S2DeviceVoltage", "binmean")
    % 统一采样周期正好等于 20 kHz PWM 周期，单点采样会发生相位混叠。
    % 因此另外从原生高频门极数据计算每个采样区间内的平均导通比例。
    makeSpec("S1GateDuty", "S1GateCmd", "binmean")
    makeSpec("S2GateDuty", "S2GateCmd", "binmean")
    makeSpec("IL_true", "ILTrue", "linear")
    makeSpec("Ibat_true", "IbatTrue", "linear")
    makeSpec("Vbus_true", "VbusTrue", "linear")
    makeSpec("Vbat_true", "VbatTrue", "linear")
    makeSpec("SOC_true", "SOCTrue", "linear")
    makeSpec("Psource_meas", "PsourceMeas", "linear")
    makeSpec("Pload_meas", "PloadMeas", "linear")
    makeSpec("Pstored_meas", "PstoredMeas", "linear")
    makeSpec("PowerBalanceResidual", "PowerBalanceResidual", "linear")];

data = struct();
missingRequired = strings(0,1);
for k = 1:numel(spec)
    item = spec(k);
    candidates = cfg.signalNames.(item.ConfigField);
    [traceTime, traceData, found] = readSignal(simOut, candidates);
    if found
        data.(item.RawField) = alignTrace( ...
            traceTime, traceData, time, item.Method, item.RawField);
    else
        data.(item.RawField) = nan(n,1);
        if ismember(item.RawField, cfg.requiredRawFields)
            missingRequired(end+1,1) = item.RawField; %#ok<AGROW>
        else
            warnOptionalOnce(item.RawField, candidates);
        end
    end
end

[data.S1_conduction_ratio, data.S1_ron_estimate] = ...
    switchMeasurementFeatures(simOut, cfg.signalNames.S1DeviceCurrent, ...
    cfg.signalNames.S1DeviceVoltage, time, cfg.sampleTime, ...
    cfg.switchMeasurement.currentThreshold);
[data.S2_conduction_ratio, data.S2_ron_estimate] = ...
    switchMeasurementFeatures(simOut, cfg.signalNames.S2DeviceCurrent, ...
    cfg.signalNames.S2DeviceVoltage, time, cfg.sampleTime, ...
    cfg.switchMeasurement.currentThreshold);

if ~isempty(missingRequired)
    error("faultdataset:MissingRequiredSignals", ...
        "RunID=%s 缺少必要信号: %s", runCase.RunID, ...
        strjoin(missingRequired, ", "));
end

% 当前模型未直接记录 gate_enable，以 Mode_Manager 的 mode_id 派生。
[modeTime, modeData, modeFound] = readSignal( ...
    simOut, cfg.signalNames.ModeID);
if modeFound
    modeID = alignTrace(modeTime, modeData, time, "previous", "mode_id");
    converterEnable = double(ismember(round(modeID), [2 3 4]));
else
    converterEnable = double(data.ModeCommand ~= 0);
    warnOptionalOnce("ConverterEnable", cfg.signalNames.ModeID);
end

% 分别保留电流环和电压环饱和标志；SatFlag 继续作为兼容旧数据的汇总列。
[satITime, satIData, satIFound] = readSignal( ...
    simOut, cfg.signalNames.SatFlagI);
[satVTime, satVData, satVFound] = readSignal( ...
    simOut, cfg.signalNames.SatFlagV);
satI = nan(n,1);
satV = nan(n,1);
if satIFound || satVFound
    satI = zeros(n,1);
    satV = zeros(n,1);
    if satIFound
        satI = alignTrace(satITime, satIData, time, "previous", "sat_flag_I");
    end
    if satVFound
        satV = alignTrace(satVTime, satVData, time, "previous", "sat_flag_V");
    end
    satFlag = double((satI ~= 0) | (satV ~= 0));
else
    satFlag = nan(n,1);
    warnOptionalOnce("SatFlag", ...
        [cfg.signalNames.SatFlagI cfg.signalNames.SatFlagV]);
end

% 若控制器误差信号不可用，按照约定 Iref-IL_meas 计算。
if all(isnan(data.CurrentError))
    data.CurrentError = data.Iref - data.IL_meas;
end

% 标签由仿真配置和时间生成，不根据文件名给故障前样本贴故障标签。
scenarioFaultID = double(runCase.FaultID);
faultActive = double(scenarioFaultID ~= 0 & ...
    time >= runCase.FaultStartTime & time < runCase.FaultEndTime);
transitionWindow = double(scenarioFaultID ~= 0 & ...
    time >= runCase.FaultStartTime & ...
    time < min(runCase.FaultStartTime + cfg.transitionDuration, ...
    runCase.FaultEndTime));
activeFaultID = scenarioFaultID .* faultActive;

% 部分/间歇开路按与模型相同的周期相位和严重度计算实际屏蔽比例。
% 不使用 Fault_Diag_Manager 的 fault_active，因为该信号可能锁存。
isGateBlockingSwitchFault = ...
    ismember(scenarioFaultID, cfg.labeling.switchFaultIDs) && ...
    cfg.adapter.switchFaultMechanism == "gate_blocking";
if isGateBlockingSwitchFault
    period = cfg.adapter.switchFaultPeriod;
    faultActive = gateBlockingBinAverage(time, cfg.sampleTime, ...
        runCase.FaultStartTime, runCase.FaultEndTime, period, ...
        runCase.FaultMagnitude);
    activeFaultID = scenarioFaultID .* faultActive;
end

% 模型 fault_active 仅用于非门极周期故障的时序核验。
[faTime, faData, faFound] = readSignal( ...
    simOut, cfg.signalNames.FaultActive);
modelFaultActive = nan(n,1);
if faFound
    modelFaultActive = alignTrace( ...
        faTime, faData, time, "previous", "ModelFaultActive");
end

% 开关开路只有在对应开关本应导通时才具有电气可观测性。这里不改变
% FaultActive/ActiveFaultID 的物理故障含义，而是增加独立可观测标签。
faultObservable = faultActive;
if scenarioFaultID == cfg.labeling.s1OpenFaultID
    faultObservable = faultActive .* double(converterEnable ~= 0) .* ...
        double(data.S1GateDuty >= cfg.labeling.minimumGateDuty);
elseif scenarioFaultID == cfg.labeling.s2OpenFaultID
    faultObservable = faultActive .* double(converterEnable ~= 0) .* ...
        double(data.S2GateDuty >= cfg.labeling.minimumGateDuty);
elseif scenarioFaultID == 0
    faultObservable = zeros(n,1);
end
observableFaultID = scenarioFaultID .* faultObservable;

% 样本级训练资格仅作审计；窗口级资格会在切窗后重新严格计算。
if scenarioFaultID == 0
    sampleTrainingEligible = ones(n,1);
elseif ismember(scenarioFaultID, cfg.labeling.switchFaultIDs)
    sampleTrainingEligible = double(faultObservable ~= 0);
else
    sampleTrainingEligible = double(faultActive ~= 0);
end
if cfg.labeling.excludeTransitionFromTraining
    sampleTrainingEligible(transitionWindow ~= 0) = 0;
end

% 对传感器故障和连续开路，模型信号继续用于核验注入时序。
isHighResistanceSwitchFault = ...
    ismember(scenarioFaultID, cfg.labeling.switchFaultIDs) && ...
    cfg.adapter.switchFaultMechanism == "high_resistance";
if faFound && ~isHighResistanceSwitchFault && ...
        ~isGateBlockingSwitchFault
    comparisonMask = time >= runCase.FaultStartTime + cfg.sampleTime;
    mismatchRatio = mean((modelFaultActive(comparisonMask) ~= 0) ~= ...
        (faultActive(comparisonMask) ~= 0), "omitnan");
    if isfinite(mismatchRatio) && mismatchRatio > 0.01
        warning("faultdataset:FaultActiveMismatch", ...
            "RunID=%s 的模型 FaultActive 与配置标签不一致，比例 %.3f。", ...
            runCase.RunID, mismatchRatio);
    end
end

rawTable = table( ...
    time, ...
    repmat(string(runCase.RunID),n,1), ...
    repmat(string(runCase.OperatingPointID),n,1), ...
    repmat(runCase.RandomSeed,n,1), ...
    repmat(runCase.SOCInit,n,1), repmat(runCase.IrefLevel,n,1), ...
    repmat(runCase.VbatInit,n,1), repmat(runCase.Rload,n,1), ...
    repmat(runCase.Pload,n,1), repmat(runCase.Rbat,n,1), ...
    repmat(runCase.Cbus,n,1), repmat(runCase.CbusESR,n,1), ...
    data.ModeCommand, converterEnable, ...
    data.Iref, repmat(runCase.VbusRef,n,1), ...
    data.IL_meas, data.Ibat_meas, data.Vbus_meas, data.Vbat_meas, ...
    data.Iload_meas, data.Ibus_source, data.SOC_est, ...
    data.CurrentError, data.VoltageError, data.DutyRaw, ...
    data.DutyApplied, data.PIiOut, data.PIiIntegral, data.PIvOut, ...
    satI, satV, satFlag, data.S1GateCmd, data.S2GateCmd, ...
    data.Validation_S1GateActual, data.Validation_S2GateActual, ...
    data.Validation_S1GateMismatch, data.Validation_S2GateMismatch, ...
    data.S1_device_current, data.S1_device_voltage, ...
    data.S2_device_current, data.S2_device_voltage, ...
    data.S1_conduction_ratio, data.S1_ron_estimate, ...
    data.S2_conduction_ratio, data.S2_ron_estimate, ...
    data.S1GateDuty, data.S2GateDuty, ...
    data.IL_true, data.Ibat_true, data.Vbus_true, data.Vbat_true, ...
    data.SOC_true, data.Psource_meas, data.Pload_meas, ...
    data.Pstored_meas, data.PowerBalanceResidual, ...
    repmat(scenarioFaultID,n,1), ...
    repmat(string(runCase.FaultName),n,1), ...
    repmat(string(runCase.FaultLocation),n,1), ...
    repmat(runCase.FaultMagnitude,n,1), ...
    repmat(runCase.FaultParameter1,n,1), ...
    repmat(runCase.FaultParameter2,n,1), ...
    repmat(runCase.FaultStartTime,n,1), ...
    repmat(runCase.FaultEndTime,n,1), ...
    faultActive, transitionWindow, faultObservable, ...
    repmat(scenarioFaultID,n,1), activeFaultID, observableFaultID, ...
    sampleTrainingEligible, ...
    'VariableNames', { ...
    'Time','RunID','OperatingPointID','RandomSeed', ...
    'SOCInit','IrefLevel','VbatInit','Rload','Pload', ...
    'Rbat','Cbus','CbusESR', ...
    'ModeCommand','ConverterEnable','Iref','VbusRef', ...
    'IL_meas','Ibat_meas','Vbus_meas','Vbat_meas','Iload_meas', ...
    'Ibus_source','SOC_est', ...
    'CurrentError','VoltageError','DutyRaw','DutyApplied', ...
    'PIiOut','PIiIntegral','PIvOut','SatFlagI','SatFlagV','SatFlag', ...
    'S1GateCmd','S2GateCmd', ...
    'Validation_S1GateActual','Validation_S2GateActual', ...
    'Validation_S1GateMismatch','Validation_S2GateMismatch', ...
    'S1_device_current','S1_device_voltage', ...
    'S2_device_current','S2_device_voltage', ...
    'S1_conduction_ratio','S1_ron_estimate', ...
    'S2_conduction_ratio','S2_ron_estimate', ...
    'S1GateDuty','S2GateDuty', ...
    'IL_true','Ibat_true','Vbus_true','Vbat_true','SOC_true', ...
    'Psource_meas','Pload_meas','Pstored_meas','PowerBalanceResidual', ...
    'FaultID','FaultName','FaultLocation','FaultMagnitude', ...
    'FaultParameter1','FaultParameter2','FaultStartTime','FaultEndTime', ...
    'FaultActive','TransitionWindow','FaultObservable', ...
    'ScenarioFaultID','ActiveFaultID','ObservableFaultID', ...
    'IsTrainingEligible'});

validateAlignedTable(rawTable);
end

function fraction = gateBlockingBinAverage( ...
        time, sampleTime, faultStart, faultEnd, period, duty)
%GATEBLOCKINGBINAVERAGE Exact blocked fraction in each logged time bin.
% Point-sampling a periodic gate fault can over- or under-count severity
% when the logging interval is harmonic with the fault period.
duty = min(max(double(duty), 0), 1);
binStart = max(time, faultStart);
binEnd = min(time + sampleTime, faultEnd);
valid = binEnd > binStart;

fraction = zeros(size(time));
if duty == 0 || ~any(valid)
    return
end

activeEnd = periodicActiveIntegral(binEnd(valid), period, duty);
activeStart = periodicActiveIntegral(binStart(valid), period, duty);
fraction(valid) = (activeEnd - activeStart) ./ sampleTime;
fraction = min(max(fraction, 0), 1);
end

function activeTime = periodicActiveIntegral(time, period, duty)
%PERIODICACTIVEINTEGRAL Integral of phase/period < duty from zero to time.
time = max(time, 0);
cycleCount = floor(time ./ period);
phaseTime = time - cycleCount .* period;
activeTime = cycleCount .* (duty .* period) + ...
    min(phaseTime, duty .* period);
end

function item = makeSpec(rawField, configField, method)
item = struct("RawField", rawField, "ConfigField", configField, ...
    "Method", method);
end

function time = makeTimeAxis(stopTime, sampleTime)
validateattributes(stopTime, {'numeric'}, {'scalar','positive','finite'});
validateattributes(sampleTime, {'numeric'}, {'scalar','positive','finite'});
count = floor(stopTime / sampleTime);
time = (0:count)' .* sampleTime;
if time(end) < stopTime - 10*eps(stopTime)
    time(end+1,1) = stopTime;
end
end

function [time, data, found] = readSignal(simOut, candidates)
time = [];
data = [];
found = false;
candidates = string(candidates);

% 按候选名优先级逐一查找，并对每个候选先查 logsout、再查
% SimulationOutput。这样 load_current 不会被后面的 I_Rh 等 logsout 名
% 抢先命中。
try
    logsout = simOut.get("logsout");
catch
    logsout = [];
end
try
    xout = simOut.get("xout");
catch
    xout = [];
end
available = string(who(simOut));
for candidate = candidates
    datasets = {logsout, xout};
    for datasetIndex = 1:numel(datasets)
        dataset = datasets{datasetIndex};
        if isa(dataset, "Simulink.SimulationData.Dataset")
            for elementIndex = 1:dataset.numElements
                element = dataset.get(elementIndex);
                if string(element.Name) == candidate
                    [time, data, found] = unpackSignal(element.Values);
                    if found
                        return;
                    end
                end
            end
        end
    end
    if ismember(candidate, available)
        value = simOut.get(candidate);
        [time, data, found] = unpackSignal(value);
        if found
            return;
        end
    end
end
end

function [conductionRatio, ronEstimate] = switchMeasurementFeatures( ...
        simOut, currentCandidates, voltageCandidates, queryTime, ...
        sampleTime, currentThreshold)
%SWITCHMEASUREMENTFEATURES Joint high-rate V/I aggregation per output bin.
[currentTime, currentData, currentFound] = ...
    readSignal(simOut, currentCandidates);
[voltageTime, voltageData, voltageFound] = ...
    readSignal(simOut, voltageCandidates);
n = numel(queryTime);
conductionRatio = nan(n,1);
ronEstimate = nan(n,1);
if ~currentFound || ~voltageFound
    return;
end

if numel(currentTime) == numel(voltageTime) && ...
        all(abs(currentTime-voltageTime) <= 10*eps(max(1,max(currentTime))))
    voltageAtCurrent = voltageData;
else
    voltageAtCurrent = interp1( ...
        voltageTime, voltageData, currentTime, "linear", NaN);
end

edges = [queryTime(:); queryTime(end)+sampleTime];
bin = discretize(currentTime, edges);
valid = ~isnan(bin) & isfinite(currentData) & isfinite(voltageAtCurrent);
conducting = valid & abs(currentData) >= currentThreshold;
if any(valid)
    conductionRatio = accumarray(bin(valid), double(conducting(valid)), ...
        [n,1], @mean, NaN);
end
if any(conducting)
    ratio = abs(voltageAtCurrent(conducting) ./ currentData(conducting));
    ronEstimate = accumarray(bin(conducting), ratio, ...
        [n,1], @median, NaN);
end
end

function [time, data, found] = unpackSignal(value)
time = [];
data = [];
found = false;
if isa(value, "timeseries")
    time = double(value.Time(:));
    data = squeeze(double(value.Data));
elseif istimetable(value)
    time = seconds(value.Properties.RowTimes - value.Properties.RowTimes(1));
    data = value{:,1};
elseif isstruct(value) && isfield(value, "time") && ...
        isfield(value, "signals") && ~isempty(value.signals)
    time = double(value.time(:));
    data = squeeze(double(value.signals(1).values));
else
    return;
end
if ~isvector(data)
    data = data(:,1);
end
data = double(data(:));
found = ~isempty(time) && numel(time) == numel(data);
end

function aligned = alignTrace(time, data, queryTime, method, fieldName)
if any(~isfinite(time))
    error("faultdataset:InvalidTime", "%s 的时间包含 NaN 或 Inf。", fieldName);
end
[time, order] = sort(time(:), "ascend");
data = data(order);
[time, uniqueIndex] = unique(time, "stable");
data = data(uniqueIndex);
if any(diff(time) <= 0)
    error("faultdataset:NonMonotonicTime", "%s 的时间不是严格递增。", fieldName);
end
if numel(time) == 1
    aligned = repmat(data(1), numel(queryTime), 1);
    return;
end
if method == "binmean"
    aligned = binMeanTrace(time, data, queryTime);
    return;
elseif method == "previous"
    aligned = interp1(time, data, queryTime, "previous", NaN);
else
    aligned = interp1(time, data, queryTime, "linear", NaN);
end
% 仅处理由浮点端点误差产生的首尾 NaN，不对大段缺测做外推。
tol = max(eps(max(abs(time))), eps);
left = isnan(aligned) & queryTime >= time(1) - tol;
right = isnan(aligned) & queryTime <= time(end) + tol;
aligned(left & queryTime <= time(1)) = data(1);
aligned(right & queryTime >= time(end)) = data(end);
end

function aligned = binMeanTrace(time, data, queryTime)
if numel(queryTime) == 1
    aligned = mean(data, "omitnan");
    return;
end
step = median(diff(queryTime));
edges = [queryTime(:); queryTime(end)+step];
bin = discretize(time, edges);
valid = ~isnan(bin) & isfinite(data);
aligned = accumarray(bin(valid), data(valid), ...
    [numel(queryTime),1], @mean, NaN);
% 极少数空区间用前值保持补齐；不对真实的大段缺测进行线性外推。
empty = ~isfinite(aligned);
if any(empty)
    held = interp1(time, data, queryTime, "previous", NaN);
    aligned(empty) = held(empty);
end
end

function warnOptionalOnce(fieldName, candidates)
persistent warnedFields
if isempty(warnedFields)
    warnedFields = strings(0,1);
end
if ~ismember(fieldName, warnedFields)
    warning("faultdataset:MissingOptionalSignal", ...
        "缺少可选信号 %s（候选名: %s），使用 NaN 占位。", ...
        fieldName, strjoin(string(candidates), ", "));
    warnedFields(end+1,1) = fieldName;
end
end

function validateAlignedTable(rawTable)
if any(diff(rawTable.Time) <= 0)
    error("faultdataset:InvalidAlignedTime", "统一时间轴不是严格递增。");
end
expectedLength = height(rawTable);
for name = string(rawTable.Properties.VariableNames)
    if height(rawTable(:,name)) ~= expectedLength
        error("faultdataset:LengthMismatch", "字段 %s 长度不一致。", name);
    end
end
end
