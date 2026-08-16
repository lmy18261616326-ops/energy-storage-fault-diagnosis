function [featureTable, featureColumns, labelColumn, ...
    metadataColumns, excludedColumns] = extract_window_features(rawTable, cfg)
%EXTRACT_WINDOW_FEATURES 滑动窗口切分并提取统计、物理和控制特征。
%   [FEATURETABLE, FEATURECOLUMNS, LABELCOLUMN, METADATACOLUMNS,
%   EXCLUDEDCOLUMNS] = EXTRACT_WINDOW_FEATURES(RAWTABLE, CFG) 对每个 RunID
%   独立切窗，避免跨仿真拼接。窗口标签依据 FaultActiveRatio 生成。
%
%   输入:
%       rawTable - extract_raw_signals 输出的一个或多个 RunID 原始表。
%       cfg      - signal_config 返回的配置。
%   输出:
%       featureTable    - 每个窗口一行的特征表。
%       featureColumns  - 可作为机器学习输入的列名。
%       labelColumn     - 标签列名 "WindowFaultID"。
%       metadataColumns - 运行和故障配置元数据列。
%       excludedColumns - 禁止进入模型的元数据、真实量和 Validation_ 列。

arguments
    rawTable table
    cfg (1,1) struct = signal_config()
end

required = ["Time","RunID","OperatingPointID","ModeCommand", ...
    "ScenarioFaultID","FaultActive","FaultObservable", ...
    "TransitionWindow"];
missing = setdiff(required, string(rawTable.Properties.VariableNames));
if ~isempty(missing)
    error("faultdataset:MissingRawColumns", ...
        "窗口提取缺少字段: %s", strjoin(missing, ", "));
end

runIDs = unique(string(rawTable.RunID), "stable");
rows = struct([]);
rowCount = 0;
for runIndex = 1:numel(runIDs)
    runData = rawTable(string(rawTable.RunID) == runIDs(runIndex),:);
    runData = sortrows(runData, "Time");
    firstStart = runData.Time(1);
    lastStart = runData.Time(end) - cfg.windowLength;
    if lastStart < firstStart
        warning("faultdataset:RunTooShort", ...
            "RunID=%s 长度不足一个窗口，已跳过。", runIDs(runIndex));
        continue;
    end
    windowStarts = (firstStart:cfg.windowStep:lastStart)';
    for windowIndex = 1:numel(windowStarts)
        startTime = windowStarts(windowIndex);
        endTime = startTime + cfg.windowLength;
        mask = runData.Time >= startTime & ...
            runData.Time <= endTime + 10*eps(endTime);
        window = runData(mask,:);
        if height(window) < 3
            continue;
        end

        rowCount = rowCount + 1;
        row = struct();
        row.RunID = string(window.RunID(1));
        row.OperatingPointID = string(window.OperatingPointID(1));
        row.WindowID = windowIndex;
        row.WindowStart = startTime;
        row.WindowEnd = endTime;
        row.RandomSeed = window.RandomSeed(1);
        row.SOCInit = window.SOCInit(1);
        row.IrefLevel = window.IrefLevel(1);
        row.VbusRefSetting = window.VbusRef(1);
        row.VbatInit = window.VbatInit(1);
        row.Rload = window.Rload(1);
        row.Pload = window.Pload(1);
        row.Rbat = window.Rbat(1);
        row.Cbus = window.Cbus(1);
        row.CbusESR = window.CbusESR(1);
        row.ModeCommand = robustMode(window.ModeCommand);
        row.ScenarioFaultID = robustMode(window.ScenarioFaultID);
        row.FaultActiveRatio = mean(window.FaultActive, "omitnan");
        row.FaultObservableRatio = mean( ...
            window.FaultObservable, "omitnan");
        row.TransitionRatio = mean(window.TransitionWindow, "omitnan");
        row.FaultMagnitude = window.FaultMagnitude(1);
        row.FaultName = string(window.FaultName(1));
        row.FaultLocation = string(window.FaultLocation(1));
        row.FaultParameter1 = window.FaultParameter1(1);
        row.FaultParameter2 = window.FaultParameter2(1);
        row.FaultStartTime = window.FaultStartTime(1);
        row.FaultEndTime = window.FaultEndTime(1);

        isSwitchFault = ismember( ...
            row.ScenarioFaultID, cfg.labeling.switchFaultIDs);
        isPhysicallyActive = row.FaultActiveRatio >= ...
            cfg.labeling.activeRatioThreshold;
        isObservable = row.FaultObservableRatio >= ...
            cfg.labeling.observableRatioThreshold;
        if isSwitchFault && isObservable
            row.WindowFaultID = row.ScenarioFaultID;
        elseif ~isSwitchFault && isPhysicallyActive
            row.WindowFaultID = row.ScenarioFaultID;
        else
            row.WindowFaultID = 0;
        end
        row.ActiveFaultID = double(isPhysicallyActive) .* ...
            row.ScenarioFaultID;
        row.ObservableFaultID = double(isObservable) .* ...
            row.ScenarioFaultID;
        crossesFault = ...
            (startTime < row.FaultStartTime && ...
            endTime > row.FaultStartTime) || ...
            (startTime < row.FaultEndTime && ...
            endTime > row.FaultEndTime);
        row.IsTransitionWindow = double(crossesFault || ...
            row.TransitionRatio > 0);
        if row.ScenarioFaultID == 0
            row.IsTrainingEligible = 1;
        elseif isSwitchFault
            row.IsTrainingEligible = double(isObservable);
        else
            row.IsTrainingEligible = double(isPhysicallyActive);
        end
        if cfg.labeling.excludeTransitionFromTraining && ...
                row.IsTransitionWindow
            row.IsTrainingEligible = 0;
        end

        for signalName = cfg.featureSignals
            x = getNumericColumn(window, signalName);
            row = addBasicStats(row, signalName, window.Time, x);
            if ismember(signalName, cfg.currentFeatureSignals)
                row = addCurrentStats(row, signalName, x, ...
                    cfg.nearZeroThreshold);
            end
        end

        % 真实量特征仅用于分析，随后明确放入 excludedColumns。
        for signalName = cfg.trueFeatureSignals
            x = getNumericColumn(window, signalName);
            row = addBasicStats(row, signalName, window.Time, x);
        end

        row = addDiscreteStats(row, window);
        row = addPhysicsStats(row, window, cfg);
        if rowCount == 1
            rows = row;
        else
            missingInRow = setdiff(fieldnames(rows), fieldnames(row));
            extraInRow = setdiff(fieldnames(row), fieldnames(rows));
            if ~isempty(missingInRow) || ~isempty(extraInRow)
                error("faultdataset:InconsistentFeatureSchema", ...
                    "窗口特征列结构不一致。缺少: %s；新增: %s。", ...
                    strjoin(string(missingInRow), ", "), ...
                    strjoin(string(extraInRow), ", "));
            end
            row = orderfields(row, rows);
            rows(rowCount,1) = row; %#ok<AGROW>
        end
    end
end

if isempty(rows)
    error("faultdataset:NoWindows", "没有生成任何有效窗口。");
end
featureTable = struct2table(rows, "AsArray", true);
featureTable.SampleWeight = calculateSampleWeights( ...
    featureTable.WindowFaultID, featureTable.IsTrainingEligible);

labelColumn = "WindowFaultID";
metadataRequested = [ ...
    "RunID","OperatingPointID","WindowID","WindowStart","WindowEnd", ...
    "RandomSeed","SOCInit","IrefLevel","VbusRefSetting","VbatInit", ...
    "Rload","Pload","Rbat","Cbus","CbusESR", ...
    "FaultName","FaultLocation","ScenarioFaultID","ActiveFaultID", ...
    "ObservableFaultID", ...
    "FaultMagnitude","FaultParameter1","FaultParameter2", ...
    "FaultStartTime","FaultEndTime","FaultActiveRatio", ...
    "FaultObservableRatio","TransitionRatio","IsTransitionWindow", ...
    "IsTrainingEligible","SampleWeight"];
variables = string(featureTable.Properties.VariableNames);
metadataColumns = intersect(metadataRequested, variables, "stable");

validationColumns = variables(startsWith(variables, "Validation_"));
trueColumns = strings(0,1);
for prefix = ["IL_true","Ibat_true","Vbus_true","Vbat_true","SOC_true"]
    trueColumns = [trueColumns; ...
        variables(startsWith(variables, prefix))']; %#ok<AGROW>
end
excludedColumns = unique( ...
    [metadataColumns(:); validationColumns(:); trueColumns(:)], "stable");

policyExcluded = strings(0,1);
if ~cfg.featurePolicy.useIdealSOCEst
    policyExcluded = [policyExcluded; ...
        variables(startsWith(variables, "SOC_est"))']; %#ok<AGROW>
end
if ~cfg.featurePolicy.useUnbalancedPowerResidual
    policyExcluded = [policyExcluded; intersect( ...
        ["PowerResidualMean";"PowerResidualRMS"], ...
        variables, "stable")]; %#ok<AGROW>
end
if ~cfg.featurePolicy.useBalancedPowerResidual
    policyExcluded = [policyExcluded; ...
        variables(startsWith(variables, "PowerBalanceResidual"))'; ...
        variables(startsWith(variables, "BalancedPowerResidual"))']; %#ok<AGROW>
end
excludedColumns = unique([excludedColumns(:); policyExcluded(:)], "stable");

candidateFeatures = setdiff(variables, ...
    [labelColumn; excludedColumns], "stable");
isNumericFeature = false(size(candidateFeatures));
for k = 1:numel(candidateFeatures)
    isNumericFeature(k) = isnumeric(featureTable.(candidateFeatures(k))) || ...
        islogical(featureTable.(candidateFeatures(k)));
end
featureColumns = candidateFeatures(isNumericFeature);
[featureColumns, lowInformationExcluded] = filterLowInformationFeatures( ...
    featureTable, featureColumns, cfg.featurePolicy);
excludedColumns = unique( ...
    [excludedColumns(:); lowInformationExcluded(:)], "stable");
end

function row = addBasicStats(row, prefix, time, x)
valid = isfinite(time) & isfinite(x);
xv = x(valid);
tv = time(valid);
if isempty(xv)
    values = nan(1,11);
else
    firstValue = xv(1);
    lastValue = xv(end);
    minimum = min(xv);
    maximum = max(xv);
    values = [ ...
        mean(xv), std(xv,0), rootMeanSquare(xv), minimum, maximum, ...
        maximum-minimum, median(xv), linearSlope(tv,xv), ...
        firstValue, lastValue, lastValue-firstValue];
end
suffixes = ["Mean","Std","RMS","Min","Max","Range","Median", ...
    "Slope","FirstValue","LastValue","Delta"];
for k = 1:numel(suffixes)
    row.(prefix + suffixes(k)) = values(k);
end
end

function row = addCurrentStats(row, prefix, x, nearZeroThreshold)
x = x(isfinite(x));
if isempty(x)
    values = nan(1,6);
else
    signalRMS = rootMeanSquare(x);
    if signalRMS <= eps
        crestFactor = 0;
    else
        crestFactor = max(abs(x)) / signalRMS;
    end
    if numel(x) > 1
        diffRMS = rootMeanSquare(diff(x));
        zeroCrossingRate = mean(x(1:end-1).*x(2:end) < 0);
    else
        diffRMS = 0;
        zeroCrossingRate = 0;
    end
    values = [sampleSkewness(x), sampleKurtosis(x), crestFactor, ...
        diffRMS, zeroCrossingRate, mean(abs(x) <= nearZeroThreshold)];
end
suffixes = ["Skewness","Kurtosis","CrestFactor","DiffRMS", ...
    "ZeroCrossingRate","NearZeroRatio"];
for k = 1:numel(suffixes)
    row.(prefix + suffixes(k)) = values(k);
end
end

function row = addDiscreteStats(row, window)
row.SatRatio = meanFinite(window.SatFlag);
row.EnableRatio = meanFinite(window.ConverterEnable);
if ismember("S1GateDuty", string(window.Properties.VariableNames))
    row.S1GateDutyRatio = meanFinite(window.S1GateDuty);
    row.S2GateDutyRatio = meanFinite(window.S2GateDuty);
else
    row.S1GateDutyRatio = meanFinite(window.S1GateCmd);
    row.S2GateDutyRatio = meanFinite(window.S2GateCmd);
end
modeValues = window.ModeCommand(isfinite(window.ModeCommand));
if numel(modeValues) < 2
    row.ModeChangeCount = 0;
else
    row.ModeChangeCount = sum(diff(modeValues) ~= 0);
end
end

function weights = calculateSampleWeights(labels, eligibility)
weights = zeros(size(labels));
eligible = eligibility ~= 0 & isfinite(labels);
classes = unique(labels(eligible), "sorted");
if isempty(classes)
    return;
end
eligibleCount = nnz(eligible);
for classID = classes'
    classMask = eligible & labels == classID;
    weights(classMask) = eligibleCount / ...
        (numel(classes) * nnz(classMask));
end
end

function [kept, excluded] = filterLowInformationFeatures( ...
    data, candidates, policy)
keepMask = true(size(candidates));
excluded = strings(0,1);
for k = 1:numel(candidates)
    name = candidates(k);
    values = double(data.(name));
    finiteValues = values(isfinite(values));
    remove = false;
    if policy.excludeAllNaN && isempty(finiteValues)
        remove = true;
    elseif policy.excludeConstant && ...
            (numel(finiteValues) <= 1 || ...
            max(finiteValues) == min(finiteValues))
        remove = true;
    elseif policy.excludeNearZeroVariance
        scale = max(1, mean(abs(finiteValues))^2);
        remove = var(finiteValues,1) <= ...
            policy.nearZeroRelativeTolerance * scale;
    end
    if remove
        keepMask(k) = false;
        excluded(end+1,1) = name; %#ok<AGROW>
    end
end
kept = candidates(keepMask);
end

function row = addPhysicsStats(row, window, cfg)
t = window.Time;

% 电流跟踪误差：优先使用模型实际输出，缺失时用 Iref-IL_meas。
currentError = window.CurrentError;
if all(~isfinite(currentError))
    currentError = window.Iref - window.IL_meas;
end
row.CurrentErrorMean = meanFinite(currentError);
row.CurrentErrorStd = stdFinite(currentError);
row.CurrentErrorRMSE = rootMeanSquare(currentError);
row.CurrentErrorMAE = mean(abs(currentError), "omitnan");
row.CurrentErrorMax = maxFinite(currentError);
row.CurrentErrorMin = minFinite(currentError);
row.CurrentErrorSlope = linearSlope(t,currentError);

currentPair = window.IL_meas - ...
    cfg.currentDirectionFactor .* window.Ibat_meas;
row.CurrentPairResidualMean = meanFinite(currentPair);
row.CurrentPairResidualStd = stdFinite(currentPair);
row.CurrentPairResidualRMS = rootMeanSquare(currentPair);
row.CurrentPairResidualMAE = mean(abs(currentPair), "omitnan");
row.CurrentPairResidualSlope = linearSlope(t,currentPair);

% 使用真实量的残差统一加 Validation_ 前缀，默认禁止进入模型。
row = addValidationResidual(row, "ILSensorResidual", ...
    window.IL_meas-window.IL_true);
row = addValidationResidual(row, "VbusSensorResidual", ...
    window.Vbus_meas-window.Vbus_true);
row = addValidationResidual(row, "VbatSensorResidual", ...
    window.Vbat_meas-window.Vbat_true);
row = addValidationResidual(row, "IbatSensorResidual", ...
    window.Ibat_meas-window.Ibat_true);

pbatMeas = window.Vbat_meas .* window.Ibat_meas;
pbusMeas = window.Vbus_meas .* window.Iload_meas;
pbatTrue = window.Vbat_true .* window.Ibat_true;
row.PbatMean = meanFinite(pbatMeas);
row.PbatStd = stdFinite(pbatMeas);
row.PbusMean = meanFinite(pbusMeas);
row.PbusStd = stdFinite(pbusMeas);
powerResidual = pbatMeas - pbusMeas;
row.PowerResidualMean = meanFinite(powerResidual);
row.PowerResidualRMS = rootMeanSquare(powerResidual);
row.BalancedPowerResidualMean = ...
    meanFinite(window.PowerBalanceResidual);
row.BalancedPowerResidualStd = ...
    stdFinite(window.PowerBalanceResidual);
row.BalancedPowerResidualRMS = ...
    rootMeanSquare(window.PowerBalanceResidual);
row.Validation_PbatTrueMean = meanFinite(pbatTrue);

dutyLimitResidual = window.DutyRaw - window.DutyApplied;
row.DutyRawMean = meanFinite(window.DutyRaw);
row.DutyAppliedMean = meanFinite(window.DutyApplied);
row.DutyLimitResidualMean = meanFinite(dutyLimitResidual);
row.DutyLimitResidualRMS = rootMeanSquare(dutyLimitResidual);
tol = 1e-9;
row.DutySatRatio = mean( ...
    window.DutyApplied <= cfg.dutyLimits(1)+tol | ...
    window.DutyApplied >= cfg.dutyLimits(2)-tol, "omitnan");
row.PIiOutMean = meanFinite(window.PIiOut);
row.PIiOutStd = stdFinite(window.PIiOut);
row.PIiIntegralMean = meanFinite(window.PIiIntegral);
row.PIiIntegralDelta = lastMinusFirst(window.PIiIntegral);
row.PIiIntegralSlope = linearSlope(t,window.PIiIntegral);
row.PIvOutMean = meanFinite(window.PIvOut);
row.PIvOutStd = stdFinite(window.PIvOut);
end

function row = addValidationResidual(row, name, residual)
row.("Validation_"+name+"Mean") = meanFinite(residual);
row.("Validation_"+name+"Std") = stdFinite(residual);
row.("Validation_"+name+"RMS") = rootMeanSquare(residual);
end

function x = getNumericColumn(tableIn, name)
if ismember(name, string(tableIn.Properties.VariableNames))
    x = double(tableIn.(name));
else
    x = nan(height(tableIn),1);
end
end

function value = robustMode(x)
x = x(isfinite(x));
if isempty(x)
    value = NaN;
else
    value = mode(x);
end
end

function value = rootMeanSquare(x)
x = x(isfinite(x));
if isempty(x)
    value = NaN;
else
    value = sqrt(mean(x.^2));
end
end

function value = linearSlope(t, x)
valid = isfinite(t) & isfinite(x);
validTime = t(valid);
if nnz(valid) < 2 || max(validTime)-min(validTime) <= eps
    value = NaN;
else
    coefficients = polyfit(t(valid), x(valid), 1);
    value = coefficients(1);
end
end

function value = sampleSkewness(x)
if numel(x) < 3 || std(x) <= eps
    value = 0;
else
    z = (x-mean(x))/std(x,1);
    value = mean(z.^3);
end
end

function value = sampleKurtosis(x)
if numel(x) < 4 || std(x) <= eps
    value = 0;
else
    z = (x-mean(x))/std(x,1);
    value = mean(z.^4);
end
end

function value = meanFinite(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = mean(x); end
end

function value = stdFinite(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = std(x,0); end
end

function value = maxFinite(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = max(x); end
end

function value = minFinite(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = min(x); end
end

function value = lastMinusFirst(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = x(end)-x(1); end
end
