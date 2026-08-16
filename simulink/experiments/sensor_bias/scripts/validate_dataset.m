function report = validate_dataset(rawDataset, featureDataset, ...
    simulationCases, failedRuns, cfg, featureColumns)
%VALIDATE_DATASET 检查数据质量、生成文字报告和诊断图。
%   REPORT = VALIDATE_DATASET(RAW, FEATURES, CASES, FAILED, CFG,
%   FEATURECOLUMNS) 检查字段、非有限值、时间、标签、范围、类别平衡、
%   常数/近零方差、泄漏、重复窗口及失败运行，并把报告和 PNG 图保存到
%   cfg.output 指定目录。
%
%   输入:
%       rawDataset      - 合并后的原始数据表。
%       featureDataset  - 合并后的窗口特征表。
%       simulationCases - 全部仿真配置。
%       failedRuns      - 失败日志表。
%       cfg             - signal_config 配置。
%       featureColumns  - 允许进入模型的候选特征列。
%   输出:
%       report          - 质量统计和问题列表结构体。

arguments
    rawDataset table
    featureDataset table
    simulationCases table
    failedRuns table
    cfg (1,1) struct = signal_config()
    featureColumns string = strings(0,1)
end

issues = strings(0,1);
rawVariables = string(rawDataset.Properties.VariableNames);
featureVariables = string(featureDataset.Properties.VariableNames);

requiredRaw = [ ...
    "Time","RunID","OperatingPointID","ModeCommand","ConverterEnable", ...
    "Iref","VbusRef","IL_meas","Ibat_meas","Vbus_meas","Vbat_meas", ...
    "Iload_meas","Ibus_source","SOC_est","CurrentError","VoltageError","DutyRaw", ...
    "DutyApplied","PIiOut","PIiIntegral","PIvOut","SatFlag", ...
    "S1GateCmd","S2GateCmd","S1GateDuty","S2GateDuty", ...
    "IL_true","Ibat_true","Vbus_true", ...
    "Vbat_true","SOC_true","Psource_meas","Pload_meas", ...
    "Pstored_meas","PowerBalanceResidual", ...
    "FaultActive","TransitionWindow", ...
    "FaultObservable","ScenarioFaultID","ActiveFaultID", ...
    "ObservableFaultID","IsTrainingEligible"];
missingRaw = setdiff(requiredRaw, rawVariables);
if ~isempty(missingRaw)
    issues(end+1,1) = "缺少必要原始字段: " + strjoin(missingRaw, ", ");
end
if ~ismember("WindowFaultID", featureVariables)
    issues(end+1,1) = "特征表缺少 WindowFaultID。";
end

[nanCounts, infCounts, missingRates] = finiteValueSummary(rawDataset);
if sum(nanCounts.Count) > 0
    issues(end+1,1) = compose( ...
        "原始数据中存在 NaN，共 %d 个（可选缺失信号允许用 NaN 占位）。", ...
        sum(nanCounts.Count));
end
if sum(infCounts.Count) > 0
    issues(end+1,1) = "原始数据中存在 Inf。";
end

runIDs = unique(string(rawDataset.RunID), "stable");
nonIncreasingRuns = strings(0,1);
duplicateTimeRuns = strings(0,1);
abnormalLengthRuns = strings(0,1);
for runID = runIDs'
    runRows = rawDataset(string(rawDataset.RunID) == runID,:);
    time = runRows.Time;
    if any(diff(time) < 0)
        nonIncreasingRuns(end+1,1) = runID; %#ok<AGROW>
    end
    if numel(unique(time)) ~= numel(time)
        duplicateTimeRuns(end+1,1) = runID; %#ok<AGROW>
    end
    caseRow = simulationCases(string(simulationCases.RunID) == runID,:);
    if ~isempty(caseRow)
        expected = floor(caseRow.SimulationStopTime(1)/cfg.sampleTime)+1;
        if abs(height(runRows)-expected) > 2
            abnormalLengthRuns(end+1,1) = runID; %#ok<AGROW>
        end
    end
end
if ~isempty(nonIncreasingRuns)
    issues(end+1,1) = "时间不递增的 RunID: " + ...
        strjoin(nonIncreasingRuns, ", ");
end
if ~isempty(duplicateTimeRuns)
    issues(end+1,1) = "存在重复时间点的 RunID: " + ...
        strjoin(duplicateTimeRuns, ", ");
end
if ~isempty(abnormalLengthRuns)
    issues(end+1,1) = "采样长度异常的 RunID: " + ...
        strjoin(abnormalLengthRuns, ", ");
end

invalidInactive = nnz(rawDataset.FaultActive == 0 & ...
    rawDataset.ActiveFaultID ~= 0);
invalidActive = nnz(rawDataset.FaultActive ~= 0 & ...
    rawDataset.ActiveFaultID ~= rawDataset.ScenarioFaultID);
invalidHealthy = nnz(rawDataset.ScenarioFaultID == 0 & ...
    rawDataset.FaultActive ~= 0);
if invalidInactive > 0
    issues(end+1,1) = compose( ...
        "FaultActive=0 但 ActiveFaultID 非零: %d 行。", invalidInactive);
end
if invalidActive > 0
    issues(end+1,1) = compose( ...
        "FaultActive=1 但 ActiveFaultID 不等于 ScenarioFaultID: %d 行。", ...
        invalidActive);
end
if invalidHealthy > 0
    issues(end+1,1) = compose( ...
        "健康场景错误激活故障: %d 行。", invalidHealthy);
end

invalidObservable = nnz(rawDataset.FaultObservable ~= 0 & ...
    rawDataset.FaultActive == 0);
invalidObservableID0 = nnz(rawDataset.FaultObservable == 0 & ...
    rawDataset.ObservableFaultID ~= 0);
invalidObservableID1 = nnz(rawDataset.FaultObservable ~= 0 & ...
    rawDataset.ObservableFaultID ~= rawDataset.ScenarioFaultID);
if invalidObservable > 0 || invalidObservableID0 > 0 || ...
        invalidObservableID1 > 0
    issues(end+1,1) = compose( ...
        ["开路可观测标签不一致：未激活却可观测=%d，" + ...
         "不可观测但ID非零=%d，可观测但ID错误=%d。"], ...
        invalidObservable, invalidObservableID0, invalidObservableID1);
end

if ismember("IsTrainingEligible", featureVariables)
    invalidEligibleTransition = nnz( ...
        featureDataset.IsTransitionWindow ~= 0 & ...
        featureDataset.IsTrainingEligible ~= 0);
    invalidEligibleFaultPreface = nnz( ...
        featureDataset.ScenarioFaultID ~= 0 & ...
        featureDataset.WindowFaultID == 0 & ...
        featureDataset.IsTrainingEligible ~= 0);
    if invalidEligibleTransition > 0
        issues(end+1,1) = compose( ...
            "过渡窗口错误进入训练: %d 个。", ...
            invalidEligibleTransition);
    end
    if invalidEligibleFaultPreface > 0
        issues(end+1,1) = compose( ...
            "故障场景中的故障前/不可观测窗口错误进入训练: %d 个。", ...
            invalidEligibleFaultPreface);
    end
end

dutyOutOfRange = nnz(rawDataset.DutyApplied < cfg.dutyLimits(1)-1e-9 | ...
    rawDataset.DutyApplied > cfg.dutyLimits(2)+1e-9);
invalidModes = setdiff(unique(rawDataset.ModeCommand(isfinite( ...
    rawDataset.ModeCommand))), cfg.allowedModes);
if dutyOutOfRange > 0
    issues(end+1,1) = compose("DutyApplied 越界: %d 行。", ...
        dutyOutOfRange);
end
if ~isempty(invalidModes)
    issues(end+1,1) = "非法 ModeCommand: " + ...
        strjoin(string(invalidModes), ", ");
end

% 负载电流应与母线电阻负载和受控功率负载同量级。该检查可发现误读
% Multimeter 的 I_Rh 支路电流。
iloadPlausibilityRatio = calculateIloadPlausibility(rawDataset);
if isfinite(iloadPlausibilityRatio) && iloadPlausibilityRatio > 0.20
    issues(end+1,1) = compose( ...
        "Iload_meas 与配置负载不符的运行比例为 %.1f%%。", ...
        100*iloadPlausibilityRatio);
end

currentDirectionRatio = calculateCurrentDirectionRatio(rawDataset, cfg);
if isfinite(currentDirectionRatio) && currentDirectionRatio > 1.25
    issues(end+1,1) = compose( ...
        ["配置方向下的电流一致性 RMS 是相反方向的 %.2f 倍，" + ...
         "请复核 currentDirectionFactor。"], currentDirectionRatio);
end

if ismember("IsTrainingEligible", featureVariables)
    eligibleFeatures = featureDataset( ...
        featureDataset.IsTrainingEligible ~= 0,:);
else
    eligibleFeatures = featureDataset;
end
presentClasses = unique(eligibleFeatures.WindowFaultID);
missingClasses = setdiff((0:4)', presentClasses);
if ~isempty(missingClasses)
    issues(end+1,1) = "窗口中缺少类别: " + ...
        strjoin(string(missingClasses), ", ");
end

runFaultPairs = unique(featureDataset(:,["RunID","ScenarioFaultID"]));
runCountsByFault = groupcounts(runFaultPairs, "ScenarioFaultID");
runCountNames = runCountsByFault.Properties.VariableNames;
runCountNames{strcmp(runCountNames,"GroupCount")} = 'RunIDCount';
runCountsByFault.Properties.VariableNames = runCountNames;
windowCountsByFault = groupcounts(featureDataset, "WindowFaultID");
windowCountNames = windowCountsByFault.Properties.VariableNames;
windowCountNames{strcmp(windowCountNames,"GroupCount")} = 'WindowCount';
windowCountsByFault.Properties.VariableNames = windowCountNames;
modeCounts = groupcounts(featureDataset, "ModeCommand");
modeCountNames = modeCounts.Properties.VariableNames;
modeCountNames{strcmp(modeCountNames,"GroupCount")} = 'WindowCount';
modeCounts.Properties.VariableNames = modeCountNames;
eligibleWindowCountsByFault = groupcounts( ...
    eligibleFeatures, "WindowFaultID");
eligibleCountNames = eligibleWindowCountsByFault.Properties.VariableNames;
eligibleCountNames{strcmp(eligibleCountNames,"GroupCount")} = ...
    'EligibleWindowCount';
eligibleWindowCountsByFault.Properties.VariableNames = eligibleCountNames;
if any(runCountsByFault.RunIDCount < 2)
    issues(end+1,1) = "至少一个类别少于 2 个独立 RunID。";
end

[constantFeatures, nearZeroVarianceFeatures] = ...
    findLowInformationFeatures(featureDataset, featureColumns);
if ~isempty(constantFeatures)
    issues(end+1,1) = "常数特征: " + ...
        strjoin(constantFeatures, ", ");
end

leakageColumns = findExactLabelLeakage( ...
    featureDataset, featureColumns, "WindowFaultID");
if ~isempty(leakageColumns)
    issues(end+1,1) = "与标签完全相同的泄漏字段: " + ...
        strjoin(leakageColumns, ", ");
end

duplicateWindowCount = countDuplicateWindows( ...
    featureDataset, featureColumns);
if duplicateWindowCount > 0
    issues(end+1,1) = compose("高度/完全重复窗口: %d 个。", ...
        duplicateWindowCount);
end

invalidDutyResidual = false;
for name = ["DutyResidualRMS","DutyLimitResidualRMS"]
    if ismember(name, featureVariables)
        values = featureDataset.(name);
        finiteValues = abs(values(isfinite(values)));
        if ~isempty(finiteValues) && max(finiteValues) < 1e-12
            invalidDutyResidual = true;
            issues(end+1,1) = name + ...
                " 全部仅为浮点误差量级，不应作为有效模型输入。";
        end
    end
end

report = struct();
report.TotalRuns = height(simulationCases);
report.SuccessRuns = numel(runIDs);
report.FailedRuns = height(failedRuns);
report.RunCountsByFault = runCountsByFault;
report.WindowCountsByFault = windowCountsByFault;
report.EligibleWindowCountsByFault = eligibleWindowCountsByFault;
report.ModeCounts = modeCounts;
report.TransitionWindows = nnz(featureDataset.IsTransitionWindow ~= 0);
report.HealthyWindows = nnz(featureDataset.WindowFaultID == 0);
report.FaultWindows = nnz(featureDataset.WindowFaultID ~= 0);
report.EligibleWindows = height(eligibleFeatures);
report.IneligibleWindows = height(featureDataset)-height(eligibleFeatures);
report.IloadImplausibleRunRatio = iloadPlausibilityRatio;
report.CurrentDirectionRMSRatio = currentDirectionRatio;
report.TopMissingFields = sortrows(missingRates, "MissingRate", "descend");
report.TopMissingFields = report.TopMissingFields( ...
    1:min(10,height(report.TopMissingFields)),:);
report.ConstantFeatures = constantFeatures;
report.NearZeroVarianceFeatures = nearZeroVarianceFeatures;
report.LabelLeakageColumns = leakageColumns;
report.DuplicateWindowCount = duplicateWindowCount;
report.InvalidDutyResidual = invalidDutyResidual;
report.AbnormalLengthRuns = abnormalLengthRuns;
report.Issues = issues;
report.NaNCounts = nanCounts;
report.InfCounts = infCounts;

writeTextReport(report, cfg);
createDiagnosticFigures(rawDataset, featureDataset, cfg);
end

function [nanSummary, infSummary, missingSummary] = finiteValueSummary(data)
variables = string(data.Properties.VariableNames);
numericMask = false(size(variables));
nanCount = zeros(numel(variables),1);
infCount = zeros(numel(variables),1);
for k = 1:numel(variables)
    value = data.(variables(k));
    if isnumeric(value) || islogical(value)
        numericMask(k) = true;
        nanCount(k) = nnz(isnan(double(value)));
        infCount(k) = nnz(isinf(double(value)));
    end
end
variables = variables(numericMask);
nanCount = nanCount(numericMask);
infCount = infCount(numericMask);
nanSummary = table(variables(:), nanCount, ...
    'VariableNames', {'Field','Count'});
infSummary = table(variables(:), infCount, ...
    'VariableNames', {'Field','Count'});
missingSummary = table(variables(:), nanCount./max(height(data),1), ...
    'VariableNames', {'Field','MissingRate'});
end

function [constants, nearZero] = findLowInformationFeatures(data, columns)
constants = strings(0,1);
nearZero = strings(0,1);
for name = columns(:)'
    if ~ismember(name, string(data.Properties.VariableNames))
        continue;
    end
    values = double(data.(name));
    values = values(isfinite(values));
    if isempty(values) || numel(unique(values)) <= 1
        constants(end+1,1) = name; %#ok<AGROW>
    elseif var(values,1) < 1e-12
        nearZero(end+1,1) = name; %#ok<AGROW>
    end
end
end

function leakage = findExactLabelLeakage(data, columns, labelName)
leakage = strings(0,1);
label = double(data.(labelName));
for name = columns(:)'
    values = double(data.(name));
    valid = isfinite(values) & isfinite(label);
    if any(valid) && all(values(valid) == label(valid)) && ...
            nnz(valid) == numel(label)
        leakage(end+1,1) = name; %#ok<AGROW>
    end
end
end

function count = countDuplicateWindows(data, columns)
usable = strings(0,1);
for name = columns(:)'
    if ismember(name, string(data.Properties.VariableNames)) && ...
            isnumeric(data.(name))
        usable(end+1,1) = name; %#ok<AGROW>
    end
end
if isempty(usable) || height(data) < 2
    count = 0;
    return;
end
matrix = data{:,cellstr(usable)};
matrix(~isfinite(matrix)) = realmax("double");
[~, uniqueRows] = unique(matrix, "rows", "stable");
count = height(data) - numel(uniqueRows);
end

function ratio = calculateIloadPlausibility(raw)
ratio = NaN;
required = ["RunID","Time","Iload_meas","Vbus_meas", ...
    "VbusRef","Rload","Pload"];
if ~isempty(setdiff(required, ...
        string(raw.Properties.VariableNames)))
    return;
end
runIDs = unique(string(raw.RunID), "stable");
bad = false(numel(runIDs),1);
usable = false(numel(runIDs),1);
for k = 1:numel(runIDs)
    one = raw(string(raw.RunID) == runIDs(k),:);
    mask = one.Time >= min(0.1, max(one.Time)/2) & ...
        isfinite(one.Iload_meas) & isfinite(one.Vbus_meas) & ...
        isfinite(one.Rload) & one.Rload > 0 & ...
        isfinite(one.Pload) & isfinite(one.VbusRef);
    if nnz(mask) < 10
        continue;
    end
    measured = median(one.Iload_meas(mask), "omitnan");
    expectedTrace = one.Vbus_meas(mask)./one.Rload(mask) + ...
        one.Pload(mask)./max(abs(one.VbusRef(mask)), eps);
    expected = median(expectedTrace, "omitnan");
    tolerance = max(0.5, 0.30*max(abs(expected),1));
    bad(k) = abs(measured-expected) > tolerance;
    usable(k) = true;
end
if any(usable)
    ratio = mean(bad(usable));
end
end

function ratio = calculateCurrentDirectionRatio(raw, cfg)
ratio = NaN;
required = ["IL_meas","Ibat_meas","ScenarioFaultID"];
if ~isempty(setdiff(required, ...
        string(raw.Properties.VariableNames)))
    return;
end
mask = raw.ScenarioFaultID == 0 & isfinite(raw.IL_meas) & ...
    isfinite(raw.Ibat_meas);
if nnz(mask) < 10
    return;
end
configured = raw.IL_meas(mask) - ...
    cfg.currentDirectionFactor .* raw.Ibat_meas(mask);
opposite = raw.IL_meas(mask) + ...
    cfg.currentDirectionFactor .* raw.Ibat_meas(mask);
configuredRMS = sqrt(mean(configured.^2));
oppositeRMS = sqrt(mean(opposite.^2));
ratio = configuredRMS / max(oppositeRMS, eps);
end

function writeTextReport(report, cfg)
path = fullfile(cfg.output.combined, "dataset_report.txt");
[fileID, message] = fopen(path, "w", "n", "UTF-8");
if fileID < 0
    error("faultdataset:ReportOpenFailed", ...
        "无法写入质量报告: %s", message);
end
cleanup = onCleanup(@() fclose(fileID));
fprintf(fileID, "故障诊断数据集质量报告\n");
fprintf(fileID, "生成时间: %s\n\n", string(datetime("now")));
fprintf(fileID, "总仿真次数: %d\n", report.TotalRuns);
fprintf(fileID, "成功次数: %d\n", report.SuccessRuns);
fprintf(fileID, "失败次数: %d\n", report.FailedRuns);
fprintf(fileID, "过渡窗口数量: %d\n", report.TransitionWindows);
fprintf(fileID, "健康窗口数量: %d\n", report.HealthyWindows);
fprintf(fileID, "故障窗口数量: %d\n", report.FaultWindows);
fprintf(fileID, "训练合格窗口数量: %d\n", report.EligibleWindows);
fprintf(fileID, "保留但默认不训练的窗口数量: %d\n", ...
    report.IneligibleWindows);
fprintf(fileID, "负载电流不可信运行比例: %.4f\n", ...
    report.IloadImplausibleRunRatio);
fprintf(fileID, "配置/相反电流方向残差 RMS 比: %.4f\n", ...
    report.CurrentDirectionRMSRatio);
fprintf(fileID, "重复窗口数量: %d\n\n", report.DuplicateWindowCount);

printTable(fileID, "每类故障的独立 RunID 数量", ...
    report.RunCountsByFault);
printTable(fileID, "每类故障的窗口数量", ...
    report.WindowCountsByFault);
printTable(fileID, "每类故障的训练合格窗口数量", ...
    report.EligibleWindowCountsByFault);
printTable(fileID, "每种模式的窗口数量", report.ModeCounts);
printTable(fileID, "缺失率最高的字段", report.TopMissingFields);

fprintf(fileID, "常数特征:\n%s\n\n", ...
    emptyText(report.ConstantFeatures));
fprintf(fileID, "近零方差特征:\n%s\n\n", ...
    emptyText(report.NearZeroVarianceFeatures));
fprintf(fileID, "标签泄漏字段:\n%s\n\n", ...
    emptyText(report.LabelLeakageColumns));
fprintf(fileID, "数据质量问题:\n");
if isempty(report.Issues)
    fprintf(fileID, "无。\n");
else
    for k = 1:numel(report.Issues)
        fprintf(fileID, "- %s\n", report.Issues(k));
    end
end
clear cleanup
end

function printTable(fileID, titleText, data)
fprintf(fileID, "%s:\n", titleText);
if isempty(data)
    fprintf(fileID, "无。\n\n");
    return;
end
variables = string(data.Properties.VariableNames);
fprintf(fileID, "%s\n", strjoin(variables, "\t"));
for rowIndex = 1:height(data)
    values = strings(1,numel(variables));
    for k = 1:numel(variables)
        values(k) = string(data.(variables(k))(rowIndex));
    end
    fprintf(fileID, "%s\n", strjoin(values, "\t"));
end
fprintf(fileID, "\n");
end

function text = emptyText(values)
if isempty(values), text = "无。"; else, text = strjoin(values, ", "); end
end

function createDiagnosticFigures(raw, features, cfg)
figureFolder = cfg.output.figures;
saveBarFigure(features.WindowFaultID, ...
    "每类故障窗口数量", "WindowFaultID", "窗口数", ...
    fullfile(figureFolder, "fault_window_counts.png"));

pairs = unique(features(:,["RunID","ScenarioFaultID"]));
[groups, ids] = findgroups(pairs.ScenarioFaultID);
counts = splitapply(@numel, pairs.RunID, groups);
saveBarValues(ids, counts, "每类故障的独立 RunID 数量", ...
    "ScenarioFaultID", "RunID 数", ...
    fullfile(figureFolder, "fault_run_counts.png"));

saveBarFigure(features.ModeCommand, ...
    "不同模式的窗口数量", "ModeCommand", "窗口数", ...
    fullfile(figureFolder, "mode_window_counts.png"));

candidateRuns = unique(string(raw.RunID(raw.ScenarioFaultID ~= 0)), "stable");
if isempty(candidateRuns)
    candidateRuns = unique(string(raw.RunID), "stable");
end
if isempty(candidateRuns)
    return;
end
selected = raw(string(raw.RunID) == candidateRuns(1),:);
faultTime = selected.FaultStartTime(1);

fig = figure("Visible","off","Color","w");
plot(selected.Time, selected.IL_true, "LineWidth",1.0); hold on;
plot(selected.Time, selected.IL_meas, "LineWidth",1.0);
plot(selected.Time, selected.Iref, "LineWidth",1.0);
stairs(selected.Time, selected.FaultActive, "LineWidth",1.0);
xline(faultTime, "--r", "Fault start");
grid on; xlabel("Time (s)"); ylabel("Signal value");
title("Inductor current and fault activation");
legend("IL true","IL measured","Iref","FaultActive", ...
    "Location","best");
exportgraphics(fig, fullfile(figureFolder, ...
    "sample_current_fault_trace.png"), "Resolution", 160);
close(fig);

fig = figure("Visible","off","Color","w");
plot(selected.Time, selected.DutyApplied, "LineWidth",1.0); hold on;
plot(selected.Time, selected.PIiOut, "LineWidth",1.0);
plot(selected.Time, selected.PIiIntegral, "LineWidth",1.0);
stairs(selected.Time, selected.SatFlag, "LineWidth",1.0);
xline(faultTime, "--r", "Fault start");
grid on; xlabel("Time (s)"); ylabel("Signal value");
title("Controller compensation signals");
legend("DutyApplied","PIiOut","PIiIntegral","SatFlag", ...
    "Location","best");
exportgraphics(fig, fullfile(figureFolder, ...
    "sample_controller_trace.png"), "Resolution", 160);
close(fig);

fig = figure("Visible","off","Color","w");
plot(selected.Time, selected.IL_meas-selected.IL_true, ...
    "LineWidth",1.0); hold on;
xline(faultTime, "--r", "Fault start");
grid on; xlabel("Time (s)"); ylabel("IL measured - IL true (A)");
title("Inductor current sensor residual");
exportgraphics(fig, fullfile(figureFolder, ...
    "sample_IL_sensor_residual.png"), "Resolution", 160);
close(fig);
end

function saveBarFigure(values, titleText, xText, yText, path)
[groups, ids] = findgroups(values);
counts = splitapply(@numel, values, groups);
saveBarValues(ids, counts, titleText, xText, yText, path);
end

function saveBarValues(ids, counts, titleText, xText, yText, path)
fig = figure("Visible","off","Color","w");
bar(categorical(string(ids)), counts);
grid on; title(titleText); xlabel(xText); ylabel(yText);
exportgraphics(fig, path, "Resolution", 160);
close(fig);
end
