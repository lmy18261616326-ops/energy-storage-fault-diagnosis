%% 基于实际稳态值计算电感电流超调量
% 先运行仿真生成 out，再运行本脚本。

outputName        = 'out';
refSignalName     = 'log_Iref';
currentSignalName = 'I_L_avg';
steadyWindow_s    = 0.02;   % 每个平台末尾用于计算实际稳态值的时间
minStableDuration_s = steadyWindow_s;
referenceSlopeTol_A_per_s = 1; % 小于该斜率视为参考电流平台
currentTol_A      = 1e-6;   % 忽略过小的实际稳态值变化

refRaw = getLoggedSignal(outputName, refSignalName);
currentRaw = getLoggedSignal(outputName, currentSignalName);
[tRef, iRefRaw] = signalData(refRaw);
[t, iCurrent] = signalData(currentRaw);
iRef = interp1(tRef, iRefRaw, t, 'previous', 'extrap');

% 根据参考电流斜率自动识别稳定平台，支持各平台持续时间不同。
dt = diff(t);
diRef = diff(iRef);
validStep = dt > 0;
refSlope = inf(size(diRef));
refSlope(validStep) = abs(diRef(validStep) ./ dt(validStep));
stableMask = [true; refSlope <= referenceSlopeTol_A_per_s];
maskEdge = diff([false; stableMask; false]);
platformStart = find(maskEdge == 1);
platformEnd = find(maskEdge == -1) - 1;
platformDuration = t(platformEnd) - t(platformStart);
keepPlatform = platformDuration >= minStableDuration_s;
platformStart = platformStart(keepPlatform);
platformEnd = platformEnd(keepPlatform);
platformDuration = platformDuration(keepPlatform);

numPlatforms = numel(platformStart);
assert(numPlatforms >= 2, ...
    '只检测到 %d 个稳定参考平台，请检查 log_Iref 或斜率阈值。', numPlatforms);
fprintf('自动检测到 %d 个稳定参考平台。\n', numPlatforms);

steadyCurrent = zeros(numPlatforms, 1);
for k = 1:numPlatforms
    segmentStartTime = t(platformStart(k));
    segmentEndTime = t(platformEnd(k));
    steadyStartTime = max(segmentStartTime, segmentEndTime - steadyWindow_s);
    steadyIndex = t >= steadyStartTime & t <= segmentEndTime;
    assert(nnz(steadyIndex) >= 2, '第 %d 段的稳态时间窗内数据不足。', k);
    steadyCurrent(k) = mean(iCurrent(steadyIndex), 'omitnan');
end

result = table();
for k = 2:numPlatforms
    transitionStartIndex = min(platformEnd(k - 1) + 1, platformStart(k));
    transitionTime = t(transitionStartIndex);
    previousSteady = steadyCurrent(k - 1);
    targetSteady = steadyCurrent(k);
    steadyChange = targetSteady - previousSteady;
    if abs(steadyChange) <= currentTol_A
        continue
    end

    % 从参考开始变化到目标平台结束，搜索相对于目标稳态值的峰值或谷值。
    responseIndex = transitionStartIndex:platformEnd(k);

    if steadyChange > 0
        [extremeCurrent, localIndex] = max(iCurrent(responseIndex));
        overshootA = max(0, extremeCurrent - targetSteady);
        direction = "Rising";
    else
        [extremeCurrent, localIndex] = min(iCurrent(responseIndex));
        overshootA = max(0, targetSteady - extremeCurrent);
        direction = "Falling";
    end
    peakIndex = responseIndex(localIndex);
    overshootPct = 100 * overshootA / abs(steadyChange);

    row = table(k - 1, transitionTime, t(platformStart(k)), ...
        t(platformEnd(k)), platformDuration(k), previousSteady, targetSteady, ...
        steadyChange, direction, extremeCurrent, t(peakIndex), overshootA, ...
        overshootPct, 'VariableNames', {'Transition','TransitionStart_s', ...
        'TargetPlatformStart_s','TargetPlatformEnd_s','TargetPlatformDuration_s', ...
        'PreviousSteady_A','TargetSteady_A','SteadyChange_A','Direction', ...
        'ExtremeCurrent_A','ExtremeTime_s','Overshoot_A','Overshoot_pct'});
    result = [result; row]; %#ok<AGROW>
end

meanOvershootPct = mean(result.Overshoot_pct, 'omitnan');
result.MeanOvershoot_pct = repmat(meanOvershootPct, height(result), 1);

disp('基于相邻实际稳态值计算的电感电流超调量：');
disp(result);
fprintf('平均超调率：%.6g %%\n', meanOvershootPct);
assignin('base', 'currentOvershootResult', result);
assignin('base', 'meanCurrentOvershoot_pct', meanOvershootPct);
analysisFolder = fileparts(fileparts(mfilename('fullpath')));
resultFolder = fullfile(analysisFolder, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end
resultFile = fullfile(resultFolder, 'current_overshoot.csv');
writetable(result, resultFile);
fprintf('超调量结果已保存：%s\n', resultFile);
openvar('currentOvershootResult');

function signal = getLoggedSignal(outputName, signalName)
    assert(evalin('base', "exist('" + outputName + "','var') == 1"), ...
        '工作区中找不到 %s，请先运行仿真。', outputName);
    simOut = evalin('base', outputName);
    assert(isa(simOut, 'Simulink.SimulationOutput'), ...
        '%s 不是 Simulink.SimulationOutput 对象。', outputName);
    outputVariables = who(simOut);
    if any(strcmp(outputVariables, signalName))
        signal = simOut.get(signalName);
        return
    end
    if any(strcmp(outputVariables, 'logsout'))
        loggedNames = simOut.logsout.getElementNames;
        if any(strcmp(loggedNames, signalName))
            signal = simOut.logsout.get(signalName).Values;
            return
        end
    end
    error('在 %s 及 %s.logsout 中找不到信号 %s。', ...
        outputName, outputName, signalName);
end

function [t, y] = signalData(signal)
    if isa(signal, 'timeseries')
        t = signal.Time;
        y = signal.Data;
    elseif istimetable(signal)
        t = seconds(signal.Properties.RowTimes - signal.Properties.RowTimes(1));
        y = signal{:, 1};
    elseif isstruct(signal) && isfield(signal, 'time') && isfield(signal, 'signals')
        t = signal.time;
        y = signal.signals.values;
    elseif isnumeric(signal) && size(signal, 2) >= 2
        t = signal(:, 1);
        y = signal(:, 2);
    else
        error('不支持该信号格式，请使用 Timeseries 或 Structure With Time。');
    end
    t = double(t(:));
    y = squeeze(double(y));
    assert(isvector(y), '每个记录变量只能包含一个信号通道。');
    y = y(:);
    assert(numel(t) == numel(y), '信号的时间与数据长度不一致。');
    valid = isfinite(t) & isfinite(y);
    t = t(valid);
    y = y(valid);
    [t, uniqueIndex] = unique(t, 'stable');
    y = y(uniqueIndex);
end
