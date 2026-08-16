%% 基于实际稳态值计算电感电流的 2% 调节时间
% 先运行仿真生成 out，再运行本脚本。

outputName        = 'out';
refSignalName     = 'log_Iref';
currentSignalName = 'I_L_avg';
expectedPlatformDuration_s = 0.4;  % 仅用于检查，不再用于强制切段
steadyWindow_s    = 0.02;   % 每个平台末尾用于计算实际稳态值的时间
minStableDuration_s = steadyWindow_s; % 短于该时间的稳定段不算平台
referenceSlopeTol_A_per_s = 1; % 小于该斜率视为参考电流平台
platformDurationTolerance_s = 5e-3; % 平台时长检查允许误差
settlingBandPct   = 2;      % 调节带，相对于相邻实际稳态值之差
currentTol_A      = 1e-6;   % 忽略过小的实际稳态电流变化

refRaw = getLoggedSignal(outputName, refSignalName);
currentRaw = getLoggedSignal(outputName, currentSignalName);
[tRef, iRefRaw] = signalData(refRaw);
[t, iCurrent] = signalData(currentRaw);
iRef = interp1(tRef, iRefRaw, t, 'previous', 'extrap');

% 根据参考电流斜率自动识别稳定平台。这样仿真末尾即使长时间保持
% 同一个参考值，也只会被识别为一个平台，而不会按总时长重复切段。
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
for k = 1:numPlatforms
    referenceLevel = median(iRef(platformStart(k):platformEnd(k)), 'omitnan');
    fprintf('平台 %d: %.6g~%.6g s, 时长 %.6g s, 参考 %.6g A\n', ...
        k, t(platformStart(k)), t(platformEnd(k)), ...
        platformDuration(k), referenceLevel);
end

% 最后一个平台可能一直持续到仿真结束，因此只对非末尾平台提示时长偏差。
if numPlatforms > 1
    durationError = abs(platformDuration(1:end-1) - expectedPlatformDuration_s);
    if any(durationError > platformDurationTolerance_s)
        warning(['部分平台时长不是预期的 %.3f s。若这是有意设置，可修改 ' ...
            'expectedPlatformDuration_s；平台识别本身不受该参数影响。'], ...
            expectedPlatformDuration_s);
    end
end

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
    % 从上一平台结束后的第一个采样点开始计时，包含参考斜坡过程。
    transitionStartIndex = min(platformEnd(k - 1) + 1, platformStart(k));
    transitionTime = t(transitionStartIndex);
    previousSteady = steadyCurrent(k - 1);
    targetSteady = steadyCurrent(k);
    steadyChange = targetSteady - previousSteady;
    if abs(steadyChange) <= currentTol_A
        continue
    end

    bandA = settlingBandPct / 100 * abs(steadyChange);
    responseIndex = transitionStartIndex:platformEnd(k);
    assert(~isempty(responseIndex), '第 %d 次切换没有可用响应数据。', k - 1);

    outsideBand = abs(iCurrent(responseIndex) - targetSteady) > bandA;
    lastOutside = find(outsideBand, 1, 'last');
    if isempty(lastOutside)
        settled = true;
        settlingIndex = responseIndex(1);
    elseif lastOutside < numel(responseIndex)
        settled = true;
        settlingIndex = responseIndex(lastOutside + 1);
    else
        settled = false;
        settlingIndex = NaN;
    end

    if settled
        settlingTime = t(settlingIndex) - transitionTime;
        settlingAbsoluteTime = t(settlingIndex);
    else
        settlingTime = NaN;
        settlingAbsoluteTime = NaN;
    end

    if steadyChange > 0
        direction = "Rising";
    else
        direction = "Falling";
    end

    row = table(k - 1, transitionTime, previousSteady, targetSteady, ...
        steadyChange, direction, bandA, settled, settlingAbsoluteTime, ...
        settlingTime, 'VariableNames', {'Transition','TransitionStart_s', ...
        'PreviousSteady_A','TargetSteady_A','SteadyChange_A','Direction', ...
        'Band2pct_A','Settled','SettlingAbsoluteTime_s','SettlingTime_2pct_s'});
    result = [result; row]; %#ok<AGROW>
end

meanSettlingTime = mean(result.SettlingTime_2pct_s, 'omitnan');
result.MeanSettlingTime_2pct_s = repmat(meanSettlingTime, height(result), 1);

disp('基于实际稳态值计算的电感电流 2% 调节时间：');
disp(result);
fprintf('平均 2%% 调节时间：%.6g s\n', meanSettlingTime);
assignin('base', 'currentSettlingTime2pctResult', result);
assignin('base', 'meanCurrentSettlingTime2pct_s', meanSettlingTime);

analysisFolder = fileparts(fileparts(mfilename('fullpath')));
resultFolder = fullfile(analysisFolder, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end
resultFile = fullfile(resultFolder, 'current_settling_time_2pct.csv');
writetable(result, resultFile);
fprintf('2%% 调节时间结果已保存：%s\n', resultFile);
openvar('currentSettlingTime2pctResult');

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
