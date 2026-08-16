%% 计算各阶梯参考平台的稳态电感电流纹波
% 先运行仿真生成 out，再运行本脚本。

outputName          = 'out';
refSignalName       = 'log_Iref';
currentSignalName   = 'log_I_L'; % 未经平均的原始电感电流
switchingFrequency_Hz = 20e3;
analysisWindow_s    = 0.02;      % 每个平台末尾20 ms
referenceLevelTol_A = 1e-3;      % 零参考和参考值比较阈值
referenceSlopeTol_A_per_s = 1;   % 小于该斜率视为参考电流平台
minStableDuration_s = analysisWindow_s;
zeroCurrentThreshold_A = 0.1;    % 小于该值时不计算百分比纹波

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
keep = (t(platformEnd) - t(platformStart)) >= minStableDuration_s;
platformStart = platformStart(keep);
platformEnd = platformEnd(keep);
assert(~isempty(platformStart), '没有检测到持续时间足够的稳定参考平台。');
fprintf('自动检测到 %d 个稳定参考平台。\n', numel(platformStart));

switchingPeriod_s = 1 / switchingFrequency_Hz;
result = table();
allCyclePeakToPeak_A = cell(numel(platformStart), 1);

for p = 1:numel(platformStart)
    platformStartTime = t(platformStart(p));
    platformEndTime = t(platformEnd(p));
    reference_A = median(iRef(platformStart(p):platformEnd(p)), 'omitnan');
    windowEnd = platformEndTime;
    windowStart = max(platformStartTime, windowEnd - analysisWindow_s);
    windowIndex = t >= windowStart & t <= windowEnd;
    assert(nnz(windowIndex) >= 3, '平台 %d 的纹波分析数据不足。', p);

    tWindow = t(windowIndex);
    iWindow = iCurrent(windowIndex);
    meanCurrent_A = mean(iWindow, 'omitnan');
    maxCurrent_A = max(iWindow);
    minCurrent_A = min(iWindow);
    maxAbsCurrent_A = max(abs(iWindow));
    globalPeakToPeak_A = max(iWindow) - min(iWindow);
    rippleComponent = iWindow - meanCurrent_A;
    rippleRMS_A = sqrt(mean(rippleComponent.^2, 'omitnan'));
    maxAbsoluteRipple_A = max(abs(rippleComponent));

    cyclePeakToPeak_A = calculateCycleRipple(tWindow, iWindow, ...
        windowStart, windowEnd, switchingPeriod_s);
    assert(~isempty(cyclePeakToPeak_A), ...
        '平台 %d 没有完整开关周期可用于纹波计算。', p);
    allCyclePeakToPeak_A{p} = cyclePeakToPeak_A;

    meanCyclePeakToPeak_A = mean(cyclePeakToPeak_A);
    maxCyclePeakToPeak_A = max(cyclePeakToPeak_A);
    minCyclePeakToPeak_A = min(cyclePeakToPeak_A);
    if abs(reference_A) > referenceLevelTol_A && ...
            abs(meanCurrent_A) >= zeroCurrentThreshold_A
        ripplePct = 100 * meanCyclePeakToPeak_A / abs(meanCurrent_A);
    else
        ripplePct = NaN;
    end

    platformDuration_s = platformEndTime - platformStartTime;
    row = table(p, platformStartTime, platformEndTime, platformDuration_s, ...
        windowStart, ...
        windowEnd, reference_A, switchingFrequency_Hz, ...
        numel(cyclePeakToPeak_A), meanCurrent_A, maxCurrent_A, minCurrent_A, ...
        maxAbsCurrent_A, globalPeakToPeak_A, ...
        meanCyclePeakToPeak_A, maxCyclePeakToPeak_A, minCyclePeakToPeak_A, ...
        rippleRMS_A, maxAbsoluteRipple_A, ripplePct, 'VariableNames', ...
        {'Platform','PlatformStart_s','PlatformEnd_s','PlatformDuration_s', ...
        'WindowStart_s', ...
        'WindowEnd_s','Reference_A','SwitchingFrequency_Hz','CycleCount', ...
        'MeanCurrent_A','MaxCurrent_A','MinCurrent_A','MaxAbsCurrent_A', ...
        'GlobalPeakToPeak_A','MeanCyclePeakToPeak_A', ...
        'MaxCyclePeakToPeak_A','MinCyclePeakToPeak_A','RippleRMS_A', ...
        'MaxAbsoluteRipple_A','Ripple_pct'});
    result = [result; row]; %#ok<AGROW>
end

validRipplePct = result.Ripple_pct(isfinite(result.Ripple_pct));
meanRipplePct = mean(validRipplePct, 'omitnan');
meanPeakToPeak_A = mean(result.MeanCyclePeakToPeak_A, 'omitnan');
result.MeanAllPlatformsPeakToPeak_A = repmat(meanPeakToPeak_A, height(result), 1);
result.MeanAllPlatformsRipple_pct = repmat(meanRipplePct, height(result), 1);

disp('各阶梯参考平台的稳态电感电流纹波：');
disp(result);
fprintf('全部平台平均峰峰值纹波：%.6g A\n', meanPeakToPeak_A);
fprintf('非零电流平台平均纹波率：%.6g %%\n', meanRipplePct);
assignin('base', 'currentRippleResult', result);
assignin('base', 'currentCyclePeakToPeakByPlatform_A', allCyclePeakToPeak_A);

analysisFolder = fileparts(fileparts(mfilename('fullpath')));
resultFolder = fullfile(analysisFolder, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end
resultFile = fullfile(resultFolder, 'current_ripple.csv');
try
    writetable(result, resultFile);
catch writeError
    if contains(writeError.message, 'Permission denied') || ...
            contains(writeError.message, '无法打开')
        resultFile = fullfile(resultFolder, 'current_ripple_latest.csv');
        writetable(result, resultFile);
        warning('current_ripple.csv可能已在Excel中打开，结果改存为current_ripple_latest.csv。');
    else
        rethrow(writeError);
    end
end
fprintf('电流纹波结果已保存：%s\n', resultFile);
openvar('currentRippleResult');

function cyclePeakToPeak = calculateCycleRipple(t, current, ...
        windowStart, windowEnd, switchingPeriod)
    firstCycleStart = ceil(windowStart / switchingPeriod) * switchingPeriod;
    cycleStart = firstCycleStart:switchingPeriod:(windowEnd - switchingPeriod);
    cyclePeakToPeak = NaN(numel(cycleStart), 1);
    for k = 1:numel(cycleStart)
        cycleIndex = t >= cycleStart(k) & ...
            t <= cycleStart(k) + switchingPeriod;
        if nnz(cycleIndex) >= 2
            cyclePeakToPeak(k) = max(current(cycleIndex)) - min(current(cycleIndex));
        end
    end
    cyclePeakToPeak = cyclePeakToPeak(isfinite(cyclePeakToPeak));
end

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
