%% 检测电感平均电流的震荡衰减特性
% 先运行仿真生成 out，再运行本脚本。

outputName          = 'out';
refSignalName       = 'log_Iref';
currentSignalName   = 'I_L_avg';
steadyWindow_s      = 0.02;     % 每个平台末尾用于估计实际稳态值
minStableDuration_s = steadyWindow_s;
referenceSlopeTol_A_per_s = 1; % 小于该斜率视为参考电流平台
rateLimit_A_per_s   = 200;      % Rate Limiter 变化率，单位 A/s
minPeakDistance_s   = 5e-4;     % 相邻包络峰的最短间隔，单位 s
noiseMultiplier     = 3;        % 峰值至少高于稳态标准差的倍数
minPeakStepFraction = 0.002;    % 峰值至少达到实际稳态变化量的比例
minPeaksForFit      = 3;        % 指数衰减拟合所需最少峰值数

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
steadyStd = zeros(numPlatforms, 1);
for k = 1:numPlatforms
    segmentStartTime = t(platformStart(k));
    segmentEndTime = t(platformEnd(k));
    steadyStartTime = max(segmentStartTime, segmentEndTime - steadyWindow_s);
    steadyIndex = t >= steadyStartTime & t <= segmentEndTime;
    assert(nnz(steadyIndex) >= 3, '第 %d 段的稳态时间窗内数据不足。', k);
    steadyCurrent(k) = mean(iCurrent(steadyIndex), 'omitnan');
    steadyStd(k) = std(iCurrent(steadyIndex), 'omitnan');
end

result = table();
for k = 2:numPlatforms
    transitionStartIndex = min(platformEnd(k - 1) + 1, platformStart(k));
    transitionTime = t(transitionStartIndex);
    previousSteady = steadyCurrent(k - 1);
    targetSteady = steadyCurrent(k);
    steadyChange = targetSteady - previousSteady;
    if abs(steadyChange) <= 1e-6
        continue
    end

    % 排除 Rate Limiter 的理论斜坡时间，再分析围绕新稳态值的震荡。
    rampTime = abs(steadyChange) / rateLimit_A_per_s;
    analysisStart = max(transitionTime + rampTime, t(platformStart(k)));
    analysisEnd = t(platformEnd(k));
    analysisIndex = find(t >= analysisStart & t <= analysisEnd);
    if numel(analysisIndex) < 3
        warning('第 %d 次切换的目标平台太短，已跳过震荡分析。', k - 1);
        continue
    end

    errorEnvelope = abs(iCurrent(analysisIndex) - targetSteady);
    peakThreshold = max(noiseMultiplier * steadyStd(k), ...
        minPeakStepFraction * abs(steadyChange));
    localPeaks = findLocalEnvelopePeaks(t(analysisIndex), errorEnvelope, ...
        peakThreshold, minPeakDistance_s);
    peakIndex = analysisIndex(localPeaks);
    peakAmplitude = errorEnvelope(localPeaks);
    peakTime = t(peakIndex);
    numPeaks = numel(peakAmplitude);

    initialPeakA = NaN;
    finalPeakA = NaN;
    finalToInitialRatio = NaN;
    decayRate_per_s = NaN;
    decayTimeConstant_s = NaN;
    oscillationFrequency_Hz = NaN;
    fitR2 = NaN;
    isDecaying = false;
    decayStatus = "InsufficientPeaks";

    if numPeaks >= 1
        initialPeakA = peakAmplitude(1);
        finalPeakA = peakAmplitude(end);
        finalToInitialRatio = finalPeakA / initialPeakA;
    end
    if numPeaks >= minPeaksForFit
        fitTime = peakTime - peakTime(1);
        fitCoeff = polyfit(fitTime, log(peakAmplitude), 1);
        fittedLogAmplitude = polyval(fitCoeff, fitTime);
        residual = log(peakAmplitude) - fittedLogAmplitude;
        totalDeviation = log(peakAmplitude) - mean(log(peakAmplitude));
        fitR2 = 1 - sum(residual.^2) / max(sum(totalDeviation.^2), eps);
        decayRate_per_s = -fitCoeff(1);
        if decayRate_per_s > 0
            decayTimeConstant_s = 1 / decayRate_per_s;
        end
        peakInterval = median(diff(peakTime));
        oscillationFrequency_Hz = 1 / (2 * peakInterval); % 绝对误差峰间隔约为半周期
        isDecaying = decayRate_per_s > 0 && finalToInitialRatio < 1;
        if isDecaying
            decayStatus = "Decaying";
        else
            decayStatus = "NonDecaying";
        end
    end

    if steadyChange > 0
        direction = "Rising";
    else
        direction = "Falling";
    end

    row = table(k - 1, transitionTime, t(platformStart(k)), ...
        t(platformEnd(k)), platformDuration(k), previousSteady, targetSteady, ...
        steadyChange, direction, rampTime, analysisStart, peakThreshold, ...
        numPeaks, initialPeakA, finalPeakA, finalToInitialRatio, ...
        decayRate_per_s, decayTimeConstant_s, oscillationFrequency_Hz, ...
        fitR2, isDecaying, decayStatus, 'VariableNames', {'Transition','TransitionStart_s', ...
        'TargetPlatformStart_s','TargetPlatformEnd_s','TargetPlatformDuration_s', ...
        'PreviousSteady_A','TargetSteady_A','SteadyChange_A','Direction', ...
        'RateLimiterRamp_s','AnalysisStart_s','PeakThreshold_A','NumPeaks', ...
        'InitialPeak_A','FinalPeak_A','FinalToInitialRatio','DecayRate_per_s', ...
        'DecayTimeConstant_s','OscillationFrequency_Hz','FitR2','IsDecaying', ...
        'DecayStatus'});
    result = [result; row]; %#ok<AGROW>
end

validFit = isfinite(result.DecayRate_per_s);
validDecay = result.IsDecaying & validFit;
detectableOscillation_pct = 100 * nnz(validFit) / height(result);
decaySuccessAmongFitted_pct = 100 * nnz(validDecay) / max(nnz(validFit), 1);
meanDecayRate_per_s = mean(result.DecayRate_per_s(validDecay), 'omitnan');
meanDecayTimeConstant_s = mean(result.DecayTimeConstant_s(validDecay), 'omitnan');
result.DetectableOscillation_pct = repmat(detectableOscillation_pct, height(result), 1);
result.DecaySuccessAmongFitted_pct = repmat(decaySuccessAmongFitted_pct, height(result), 1);
result.MeanDecayRate_per_s = repmat(meanDecayRate_per_s, height(result), 1);
result.MeanDecayTimeConstant_s = repmat(meanDecayTimeConstant_s, height(result), 1);

disp('电感平均电流震荡衰减检测结果：');
disp(result);
fprintf('具有足够峰值、可进行衰减拟合的切换比例：%.3f %%\n', detectableOscillation_pct);
fprintf('可拟合切换中的衰减震荡比例：%.3f %%\n', decaySuccessAmongFitted_pct);
fprintf('平均指数衰减率：%.6g 1/s\n', meanDecayRate_per_s);
fprintf('平均衰减时间常数：%.6g s\n', meanDecayTimeConstant_s);
assignin('base', 'currentOscillationDecayResult', result);

analysisFolder = fileparts(fileparts(mfilename('fullpath')));
resultFolder = fullfile(analysisFolder, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end
resultFile = fullfile(resultFolder, 'current_oscillation_decay.csv');
writetable(result, resultFile);
fprintf('震荡衰减结果已保存：%s\n', resultFile);
openvar('currentOscillationDecayResult');

function peakIndex = findLocalEnvelopePeaks(t, amplitude, threshold, minDistance)
    candidate = find(amplitude(2:end-1) >= amplitude(1:end-2) & ...
        amplitude(2:end-1) > amplitude(3:end)) + 1;
    candidate = candidate(amplitude(candidate) >= threshold);
    peakIndex = zeros(0, 1);
    for n = 1:numel(candidate)
        idx = candidate(n);
        if isempty(peakIndex) || t(idx) - t(peakIndex(end)) >= minDistance
            peakIndex(end + 1, 1) = idx; %#ok<AGROW>
        elseif amplitude(idx) > amplitude(peakIndex(end))
            peakIndex(end) = idx;
        end
    end
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
