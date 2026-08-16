%% 计算各电感电流参考平台的稳态误差
% 先运行仿真生成 out，再运行本脚本。

outputName        = 'out';       % SimulationOutput 变量名
refSignalName     = 'log_Iref';     % 参考电流在 out 中的名称
currentSignalName = 'log_I_L';   % 实际电感电流在 out 中的名称
averageWindow     = 0.02;        % 各平台末尾平均时间窗，单位 s
referenceLevelTol_A = 1e-3;      % 零参考和参考值比较阈值
referenceSlopeTol_A_per_s = 1;   % 小于该斜率视为参考电流平台
minStableDuration = averageWindow; % 小于该时长的“稳定段”不计入结果
includeZeroRef    = true;        % 保留 0 A 平台（例如 0~0.1 s）

refRaw = getLoggedSignal(outputName, refSignalName);
currentRaw = getLoggedSignal(outputName, currentSignalName);
[tRef, iRef] = signalData(refRaw);
[tCurrent, iCurrent] = signalData(currentRaw);

iRef = interp1(tRef, iRef, tCurrent, 'previous', 'extrap');
t = tCurrent;

% 根据参考电流斜率识别连续稳定区间，支持各平台持续时间不同。
dt = diff(t);
diRef = diff(iRef);
validStep = dt > 0;
refSlope = inf(size(diRef));
refSlope(validStep) = abs(diRef(validStep) ./ dt(validStep));
stableMask = [true; refSlope <= referenceSlopeTol_A_per_s];
maskEdge = diff([false; stableMask; false]);
segmentStart = find(maskEdge == 1);
segmentEnd = find(maskEdge == -1) - 1;
segmentDuration = t(segmentEnd) - t(segmentStart);
keepSegment = segmentDuration >= minStableDuration;
segmentStart = segmentStart(keepSegment);
segmentEnd = segmentEnd(keepSegment);
assert(~isempty(segmentStart), '没有检测到持续时间足够的稳定参考平台。');
fprintf('自动检测到 %d 个稳定参考平台。\n', numel(segmentStart));

result = table();
for k = 1:numel(segmentStart)
    idx0 = segmentStart(k);
    idx1 = segmentEnd(k);
    % 使用稳定段末尾的中位数作为平台参考，抑制微小数值噪声。
    refMean = median(iRef(idx0:idx1), 'omitnan');
    if ~includeZeroRef && abs(refMean) <= referenceLevelTol_A
        continue %#ok<UNRCH>
    end

    averageStart = max(t(idx0), t(idx1) - averageWindow);
    steadyIndex = t >= averageStart & t <= t(idx1);
    if nnz(steadyIndex) < 2
        warning('参考平台 %d 的平均时间窗内数据不足，已跳过。', k);
        continue
    end

    currentMean = mean(iCurrent(steadyIndex), 'omitnan');
    signedError = currentMean - refMean;
    absoluteError = abs(signedError);
    if abs(refMean) > referenceLevelTol_A
        relativeErrorPct = 100 * absoluteError / abs(refMean);
    else
        relativeErrorPct = NaN; % 参考为零时，相对误差无定义
    end
    platformDuration_s = t(idx1) - t(idx0);
    row = table(k, t(idx0), t(idx1), platformDuration_s, averageStart, ...
        refMean, currentMean, ...
        signedError, absoluteError, relativeErrorPct, 'VariableNames', ...
        {'Platform','PlatformStart_s','PlatformEnd_s','PlatformDuration_s', ...
         'AverageStart_s', ...
         'Reference_A','MeanCurrent_A','SignedError_A','AbsoluteError_A', ...
         'RelativeError_pct'});
    result = [result; row]; %#ok<AGROW>
end

meanAbsoluteErrorA = mean(result.AbsoluteError_A, 'omitnan');
meanRelativeErrorPct = mean(result.RelativeError_pct, 'omitnan');
result.MeanAbsoluteError_A = repmat(meanAbsoluteErrorA, height(result), 1);
result.MeanRelativeError_pct = repmat(meanRelativeErrorPct, height(result), 1);

disp('各参考电流平台的稳态误差：');
disp(result);
fprintf('平均绝对稳态误差：%.6g A\n', meanAbsoluteErrorA);
fprintf('平均相对稳态误差：%.6g %%\n', meanRelativeErrorPct);
assignin('base', 'steadyStateErrorResult', result);
assignin('base', 'meanAbsoluteSteadyStateError_A', meanAbsoluteErrorA);
assignin('base', 'meanRelativeSteadyStateError_pct', meanRelativeErrorPct);
analysisFolder = fileparts(fileparts(mfilename('fullpath')));
resultFolder = fullfile(analysisFolder, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end
resultFile = fullfile(resultFolder, 'current_steady_state_error.csv');
writetable(result, resultFile);
fprintf('稳态误差结果已保存：%s\n', resultFile);
openvar('steadyStateErrorResult');

function signal = getLoggedSignal(outputName, signalName)
    assert(evalin('base', "exist('" + outputName + "','var') == 1"), ...
        '工作区中找不到 %s，请先运行仿真。', outputName);
    simOut = evalin('base', outputName);
    assert(isa(simOut, 'Simulink.SimulationOutput'), ...
        '%s 不是 Simulink.SimulationOutput 对象。', outputName);

    outputVariables = who(simOut);
    if any(strcmp(outputVariables, signalName))
        signal = simOut.get(signalName);       % out.I_ref 等
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
