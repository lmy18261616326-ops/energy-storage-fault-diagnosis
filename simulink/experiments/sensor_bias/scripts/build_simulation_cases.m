function cases = build_simulation_cases(cfg)
%BUILD_SIMULATION_CASES 生成批量仿真的配置表。
%   CASES = BUILD_SIMULATION_CASES(CFG) 对工况、故障、严重度、发生时刻和
%   重复次数做笛卡尔组合。每一行是一轮独立仿真。
%
%   输入:
%       cfg   - signal_config 返回的配置结构体。
%   输出:
%       cases - 每行一次仿真的 table。

arguments
    cfg (1,1) struct = signal_config()
end

validateConfigCompatibility(cfg);

[modeGrid, socGrid, irefGrid, vrefGrid, vbatGrid, rloadGrid, ...
    ploadGrid] = ndgrid( ...
    cfg.cases.modeCommands, cfg.cases.socInit, cfg.cases.irefLevels, ...
    cfg.cases.vbusRef, cfg.cases.vbatInit, cfg.cases.rload, ...
    cfg.cases.pload);

operatingPoints = table( ...
    modeGrid(:), socGrid(:), irefGrid(:), vrefGrid(:), ...
    vbatGrid(:), rloadGrid(:), ploadGrid(:), ...
    'VariableNames', {'ModeCommand','SOCInit','IrefLevel','VbusRef', ...
    'VbatInit','Rload','Pload'});
operatingPointPrefix = "op";
if isfield(cfg.cases, "operatingPointPrefix") && ...
        strlength(string(cfg.cases.operatingPointPrefix)) > 0
    operatingPointPrefix = string(cfg.cases.operatingPointPrefix);
end
operatingPoints.OperatingPointID = compose( ...
    operatingPointPrefix + "_%04d", ...
    (1:height(operatingPoints))');
operatingPoints = movevars(operatingPoints, "OperatingPointID", ...
    "Before", 1);
if isfield(cfg.cases, "operatingPointIDs") && ...
        ~isempty(cfg.cases.operatingPointIDs)
    requestedIDs = unique(string(cfg.cases.operatingPointIDs), "stable");
    knownIDs = string(operatingPoints.OperatingPointID);
    missingIDs = setdiff(requestedIDs, knownIDs);
    if ~isempty(missingIDs)
        error("faultdataset:UnknownOperatingPointID", ...
            "请求了不存在的 OperatingPointID: %s", ...
            strjoin(missingIDs, ", "));
    end
    operatingPoints = operatingPoints( ...
        ismember(knownIDs, requestedIDs), :);
end

runIDPrefix = "run";
if isfield(cfg.cases, "runIDPrefix") && ...
        strlength(string(cfg.cases.runIDPrefix)) > 0
    runIDPrefix = string(cfg.cases.runIDPrefix);
end

rows = struct([]);
rowIndex = 0;
runNumber = 0;
for opIndex = 1:height(operatingPoints)
    op = operatingPoints(opIndex,:);
    for faultIndex = 1:height(cfg.faultList)
        fault = cfg.faultList(faultIndex,:);
        magnitudes = fault.Magnitudes{1};
        if fault.FaultID == 0
            startTimes = cfg.cases.stopTime;
            faultDurations = Inf;
        elseif cfg.cases.randomizeFaultStart
            % 每个重复运行按 RandomSeed 单独生成，NaN 仅作为单次循环占位。
            startTimes = NaN;
            faultDurations = cfg.cases.faultDurations;
        else
            startTimes = cfg.cases.faultStartTimes;
            faultDurations = cfg.cases.faultDurations;
        end
        for magnitude = magnitudes
            for startTime = startTimes
                for faultDuration = faultDurations
                    for repetition = 1:cfg.cases.repetitions
                    rowIndex = rowIndex + 1;
                    runNumber = runNumber + 1;
                    rows(rowIndex).RunID = compose( ...
                        "%s_%05d", runIDPrefix, runNumber);
                    rows(rowIndex).OperatingPointID = op.OperatingPointID;
                    rows(rowIndex).RandomSeed = ...
                        cfg.cases.randomSeedBase + runNumber - 1;
                    stream = RandStream("mt19937ar", ...
                        "Seed", rows(rowIndex).RandomSeed);
                    rows(rowIndex).ModeCommand = op.ModeCommand;
                    rows(rowIndex).SOCInit = op.SOCInit;
                    rows(rowIndex).IrefLevel = op.IrefLevel;
                    rows(rowIndex).VbusRef = op.VbusRef;
                    rows(rowIndex).VbatInit = op.VbatInit;
                    rows(rowIndex).Rload = op.Rload;
                    rows(rowIndex).Pload = op.Pload;
                    [rows(rowIndex).Rbat, rows(rowIndex).Cbus, ...
                        rows(rowIndex).CbusESR] = ...
                        randomizePlantParameters(cfg, stream);
                    rows(rowIndex).FaultID = fault.FaultID;
                    rows(rowIndex).FaultName = fault.FaultName;
                    rows(rowIndex).FaultLocation = fault.FaultLocation;
                    rows(rowIndex).FaultMagnitude = magnitude;
                    rows(rowIndex).FaultParameter1 = fault.ModelFaultID;
                    if ismember(fault.FaultID, ...
                            cfg.labeling.switchFaultIDs)
                        if cfg.adapter.switchFaultMechanism == ...
                                "gate_blocking"
                            rows(rowIndex).FaultParameter2 = ...
                                cfg.adapter.switchFaultPeriod;
                        else
                            rows(rowIndex).FaultParameter2 = NaN;
                        end
                    else
                        rows(rowIndex).FaultParameter2 = repetition;
                    end
                    if fault.FaultID == 0
                        rows(rowIndex).FaultStartTime = cfg.cases.stopTime;
                    elseif cfg.cases.randomizeFaultStart
                        range = cfg.cases.faultStartRange;
                        rows(rowIndex).FaultStartTime = range(1) + ...
                            diff(range) * rand(stream);
                    else
                        rows(rowIndex).FaultStartTime = startTime;
                    end
                    rows(rowIndex).FaultEndTime = min( ...
                        rows(rowIndex).FaultStartTime + faultDuration, ...
                        cfg.cases.stopTime);
                    rows(rowIndex).SimulationStopTime = cfg.cases.stopTime;
                    end
                end
            end
        end
    end
end

if isempty(rows)
    error("faultdataset:NoCases", "没有生成任何仿真工况。");
end
cases = struct2table(rows, "AsArray", true);

if height(cases) > cfg.cases.maxRunCount
    error("faultdataset:TooManyCases", ...
        "计划生成 %d 次仿真，超过 maxRunCount=%d。请缩小工况网格或提高上限。", ...
        height(cases), cfg.cases.maxRunCount);
end

cases.RunID = string(cases.RunID);
cases.OperatingPointID = string(cases.OperatingPointID);
cases.FaultName = string(cases.FaultName);
cases.FaultLocation = string(cases.FaultLocation);
end

function validateConfigCompatibility(cfg)
if ~cfg.compatibility.supportsIrefLevel && ...
        any(abs(cfg.cases.irefLevels - ...
        cfg.compatibility.irefFixedValue) > eps)
    error("faultdataset:IrefNotTunable", ...
        ["当前 Mode_Manager 把充放电电流上限写成 Stateflow 常数，" ...
        "IrefLevel 尚不能由 SimulationInput 改变。请先完成模型适配，" ...
        "或者把 cfg.cases.irefLevels 固定为 %g。"], ...
        cfg.compatibility.irefFixedValue);
end
if cfg.cases.randomizeFaultStart
    faultTimes = cfg.cases.faultStartRange;
else
    faultTimes = cfg.cases.faultStartTimes;
end
if ~cfg.compatibility.supportsFaultEndTime && ...
        any(faultTimes >= cfg.cases.stopTime)
    error("faultdataset:InvalidFaultTime", ...
        "FaultStartTime 必须小于 SimulationStopTime。");
end
if any(faultTimes < 0) || ...
        (cfg.cases.randomizeFaultStart && numel(faultTimes) ~= 2) || ...
        (cfg.cases.randomizeFaultStart && faultTimes(2) <= faultTimes(1))
    error("faultdataset:InvalidFaultTimeRange", ...
        "faultStartRange 必须是严格递增且非负的两个时刻。");
end
if isempty(cfg.cases.faultDurations) || ...
        any(cfg.cases.faultDurations <= 0 | isnan(cfg.cases.faultDurations))
    error("faultdataset:InvalidFaultDuration", ...
        "faultDurations 必须为正数或 Inf。");
end
if any(cfg.cases.socInit < 0 | cfg.cases.socInit > 100)
    error("faultdataset:InvalidSOC", "SOCInit 必须位于 [0,100]。");
end
end

function [rbat, cbus, cbusESR] = randomizePlantParameters(cfg, stream)
domain = cfg.cases.domainRandomization;
if ~domain.enabled
    rbat = domain.RbatNominal;
    cbus = domain.CbusNominal;
    cbusESR = domain.CbusESRNominal;
    return;
end
rbat = drawRelative(domain.RbatNominal, ...
    domain.RbatRelativeRange, stream);
cbus = drawRelative(domain.CbusNominal, ...
    domain.CbusRelativeRange, stream);
cbusESR = drawRelative(domain.CbusESRNominal, ...
    domain.CbusESRRelativeRange, stream);
end

function value = drawRelative(nominal, relativeRange, stream)
value = nominal .* (1 + relativeRange .* (2*rand(stream)-1));
end
