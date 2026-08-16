%% 绘制历史误报健康窗口与最相似故障窗口

scriptFolder = string(fileparts(mfilename("fullpath")));
projectRoot = string(fileparts(fileparts(fileparts( ...
    fileparts(scriptFolder)))));
manifestFile = fullfile(projectRoot, "ML", "results", ...
    "target_false_alarm_analysis", "waveform_manifest.csv");
rawRunFolder = fullfile(scriptFolder, "dataset_output", "raw_runs");
outputFolder = fullfile(projectRoot, "ML", "results", ...
    "target_false_alarm_analysis", "waveforms");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

manifest = readtable(manifestFile, "TextType", "string");
manifest = unique(manifest(:, [ ...
    "OperatingPointID", "DestinationClassID", ...
    "HealthyRunID", "HealthyWindowStart", ...
    "FaultRunID", "FaultWindowStart"]), "rows", "stable");

for rowIndex = 1:height(manifest)
    item = manifest(rowIndex,:);
    healthy = loadRun(rawRunFolder, item.HealthyRunID);
    fault = loadRun(rawRunFolder, item.FaultRunID);
    healthyWindow = selectWindow(healthy, item.HealthyWindowStart);
    faultWindow = selectWindow(fault, item.FaultWindowStart);

    figureHandle = figure( ...
        "Color", "white", "Position", [100 100 1320 980], ...
        "Visible", "off");
    layout = tiledlayout(4, 2, "TileSpacing", "compact", ...
        "Padding", "compact");
    plotCurrent(nexttile(layout), healthyWindow, "健康");
    plotCurrent(nexttile(layout), faultWindow, "误判对应故障");
    plotVoltage(nexttile(layout), healthyWindow, "健康");
    plotVoltage(nexttile(layout), faultWindow, "误判对应故障");
    plotControl(nexttile(layout), healthyWindow, "健康");
    plotControl(nexttile(layout), faultWindow, "误判对应故障");
    plotGate(nexttile(layout), healthyWindow, "健康");
    plotGate(nexttile(layout), faultWindow, "误判对应故障");
    title(layout, sprintf( ...
        "%s：健康 %s vs 故障类 %d / %s", ...
        item.OperatingPointID, item.HealthyRunID, ...
        item.DestinationClassID, item.FaultRunID), ...
        "Interpreter", "none");

    outputFile = fullfile(outputFolder, ...
        item.OperatingPointID + "_healthy_vs_class_" + ...
        string(item.DestinationClassID) + ".png");
    exportgraphics(figureHandle, outputFile, "Resolution", 180);
    close(figureHandle);
end

fprintf("已生成 %d 张波形对比图：%s\n", height(manifest), outputFolder);

function rawTable = loadRun(folder, runID)
content = load(fullfile(folder, runID + ".mat"), "rawTable");
if ~isfield(content, "rawTable")
    error("faultdataset:MissingRawTable", "%s 缺少 rawTable。", runID);
end
rawTable = content.rawTable;
end

function selected = selectWindow(rawTable, windowStart)
margin = 0.02;
windowEnd = windowStart + 0.010;
selected = rawTable(rawTable.Time >= max(0, windowStart - margin) & ...
    rawTable.Time <= windowEnd + margin, :);
end

function plotCurrent(axisHandle, data, prefix)
plot(axisHandle, data.Time, data.IL_meas, "LineWidth", 1.0);
hold(axisHandle, "on");
plot(axisHandle, data.Time, data.Ibat_meas, "LineWidth", 1.0);
plot(axisHandle, data.Time, data.Iref, "--", "LineWidth", 1.0);
grid(axisHandle, "on");
ylabel(axisHandle, "电流 / A");
title(axisHandle, prefix + "：电流");
legend(axisHandle, "IL", "Ibat", "Iref", "Location", "best");
end

function plotVoltage(axisHandle, data, prefix)
plot(axisHandle, data.Time, data.Vbus_meas, "LineWidth", 1.0);
hold(axisHandle, "on");
plot(axisHandle, data.Time, data.Vbat_meas, "LineWidth", 1.0);
grid(axisHandle, "on");
ylabel(axisHandle, "电压 / V");
title(axisHandle, prefix + "：电压");
legend(axisHandle, "Vbus", "Vbat", "Location", "best");
end

function plotControl(axisHandle, data, prefix)
plot(axisHandle, data.Time, data.CurrentError, "LineWidth", 1.0);
hold(axisHandle, "on");
plot(axisHandle, data.Time, data.DutyApplied, "LineWidth", 1.0);
grid(axisHandle, "on");
ylabel(axisHandle, "控制量");
title(axisHandle, prefix + "：电流误差与占空比");
legend(axisHandle, "CurrentError", "DutyApplied", ...
    "Location", "best");
end

function plotGate(axisHandle, data, prefix)
stairs(axisHandle, data.Time, data.S1GateDuty, "LineWidth", 1.0);
hold(axisHandle, "on");
stairs(axisHandle, data.Time, data.S2GateDuty, "LineWidth", 1.0);
stairs(axisHandle, data.Time, data.FaultActive, "--", "LineWidth", 1.0);
grid(axisHandle, "on");
xlabel(axisHandle, "时间 / s");
ylabel(axisHandle, "比例");
title(axisHandle, prefix + "：门极与故障状态");
legend(axisHandle, "S1 duty", "S2 duty", "FaultActive", ...
    "Location", "best");
end
