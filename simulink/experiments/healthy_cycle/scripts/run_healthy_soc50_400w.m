%% Healthy charge, standby, and discharge operating-point check
% This experiment is intentionally isolated from the fault-dataset runs.
% It uses the same plant/control configuration and tests three healthy
% operating modes at SOC = 50% with a 400 W controllable bus load.

scriptFolder = string(fileparts(mfilename("fullpath")));
experimentFolder = fileparts(scriptFolder);
projectFolder = fileparts(fileparts(experimentFolder));
datasetScripts = fullfile(projectFolder, "experiments", ...
    "sensor_bias", "scripts");
addpath(datasetScripts);

runName = "2026-07-30_soc50_400W_healthfix_v5_final";
outputRoot = fullfile(experimentFolder, "results", runName);

baseCfg = signal_config();
healthyFault = baseCfg.faultList(baseCfg.faultList.FaultID == 0, :);

overrides = struct();
overrides.faultList = healthyFault;
overrides.cases.modeCommands = [0 1 2];
overrides.cases.socInit = 50;
overrides.cases.irefLevels = 10;
overrides.cases.vbusRef = 400;
overrides.cases.vbatInit = NaN;
overrides.cases.rload = 200;
overrides.cases.pload = 400;
overrides.cases.randomizeFaultStart = false;
overrides.cases.faultStartTimes = 1.0;
overrides.cases.faultDurations = Inf;
overrides.cases.stopTime = 1.0;
overrides.cases.repetitions = 1;
overrides.cases.maxRunCount = 10;
overrides.cases.domainRandomization.enabled = false;
overrides.execution.useParallel = false;
overrides.execution.useFastRestart = false;
overrides.execution.saveModelStates = false;
overrides.output.root = outputRoot;
overrides.output.saveRunCSV = true;
overrides.output.saveCombinedCSV = true;
overrides.output.overwritePolicy = "resume";

result = collect_fault_dataset(overrides);
loaded = load(result.RawDatasetFile, "rawDataset");
rawDataset = loaded.rawDataset;

steadyStartTime = 0.8;
summary = summarizeHealthyModes(rawDataset, steadyStartTime);
writetable(summary, fullfile(outputRoot, "steady_state_summary.csv"));

assessment = assessHealthyModes(summary, overrides);
writelines(assessment, fullfile(outputRoot, "assessment_zh.txt"), ...
    "Encoding", "UTF-8");

plotHealthyModes(rawDataset, outputRoot);
save(fullfile(outputRoot, "healthy_check_result.mat"), ...
    "summary", "assessment", "overrides", "result", "-v7.3");

disp(summary);
fprintf("\n%s\n", strjoin(assessment, newline));
fprintf("\n健康工况检查结果已保存到：\n%s\n", outputRoot);

function summary = summarizeHealthyModes(rawDataset, steadyStartTime)
modeCommands = [0; 1; 2];
modeNames = ["等待"; "充电"; "放电"];
rows = repmat(struct(), numel(modeCommands), 1);

for k = 1:numel(modeCommands)
    mode = modeCommands(k);
    modeData = rawDataset(rawDataset.ModeCommand == mode, :);
    steady = modeData.Time >= steadyStartTime;

    batteryPower = modeData.Vbat_true .* modeData.Ibat_true;
    rows(k).ModeCommand = mode;
    rows(k).ModeName = modeNames(k);
    rows(k).SteadySampleCount = nnz(steady);
    rows(k).ConverterEnableMean = finiteMean( ...
        modeData.ConverterEnable(steady));
    rows(k).VbusMean_V = finiteMean(modeData.Vbus_true(steady));
    rows(k).VbusMin_V = finiteMin(modeData.Vbus_true(steady));
    rows(k).VbusMax_V = finiteMax(modeData.Vbus_true(steady));
    rows(k).VbusStd_V = finiteStd(modeData.Vbus_true(steady));
    rows(k).VbatMean_V = finiteMean(modeData.Vbat_true(steady));
    rows(k).IrefMean_A = finiteMean(modeData.Iref(steady));
    rows(k).ILMean_A = finiteMean(modeData.IL_true(steady));
    rows(k).IbatMean_A = finiteMean(modeData.Ibat_true(steady));
    rows(k).CurrentErrorRMS_A = finiteRms( ...
        modeData.CurrentError(steady));
    rows(k).DutyMean = finiteMean(modeData.DutyApplied(steady));
    rows(k).S1DutyMean = finiteMean(modeData.S1GateDuty(steady));
    rows(k).S2DutyMean = finiteMean(modeData.S2GateDuty(steady));
    rows(k).PbatMean_W = finiteMean(batteryPower(steady));
    rows(k).PsourceMean_W = finiteMean(modeData.Psource_meas(steady));
    rows(k).PloadMeasuredMean_W = finiteMean( ...
        modeData.Pload_meas(steady));
    rows(k).PstoredMean_W = finiteMean(modeData.Pstored_meas(steady));
    rows(k).PowerBalanceResidualMean_W = finiteMean( ...
        modeData.PowerBalanceResidual(steady));
    rows(k).PowerBalanceResidualRMS_W = finiteRms( ...
        modeData.PowerBalanceResidual(steady));
    startup = modeData.Time <= 0.1;
    rows(k).StartupPeakAbsIbat_A = finiteMax( ...
        abs(modeData.Ibat_true(startup)));
    rows(k).StartupPeakAbsPbat_W = finiteMax( ...
        abs(batteryPower(startup)));
    rows(k).SOCStart_pct = firstFinite(modeData.SOC_true);
    rows(k).SOCEnd_pct = lastFinite(modeData.SOC_true);
    rows(k).SOCChange_pct = rows(k).SOCEnd_pct - ...
        rows(k).SOCStart_pct;
    rows(k).SOCEstStart_pct = firstFinite(modeData.SOC_est);
    rows(k).SOCEstEnd_pct = lastFinite(modeData.SOC_est);
    rows(k).SOCEstChange_pct = rows(k).SOCEstEnd_pct - ...
        rows(k).SOCEstStart_pct;
    rows(k).SatFlagIRatio = finiteMean(modeData.SatFlagI(steady));
    rows(k).SatFlagVRatio = finiteMean(modeData.SatFlagV(steady));
    rows(k).SatFlagRatio = finiteMean(modeData.SatFlag(steady));
end

summary = struct2table(rows, "AsArray", true);
end

function assessment = assessHealthyModes(summary, overrides)
assessment = strings(0, 1);
assessment(end+1) = "健康充放电仿真检查";
assessment(end+1) = "====================";
assessment(end+1) = sprintf( ...
    "设定：SOC=50%%，母线参考=400 V，可控负载命令=400 W，" + ...
    "固定母线电阻=200 ohm，充放电电流目标=10 A。");

standby = summary(summary.ModeCommand == 0, :);
charge = summary(summary.ModeCommand == 1, :);
discharge = summary(summary.ModeCommand == 2, :);

assessment(end+1) = passFail( ...
    abs(standby.IbatMean_A) < 0.2 && ...
    abs(standby.PbatMean_W) < 50 && ...
    abs(standby.SOCChange_pct) < 5e-4, ...
    "等待状态：电池电流和功率接近 0，SOC 基本不变。");
assessment(end+1) = passFail( ...
    charge.IbatMean_A < -1 && charge.PbatMean_W < -100 && ...
    charge.SOCChange_pct > 0 && charge.ConverterEnableMean > 0.9, ...
    "充电状态：电池吸收功率，SOC 上升，变换器已使能。");
assessment(end+1) = passFail( ...
    discharge.IbatMean_A > 1 && discharge.PbatMean_W > 100 && ...
    discharge.SOCChange_pct < 0 && ...
    discharge.ConverterEnableMean > 0.9, ...
    "放电状态：电池输出功率，SOC 下降，变换器已使能。");

for k = 1:height(summary)
    assessment(end+1) = passFail( ...
        abs(summary.VbusMean_V(k) - 400) <= 20, ...
        sprintf("%s状态：稳态母线平均电压 %.2f V，" + ...
        "相对 400 V 偏差 %.2f%%。", summary.ModeName(k), ...
        summary.VbusMean_V(k), ...
        100 * (summary.VbusMean_V(k) - 400) / 400));
end

controlledLoad = overrides.cases.pload;
fixedLoadExpected = overrides.cases.vbusRef.^2 / ...
    overrides.cases.rload;
totalLoadExpected = controlledLoad + fixedLoadExpected;
measuredLoad = mean(summary.PloadMeasuredMean_W, "omitnan");
assessment(end+1) = sprintf( ...
    "[说明] 400 W 是可控负载支路命令；200 ohm 固定支路在 400 V " + ...
    "附近还消耗约 %.0f W，因此模型测得的总负载功率约 %.1f W，" + ...
    "理论合计约 %.0f W。", fixedLoadExpected, measuredLoad, ...
    totalLoadExpected);

socSetpoint = overrides.cases.socInit;
assessment(end+1) = passFail( ...
    all(abs(summary.SOCEstStart_pct - socSetpoint) < 0.01), ...
    sprintf("独立库仑计 SOC_est 的初值为 %.3f%%，与 50%% 设定一致。", ...
    mean(summary.SOCEstStart_pct, "omitnan")));
assessment(end+1) = passFail( ...
    all(abs(summary.SOCStart_pct - socSetpoint) < 0.1), ...
    sprintf("电池模块 SOC_true 的初值为 %.3f%%；与 50%% 设定相差 %.3f " + ...
    "个百分点。", mean(summary.SOCStart_pct, "omitnan"), ...
    mean(summary.SOCStart_pct, "omitnan") - socSetpoint));
assessment(end+1) = sprintf( ...
    "[说明] 电流环稳态饱和比例：等待 %.1f%%、充电 %.1f%%、放电 %.1f%%；" + ...
    "电压环分别为 %.1f%%、%.1f%%、%.1f%%。电压环输出达到 ±10 A " + ...
    "限流值属于限流动作，不代表电流控制器失控。", ...
    100 * standby.SatFlagIRatio, 100 * charge.SatFlagIRatio, ...
    100 * discharge.SatFlagIRatio, 100 * standby.SatFlagVRatio, ...
    100 * charge.SatFlagVRatio, 100 * discharge.SatFlagVRatio);
startupIsHealthy = standby.StartupPeakAbsIbat_A < 1 && ...
    charge.StartupPeakAbsIbat_A <= 12 && ...
    discharge.StartupPeakAbsIbat_A <= 12;
assessment(end+1) = passFail( ...
    startupIsHealthy, ...
    sprintf("启动峰值：等待 %.2f A、充电 %.2f A、放电 %.2f A；" + ...
    "充放电峰值与 10 A 目标一致，最大电池功率 %.2f kW。", ...
    standby.StartupPeakAbsIbat_A, charge.StartupPeakAbsIbat_A, ...
    discharge.StartupPeakAbsIbat_A, ...
    max(summary.StartupPeakAbsPbat_W) / 1000));

assessment(end+1) = sprintf( ...
    "[提示] 充电模式母线压降是三个状态中最大的；" + ...
    "当前稳态为 %.2f V，应重点优化母线电压支撑。", ...
    charge.VbusMean_V);
end

function line = passFail(condition, message)
if condition
    prefix = "[通过] ";
else
    prefix = "[需检查] ";
end
line = prefix + message;
end

function plotHealthyModes(rawDataset, outputRoot)
modeCommands = [0 1 2];
modeNames = ["等待", "充电", "放电"];
colors = lines(3);

figureHandle = figure("Visible", "off", "Color", "white", ...
    "Position", [100 100 1200 800]);
layout = tiledlayout(2, 2, "TileSpacing", "compact", ...
    "Padding", "compact");

nexttile;
hold on;
for k = 1:3
    data = rawDataset(rawDataset.ModeCommand == modeCommands(k), :);
    plot(data.Time, data.Vbus_true, "Color", colors(k, :), ...
        "DisplayName", modeNames(k));
end
yline(400, "--k", "400 V reference", "HandleVisibility", "off");
xlabel("Time (s)");
ylabel("DC bus voltage (V)");
title("Bus voltage");
grid on;
legend("Location", "best");

nexttile;
hold on;
for k = 1:3
    data = rawDataset(rawDataset.ModeCommand == modeCommands(k), :);
    plot(data.Time, data.Ibat_true, "Color", colors(k, :), ...
        "DisplayName", modeNames(k));
end
yline(0, ":k", "HandleVisibility", "off");
ylim([-12 12]);
xlabel("Time (s)");
ylabel("Battery current (A)");
title("Battery current: negative=charging, positive=discharging");
grid on;

nexttile;
hold on;
for k = 1:3
    data = rawDataset(rawDataset.ModeCommand == modeCommands(k), :);
    batteryPower = data.Vbat_true .* data.Ibat_true;
    plot(data.Time, batteryPower, "Color", colors(k, :), ...
        "DisplayName", modeNames(k));
end
yline(0, ":k", "HandleVisibility", "off");
ylim([-2500 2500]);
xlabel("Time (s)");
ylabel("Battery power (W)");
title("Battery power: negative=absorbing, positive=supplying");
grid on;

nexttile;
hold on;
for k = 1:3
    data = rawDataset(rawDataset.ModeCommand == modeCommands(k), :);
    plot(data.Time, data.SOC_true, "Color", colors(k, :), ...
        "DisplayName", modeNames(k));
end
xlabel("Time (s)");
ylabel("SOC (%)");
title("Battery SOC");
grid on;

title(layout, "Healthy operating-point check: SOC 50%, 400 W load command");
exportgraphics(figureHandle, ...
    fullfile(outputRoot, "healthy_electrical_states.png"), ...
    "Resolution", 180);
close(figureHandle);
end

function value = finiteMean(data)
value = mean(data(isfinite(data)), "omitnan");
end

function value = finiteStd(data)
value = std(data(isfinite(data)), "omitnan");
end

function value = finiteMin(data)
data = data(isfinite(data));
if isempty(data)
    value = NaN;
else
    value = min(data);
end
end

function value = finiteMax(data)
data = data(isfinite(data));
if isempty(data)
    value = NaN;
else
    value = max(data);
end
end

function value = finiteRms(data)
data = data(isfinite(data));
if isempty(data)
    value = NaN;
else
    value = sqrt(mean(data.^2));
end
end

function value = firstFinite(data)
index = find(isfinite(data), 1, "first");
if isempty(index)
    value = NaN;
else
    value = data(index);
end
end

function value = lastFinite(data)
index = find(isfinite(data), 1, "last");
if isempty(index)
    value = NaN;
else
    value = data(index);
end
end
