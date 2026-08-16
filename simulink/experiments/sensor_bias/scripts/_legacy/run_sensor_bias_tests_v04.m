%% Sensor-bias validation and dataset generation for v04
% The saved model is not modified by this script. All scenario parameters
% are applied through Simulink.SimulationInput.

modelName = "main_model_fd_v04_faultdiag";
scriptFolder = fileparts(mfilename("fullpath"));
experimentFolder = fileparts(scriptFolder);
experimentsFolder = fileparts(experimentFolder);
simulinkFolder = fileparts(experimentsFolder);
modelFolder = fullfile(simulinkFolder, "models");
if ~exist("sensorTestQuickMode", "var")
    sensorTestQuickMode = false;
end

if sensorTestQuickMode
    resultsFolderName = "smoke";
else
    resultsFolderName = "full";
end
resultsFolder = fullfile( ...
    experimentFolder, "results", resultsFolderName);
addpath(modelFolder);

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

if sensorTestQuickMode
    faultTime = 0.40;
    loadStepTime = 0.30;
    stopTime = 0.65;
    analysisWindow = [0.25, stopTime - 0.02];
    postFaultWindow = [faultTime + 0.08, stopTime - 0.02];
else
    faultTime = 0.60;
    loadStepTime = 0.40;
    stopTime = 0.90;
    analysisWindow = [0.45, stopTime - 0.02];
    postFaultWindow = [faultTime + 0.08, stopTime - 0.02];
end
diagnosticSampleTime = 1e-4;
featureWindow = 0.020;
featureStride = 0.010;

vbusBiasTest = 5.0;
vbatBiasTest = 1.0;
currentBiasTest = 0.5;

base = createScenarioTable( ...
    vbusBiasTest, vbatBiasTest, currentBiasTest);

if sensorTestQuickMode
    base = base(ismember( ...
        base.FaultName, ["healthy", "il_bias_pos"]), :);
    modeCommand = 2;
else
    modeCommand = [1; 2];
end
numScenarios = height(base) * numel(modeCommand);
scenario = repmat(base, numel(modeCommand), 1);
scenario.ModeCommand = repelem( ...
    modeCommand(:), height(base), 1);
scenario.Scenario = strings(numScenarios, 1);
scenario.RunID = strings(numScenarios, 1);

for scenarioIndex = 1:numScenarios
    suffix = "_mode" + string(scenario.ModeCommand(scenarioIndex));
    scenario.Scenario(scenarioIndex) = ...
        scenario.FaultName(scenarioIndex) + suffix;
    scenario.RunID(scenarioIndex) = scenario.Scenario(scenarioIndex);
end

in = repmat(Simulink.SimulationInput(modelName), numScenarios, 1);
for scenarioIndex = 1:numScenarios
    in(scenarioIndex) = Simulink.SimulationInput(modelName);
    in(scenarioIndex) = in(scenarioIndex).setModelParameter( ...
        "StopTime", string(stopTime));
    in(scenarioIndex) = setScenarioVariables( ...
        in(scenarioIndex), scenario(scenarioIndex, :), ...
        faultTime, loadStepTime, modelName);
    in(scenarioIndex) = in(scenarioIndex).setUserString( ...
        scenario.Scenario(scenarioIndex));
end

% Sensor-bias cases do not change compiled physical-network parameters.
out = sim(in, "UseFastRestart", "on", "ShowProgress", "off");

summary = initializeSummary(scenario);
featureTable = table;

for scenarioIndex = 1:numScenarios
    trace = readScenarioTraces(out(scenarioIndex));

    analysisTime = ...
        (analysisWindow(1):diagnosticSampleTime:analysisWindow(2))';
    data = sampleScenarioTraces(trace, analysisTime);

    postTime = ...
        (postFaultWindow(1):diagnosticSampleTime:postFaultWindow(2))';
    postData = sampleScenarioTraces(trace, postTime);

    observedBias = [ ...
        median(postData.VbusMeas - postData.VbusTrue), ...
        median(postData.VbatMeas - postData.VbatTrue), ...
        median(postData.IbatMeas - postData.IbatTrue), ...
        median(postData.ILMeas - postData.ILTrue)];

    expectedBias = [ ...
        scenario.VbusBias(scenarioIndex), ...
        scenario.VbatBias(scenarioIndex), ...
        scenario.IbatBias(scenarioIndex), ...
        scenario.ILBias(scenarioIndex)];

    absoluteTolerance = [0.05, 0.05, 0.01, 0.01];
    allowedError = max(0.01 * abs(expectedBias), absoluteTolerance);

    trackingError = postData.ILMeas - postData.Iref;
    currentPairResidual = postData.IbatMeas + postData.ILMeas;

    summary.VbusTrueMean(scenarioIndex) = mean(postData.VbusTrue);
    summary.VbusMeasMean(scenarioIndex) = mean(postData.VbusMeas);
    summary.VbatTrueMean(scenarioIndex) = mean(postData.VbatTrue);
    summary.VbatMeasMean(scenarioIndex) = mean(postData.VbatMeas);
    summary.IbatTrueMean(scenarioIndex) = mean(postData.IbatTrue);
    summary.IbatMeasMean(scenarioIndex) = mean(postData.IbatMeas);
    summary.ILTrueMean(scenarioIndex) = mean(postData.ILTrue);
    summary.ILMeasMean(scenarioIndex) = mean(postData.ILMeas);

    summary.VbusBiasObserved(scenarioIndex) = observedBias(1);
    summary.VbatBiasObserved(scenarioIndex) = observedBias(2);
    summary.IbatBiasObserved(scenarioIndex) = observedBias(3);
    summary.ILBiasObserved(scenarioIndex) = observedBias(4);
    summary.InjectionPass(scenarioIndex) = ...
        all(abs(observedBias - expectedBias) <= allowedError);

    summary.CurrentTrackingErrorRMS(scenarioIndex) = ...
        sqrt(mean(trackingError .^ 2));
    summary.CurrentPairResidualRMS(scenarioIndex) = ...
        sqrt(mean(currentPairResidual .^ 2));
    summary.DutyMean(scenarioIndex) = mean(postData.DutyApplied);
    summary.DutySaturationRatio(scenarioIndex) = mean( ...
        postData.DutyApplied <= 1e-3 | ...
        postData.DutyApplied >= 1 - 1e-3);
    summary.TripDetected(scenarioIndex) = ...
        double(any(trace.TripDetected.Data ~= 0));
    summary.TripApplied(scenarioIndex) = ...
        double(any(trace.TripApplied.Data ~= 0));

    scenarioFeatures = extractWindowFeatures( ...
        analysisTime, data, featureWindow, featureStride, ...
        scenario(scenarioIndex, :), faultTime);
    featureTable = [featureTable; scenarioFeatures]; %#ok<AGROW>
end

writetable(summary, ...
    fullfile(resultsFolder, "sensor_bias_summary.csv"));
writetable(featureTable, ...
    fullfile(resultsFolder, "sensor_bias_feature_dataset.csv"));
save(fullfile(resultsFolder, "sensor_bias_dataset.mat"), ...
    "scenario", "summary", "featureTable");

if ~sensorTestQuickMode
    createSensorBiasFigure( ...
        out, scenario, faultTime, stopTime, resultsFolder);
end

disp(summary);
fprintf("FEATURES=%s\n", ...
    fullfile(resultsFolder, "sensor_bias_feature_dataset.csv"));
fprintf("SUMMARY=%s\n", ...
    fullfile(resultsFolder, "sensor_bias_summary.csv"));

function base = createScenarioTable( ...
    vbusBiasTest, vbatBiasTest, currentBiasTest)

faultNames = [ ...
    "healthy"
    "load_step"
    "vbus_bias_pos"
    "vbus_bias_neg"
    "vbat_bias_pos"
    "vbat_bias_neg"
    "il_bias_pos"
    "il_bias_neg"
    "ibat_bias_pos"
    "ibat_bias_neg"];

base = table(faultNames, ...
    'VariableNames', {'FaultName'});
numBaseScenarios = height(base);

base.FaultID = zeros(numBaseScenarios, 1);
base.Severity = zeros(numBaseScenarios, 1);
base.BiasSign = zeros(numBaseScenarios, 1);
base.Rbat = 0.5 * ones(numBaseScenarios, 1);
base.Cbus = 2.2e-3 * ones(numBaseScenarios, 1);
base.CbusESR = 1e-3 * ones(numBaseScenarios, 1);
base.VbusBias = zeros(numBaseScenarios, 1);
base.VbatBias = zeros(numBaseScenarios, 1);
base.IbatBias = zeros(numBaseScenarios, 1);
base.ILBias = zeros(numBaseScenarios, 1);
base.DutyBias = zeros(numBaseScenarios, 1);
base.DutyStuckEnable = zeros(numBaseScenarios, 1);
base.DutyStuckValue = 0.5 * ones(numBaseScenarios, 1);
base.LoadStepA = zeros(numBaseScenarios, 1);
base.S1Open = zeros(numBaseScenarios, 1);
base.S2Open = zeros(numBaseScenarios, 1);

idx = base.FaultName == "load_step";
base.LoadStepA(idx) = 1.0;

idx = ismember(base.FaultName, ...
    ["vbus_bias_pos", "vbus_bias_neg"]);
base.FaultID(idx) = 2;
base.VbusBias(idx) = [vbusBiasTest; -vbusBiasTest];
base.Severity(idx) = vbusBiasTest;
base.BiasSign(idx) = [1; -1];

idx = ismember(base.FaultName, ...
    ["il_bias_pos", "il_bias_neg"]);
base.FaultID(idx) = 3;
base.ILBias(idx) = [currentBiasTest; -currentBiasTest];
base.Severity(idx) = currentBiasTest;
base.BiasSign(idx) = [1; -1];

idx = ismember(base.FaultName, ...
    ["vbat_bias_pos", "vbat_bias_neg"]);
base.FaultID(idx) = 8;
base.VbatBias(idx) = [vbatBiasTest; -vbatBiasTest];
base.Severity(idx) = vbatBiasTest;
base.BiasSign(idx) = [1; -1];

idx = ismember(base.FaultName, ...
    ["ibat_bias_pos", "ibat_bias_neg"]);
base.FaultID(idx) = 9;
base.IbatBias(idx) = [currentBiasTest; -currentBiasTest];
base.Severity(idx) = currentBiasTest;
base.BiasSign(idx) = [1; -1];
end

function summary = initializeSummary(scenario)
numScenarios = height(scenario);
summary = scenario(:, [ ...
    "Scenario", "RunID", "FaultName", "FaultID", ...
    "ModeCommand", "Severity", "BiasSign", ...
    "VbusBias", "VbatBias", "IbatBias", "ILBias"]);

summary.VbusTrueMean = zeros(numScenarios, 1);
summary.VbusMeasMean = zeros(numScenarios, 1);
summary.VbatTrueMean = zeros(numScenarios, 1);
summary.VbatMeasMean = zeros(numScenarios, 1);
summary.IbatTrueMean = zeros(numScenarios, 1);
summary.IbatMeasMean = zeros(numScenarios, 1);
summary.ILTrueMean = zeros(numScenarios, 1);
summary.ILMeasMean = zeros(numScenarios, 1);

summary.VbusBiasObserved = zeros(numScenarios, 1);
summary.VbatBiasObserved = zeros(numScenarios, 1);
summary.IbatBiasObserved = zeros(numScenarios, 1);
summary.ILBiasObserved = zeros(numScenarios, 1);
summary.InjectionPass = false(numScenarios, 1);

summary.CurrentTrackingErrorRMS = zeros(numScenarios, 1);
summary.CurrentPairResidualRMS = zeros(numScenarios, 1);
summary.DutyMean = zeros(numScenarios, 1);
summary.DutySaturationRatio = zeros(numScenarios, 1);
summary.TripDetected = zeros(numScenarios, 1);
summary.TripApplied = zeros(numScenarios, 1);
end

function trace = readScenarioTraces(out)
trace = struct( ...
    "VbusMeas", out.get("log_Vbus"), ...
    "VbusTrue", out.get("log_Vbus_true"), ...
    "VbatMeas", out.get("log_Vbat"), ...
    "VbatTrue", out.get("log_Vbat_true"), ...
    "IbatMeas", out.get("log_Ibat"), ...
    "IbatTrue", out.get("log_Ibat_true"), ...
    "ILMeas", out.get("log_I_L"), ...
    "ILTrue", out.get("log_IL_true"), ...
    "Iref", out.get("log_Iref"), ...
    "DutyCommand", out.get("log_Duty_cmd"), ...
    "DutyApplied", out.get("log_Duty_applied"), ...
    "TripDetected", out.get("log_trip_detected"), ...
    "TripApplied", out.get("log_trip_applied"));
end

function data = sampleScenarioTraces(trace, queryTime)
data = struct;
fieldNames = [ ...
    "VbusMeas", "VbusTrue", "VbatMeas", "VbatTrue", ...
    "IbatMeas", "IbatTrue", "ILMeas", "ILTrue", ...
    "Iref", "DutyCommand", "DutyApplied"];

for fieldIndex = 1:numel(fieldNames)
    name = fieldNames(fieldIndex);
    data.(name) = sampleTrace(trace.(name), queryTime);
end
end

function in = setScenarioVariables( ...
    in, row, faultTime, loadStepTime, modelName)
in = setModelVariable(in, modelName, "FD_FAULT_TIME", faultTime);
in = setModelVariable(in, modelName, ...
    "FD_LOAD_STEP_TIME", loadStepTime);
in = setModelVariable(in, modelName, ...
    "FD_MODE_OVERRIDE_ENABLE", 1);
in = setModelVariable(in, modelName, ...
    "FD_MODE_COMMAND", row.ModeCommand);
in = setModelVariable(in, modelName, "FD_PROTECTION_MODE", 2);
in = setModelVariable(in, modelName, "FD_FAULT_ID", row.FaultID);
in = setModelVariable(in, modelName, "FD_RBAT", row.Rbat);
in = setModelVariable(in, modelName, "FD_CBUS", row.Cbus);
in = setModelVariable(in, modelName, "FD_CBUS_ESR", row.CbusESR);
in = setModelVariable(in, modelName, ...
    "FD_VBUS_BIAS", row.VbusBias);
in = setModelVariable(in, modelName, ...
    "FD_VBAT_BIAS", row.VbatBias);
in = setModelVariable(in, modelName, ...
    "FD_IBAT_BIAS", row.IbatBias);
in = setModelVariable(in, modelName, "FD_IL_BIAS", row.ILBias);
in = setModelVariable(in, modelName, ...
    "FD_DUTY_BIAS", row.DutyBias);
in = setModelVariable(in, modelName, ...
    "FD_DUTY_STUCK_ENABLE", row.DutyStuckEnable);
in = setModelVariable(in, modelName, ...
    "FD_DUTY_STUCK_VALUE", row.DutyStuckValue);
in = setModelVariable(in, modelName, ...
    "FD_LOAD_STEP_A", row.LoadStepA);
in = setModelVariable(in, modelName, "FD_S1_OPEN", row.S1Open);
in = setModelVariable(in, modelName, "FD_S2_OPEN", row.S2Open);
end

function in = setModelVariable( ...
    in, modelName, variableName, value)
in = in.setVariable( ...
    variableName, value, "Workspace", modelName);
end

function data = sampleTrace(trace, queryTime)
data = interp1( ...
    double(trace.Time(:)), double(trace.Data(:)), ...
    queryTime, "linear", "extrap");
end

function featureTable = extractWindowFeatures( ...
    t, data, windowLength, stride, row, faultTime)

sampleTime = median(diff(t));
windowSamples = max(4, round(windowLength / sampleTime));
strideSamples = max(1, round(stride / sampleTime));
startIndex = 1:strideSamples:(numel(t) - windowSamples + 1);
numWindows = numel(startIndex);

featureTable = table( ...
    repmat(string(row.Scenario), numWindows, 1), ...
    repmat(string(row.RunID), numWindows, 1), ...
    repmat(string(row.FaultName), numWindows, 1), ...
    repmat(row.FaultID, numWindows, 1), ...
    zeros(numWindows, 1), ...
    repmat(row.ModeCommand, numWindows, 1), ...
    repmat(row.Severity, numWindows, 1), ...
    repmat(row.BiasSign, numWindows, 1), ...
    zeros(numWindows, 1), ...
    false(numWindows, 1), false(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), zeros(numWindows, 1), ...
    zeros(numWindows, 1), ...
    'VariableNames', [ ...
    "Scenario", "RunID", "FaultName", "ExpectedFaultID", ...
    "FaultID", "ModeCommand", "Severity", "BiasSign", ...
    "WindowStart", "FaultActive", "TransitionWindow", ...
    "VbusMean", "VbusStd", "VbusRange", "VbusSlope", ...
    "VbatMean", "VbatStd", ...
    "ILMean", "ILRMS", "ILStd", "ILRange", "ILSlope", ...
    "IbatMean", "IbatRMS", "IbatStd", "IbatRange", ...
    "IbatSlope", "CurrentPairResidualMean", ...
    "CurrentPairResidualRMS", ...
    "CurrentTrackingErrorRMSE", ...
    "CurrentTrackingErrorMAE", ...
    "DutyMean", "DutyStd", "DutyResidualRMS"]);

for windowIndex = 1:numWindows
    idx = startIndex(windowIndex): ...
        (startIndex(windowIndex) + windowSamples - 1);
    localTime = t(idx);
    localVbus = data.VbusMeas(idx);
    localVbat = data.VbatMeas(idx);
    localIL = data.ILMeas(idx);
    localIbat = data.IbatMeas(idx);
    localTrackingError = localIL - data.Iref(idx);
    localCurrentPairResidual = localIbat + localIL;
    localDuty = data.DutyApplied(idx);
    localDutyResidual = ...
        data.DutyApplied(idx) - data.DutyCommand(idx);

    startsBeforeFault = localTime(1) < faultTime;
    endsAfterFault = localTime(end) >= faultTime;
    isTransitionWindow = startsBeforeFault && endsAfterFault;
    isFaultActive = row.FaultID ~= 0 && localTime(1) >= faultTime;

    featureTable.WindowStart(windowIndex) = localTime(1);
    featureTable.FaultActive(windowIndex) = isFaultActive;
    featureTable.TransitionWindow(windowIndex) = isTransitionWindow;
    if isFaultActive
        featureTable.FaultID(windowIndex) = row.FaultID;
    end

    featureTable.VbusMean(windowIndex) = mean(localVbus);
    featureTable.VbusStd(windowIndex) = std(localVbus);
    featureTable.VbusRange(windowIndex) = ...
        max(localVbus) - min(localVbus);
    featureTable.VbusSlope(windowIndex) = ...
        linearSlope(localTime, localVbus);
    featureTable.VbatMean(windowIndex) = mean(localVbat);
    featureTable.VbatStd(windowIndex) = std(localVbat);

    featureTable.ILMean(windowIndex) = mean(localIL);
    featureTable.ILRMS(windowIndex) = sqrt(mean(localIL .^ 2));
    featureTable.ILStd(windowIndex) = std(localIL);
    featureTable.ILRange(windowIndex) = ...
        max(localIL) - min(localIL);
    featureTable.ILSlope(windowIndex) = ...
        linearSlope(localTime, localIL);

    featureTable.IbatMean(windowIndex) = mean(localIbat);
    featureTable.IbatRMS(windowIndex) = ...
        sqrt(mean(localIbat .^ 2));
    featureTable.IbatStd(windowIndex) = std(localIbat);
    featureTable.IbatRange(windowIndex) = ...
        max(localIbat) - min(localIbat);
    featureTable.IbatSlope(windowIndex) = ...
        linearSlope(localTime, localIbat);

    featureTable.CurrentPairResidualMean(windowIndex) = ...
        mean(localCurrentPairResidual);
    featureTable.CurrentPairResidualRMS(windowIndex) = ...
        sqrt(mean(localCurrentPairResidual .^ 2));
    featureTable.CurrentTrackingErrorRMSE(windowIndex) = ...
        sqrt(mean(localTrackingError .^ 2));
    featureTable.CurrentTrackingErrorMAE(windowIndex) = ...
        mean(abs(localTrackingError));

    featureTable.DutyMean(windowIndex) = mean(localDuty);
    featureTable.DutyStd(windowIndex) = std(localDuty);
    featureTable.DutyResidualRMS(windowIndex) = ...
        sqrt(mean(localDutyResidual .^ 2));
end
end

function slope = linearSlope(t, data)
timeOffset = t - mean(t);
slope = sum(timeOffset .* (data - mean(data))) / ...
    sum(timeOffset .^ 2);
end

function createSensorBiasFigure( ...
    out, scenario, faultTime, stopTime, resultsFolder)

selectedNames = [ ...
    "vbus_bias_pos_mode2", ...
    "vbat_bias_pos_mode2", ...
    "il_bias_pos_mode2", ...
    "ibat_bias_pos_mode2"];
measuredSignal = [ ...
    "log_Vbus", "log_Vbat", "log_I_L", "log_Ibat"];
trueSignal = [ ...
    "log_Vbus_true", "log_Vbat_true", ...
    "log_IL_true", "log_Ibat_true"];
yLabel = [ ...
    "V_{bus} (V)", "V_{bat} (V)", ...
    "I_L (A)", "I_{bat} (A)"];

figureHandle = figure( ...
    "Color", "white", "Name", "Sensor bias validation");
layout = tiledlayout(4, 1, "TileSpacing", "compact");
title(layout, "Sensor-bias true and measured signals");

for plotIndex = 1:numel(selectedNames)
    scenarioIndex = find( ...
        scenario.Scenario == selectedNames(plotIndex), 1);
    measured = out(scenarioIndex).get(measuredSignal(plotIndex));
    truth = out(scenarioIndex).get(trueSignal(plotIndex));

    nexttile;
    hold on;
    measuredIndex = measured.Time >= 0.45 & ...
        measured.Time <= stopTime;
    truthIndex = truth.Time >= 0.45 & truth.Time <= stopTime;
    plot(measured.Time(measuredIndex), ...
        measured.Data(measuredIndex), ...
        "DisplayName", "measured");
    plot(truth.Time(truthIndex), truth.Data(truthIndex), ...
        "--", "DisplayName", "true");
    xline(faultTime, "--k", "fault time", ...
        "HandleVisibility", "off");
    grid on;
    ylabel(yLabel(plotIndex));
    legend("Location", "best");
end

xlabel("Time (s)");
exportgraphics(figureHandle, ...
    fullfile(resultsFolder, "sensor_bias_true_measured.png"), ...
    "Resolution", 180);
end
