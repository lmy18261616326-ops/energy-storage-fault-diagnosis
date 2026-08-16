function result = generate_reference_dataset(outputRoot)
%GENERATE_REFERENCE_DATASET Generate a synthetic bidirectional DC-DC dataset.
%   The generated values are engineering reference/simulation data, not
%   measurements. Every channel is synchronized to a 1 us time base.

arguments
    outputRoot (1,1) string = ""
end

scriptFolder = string(fileparts(mfilename("fullpath")));
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
if strlength(outputRoot) == 0
    outputRoot = fullfile(projectRoot, "output", ...
        "reference_experiment_dataset_2026-08-04");
end

if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

h5File = fullfile(outputRoot, "bidirectional_dcdc_reference_raw.h5");
manifestFile = fullfile(outputRoot, "run_manifest.csv");
summaryFile = fullfile(outputRoot, "run_summary.csv");
coverageFile = fullfile(outputRoot, "coverage_summary.csv");
receiptFile = fullfile(outputRoot, "qa_receipt.json");
readmeFile = fullfile(outputRoot, "README_CN.md");

if isfile(h5File)
    delete(h5File);
end

cfg = referenceConfig();
manifest = buildManifest(cfg);

signalNames = [ ...
    "vbus_ref_V", "vbus_true_V", "vbus_meas_V", ...
    "vbat_true_V", "vbat_meas_V", ...
    "il_ref_A", "il_true_A", "il_meas_A", ...
    "ibat_true_A", "ibat_meas_A", ...
    "load_current_A", "source_current_A", "soc_pct", ...
    "duty_cmd", ...
    "s1_gate_cmd", "s1_gate_actual", ...
    "s2_gate_cmd", "s2_gate_actual", ...
    "s1_device_voltage_V", "s1_device_current_A", ...
    "s1_equiv_active_current_A", ...
    "s2_device_voltage_V", "s2_device_current_A", ...
    "s2_equiv_active_current_A", ...
    "s1_on_resistance_ohm", "s2_on_resistance_ohm", ...
    "vbus_bias_actual_V", "il_bias_actual_A", ...
    "fault_trigger", "fault_active", "fault_effective", ...
    "fault_code", "mode_code", "temperature_C"];

signalUnits = [ ...
    "V", "V", "V", "V", "V", ...
    "A", "A", "A", "A", "A", "A", "A", "%", ...
    "1", "bool", "bool", "bool", "bool", ...
    "V", "A", "A", "V", "A", "A", ...
    "ohm", "ohm", "V", "A", ...
    "bool", "bool", "bool", "code", "code", "degC"];

t = single((0:cfg.SampleTime_s:cfg.StopTime_s)');
nSamples = numel(t);
nSignals = numel(signalNames);

h5create(h5File, "/time_s", [nSamples 1], ...
    "Datatype", "single", "ChunkSize", [min(nSamples, 8192) 1], ...
    "Deflate", 4);
h5write(h5File, "/time_s", t);
writeStringMatrix(h5File, "/meta/signal_names", signalNames);
writeStringMatrix(h5File, "/meta/signal_units", signalUnits);
h5writeatt(h5File, "/", "dataset_name", ...
    "Bidirectional DC-DC fault reference dataset");
h5writeatt(h5File, "/", "data_origin", ...
    "synthetic engineering reference data; NOT measured data");
h5writeatt(h5File, "/", "sample_interval_s", cfg.SampleTime_s);
h5writeatt(h5File, "/", "sample_rate_Hz", 1/cfg.SampleTime_s);
h5writeatt(h5File, "/", "switching_frequency_Hz", cfg.SwitchFrequency_Hz);
h5writeatt(h5File, "/", "nominal_bus_voltage_V", cfg.VbusRef_V);
h5writeatt(h5File, "/", "current_sign_convention", ...
    "positive=battery discharge to DC bus; negative=battery charge");
h5writeatt(h5File, "/", "matrix_layout", ...
    "rows=time samples; columns follow /meta/signal_names");

summaryRows = repmat(emptySummaryRow(), height(manifest), 1);
fprintf("Generating %d runs at %.0f Hz (%d samples/run)...\n", ...
    height(manifest), 1/cfg.SampleTime_s, nSamples);
for k = 1:height(manifest)
    row = manifest(k,:);
    [data, metrics, realized] = synthesizeRun(row, t, cfg);
    group = "/runs/" + row.RunID;
    datasetPath = group + "/signals";
    h5create(h5File, datasetPath, [nSamples nSignals], ...
        "Datatype", "single", ...
        "ChunkSize", [min(nSamples, 4096) nSignals], "Deflate", 4);
    h5write(h5File, datasetPath, data);
    h5writeatt(h5File, group, "RunID", char(row.RunID));
    h5writeatt(h5File, group, "fault_group", char(row.FaultGroup));
    h5writeatt(h5File, group, "fault_subtype", char(row.FaultSubtype));
    h5writeatt(h5File, group, "mode", char(row.Mode));
    h5writeatt(h5File, group, "fault_start_s", row.FaultStart_s);
    h5writeatt(h5File, group, "fault_end_s", row.FaultEnd_s);
    h5writeatt(h5File, group, "random_seed", row.RandomSeed);

    manifest.ActualBias{k} = realized.ActualBias;
    manifest.ActualResistance_mOhm(k) = realized.ActualResistance_mOhm;
    manifest.HDF5Group(k) = group;
    summaryRows(k) = metrics;

    if mod(k, 10) == 0 || k == height(manifest)
        fprintf("  completed %d/%d\n", k, height(manifest));
    end
end

summary = struct2table(summaryRows, "AsArray", true);
coverage = buildCoverage(manifest);

writetable(manifest, manifestFile);
writetable(summary, summaryFile);
writetable(coverage, coverageFile);
writeReadme(readmeFile, cfg, signalNames, signalUnits, height(manifest));

allCoveragePass = all(coverage.Pass);
receipt = struct( ...
    "dataset_origin", "synthetic_reference_not_measured", ...
    "run_count", height(manifest), ...
    "samples_per_run", nSamples, ...
    "signal_count", nSignals, ...
    "sample_interval_s", cfg.SampleTime_s, ...
    "switching_frequency_Hz", cfg.SwitchFrequency_Hz, ...
    "all_required_coverage_pass", allCoveragePass, ...
    "hdf5_file", char(h5File), ...
    "generated_at", char(datetime("now", "TimeZone", "Asia/Shanghai", ...
        "Format", "yyyy-MM-dd'T'HH:mm:ssXXX")));
fid = fopen(receiptFile, "w", "n", "UTF-8");
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s", jsonencode(receipt, "PrettyPrint", true));
clear cleanup

assert(height(manifest) == 162, "Unexpected run count.");
assert(allCoveragePass, "At least one coverage requirement failed.");
assert(nSamples == 20001, "Unexpected sample count.");
assert(isfile(h5File) && isfile(manifestFile) && isfile(summaryFile));

result = struct( ...
    "OutputRoot", outputRoot, ...
    "HDF5File", h5File, ...
    "ManifestFile", manifestFile, ...
    "SummaryFile", summaryFile, ...
    "CoverageFile", coverageFile, ...
    "ReceiptFile", receiptFile, ...
    "RunCount", height(manifest), ...
    "SamplesPerRun", nSamples, ...
    "SignalCount", nSignals, ...
    "CoveragePass", allCoveragePass);
save(fullfile(outputRoot, "generation_result.mat"), "result", "cfg", ...
    "signalNames", "signalUnits", "-v7");
fprintf("Reference dataset written to: %s\n", outputRoot);
end

function cfg = referenceConfig()
cfg = struct();
cfg.SampleTime_s = 1e-6;
cfg.StopTime_s = 0.020;
cfg.SwitchFrequency_Hz = 20e3;
cfg.VbusRef_V = 400;
cfg.NominalSwitchResistance_Ohm = 1e-3;
cfg.DeadTime_s = 1e-6;
cfg.CurrentLevels_A = [5 10 15];
cfg.Temperatures_C = [25 45];
cfg.HighResistanceLevels_mOhm = [5 8 10 12 15 20 50];
cfg.BatteryCapacity_Ah = 50;
cfg.RandomSeedBase = 2026080400;
end

function manifest = buildManifest(cfg)
rows = repmat(emptyManifestRow(), 162, 1);
idx = 0;

% 36 healthy = 2 directions x 3 currents x 2 temperatures x 3 repeats.
for k = 1:36
    [idx, rows] = addCase(rows, idx, cfg, "healthy", "none", ...
        "none", 0, 0, NaN, k, 0);
end

% Sensor bias cases. Magnitudes include both polarities.
vBiasLevels = [-10 -8 -6 -4 -2 2 4 6 8 10];
for k = 1:18
    magnitude = vBiasLevels(mod(k-1, numel(vBiasLevels))+1);
    [idx, rows] = addCase(rows, idx, cfg, "vbus_bias", ...
        "sensor_bias", "Vbus_sensor", magnitude, 1, NaN, k, 100);
end
iBiasLevels = [-1.5 -1.0 -0.75 -0.5 -0.25 0.25 0.5 0.75 1.0 1.5];
for k = 1:18
    magnitude = iBiasLevels(mod(k-1, numel(iBiasLevels))+1);
    [idx, rows] = addCase(rows, idx, cfg, "il_bias", ...
        "sensor_bias", "inductor_current_sensor", magnitude, 2, NaN, k, 200);
end

% S1/S2 full, partial and intermittent open: 8 runs per subtype.
subtypes = ["full_open" "partial_open" "intermittent_open"];
for switchName = ["S1" "S2"]
    faultCode = 3 + double(switchName == "S2");
    switchOffset = 300 + 100*double(switchName == "S2");
    localIndex = 0;
    for s = 1:numel(subtypes)
        for k = 1:8
            localIndex = localIndex + 1;
            if subtypes(s) == "full_open"
                magnitude = 1;
            elseif subtypes(s) == "partial_open"
                partialLevels = [0.25 0.50 0.75];
                magnitude = partialLevels(mod(k-1, 3)+1);
            else
                magnitude = 0.50;
            end
            [idx, rows] = addCase(rows, idx, cfg, switchName + "_open", ...
                subtypes(s), switchName, magnitude, faultCode, NaN, ...
                localIndex, switchOffset);
        end
    end
end

% 42 high-resistance cases = 2 switches x 7 values x 3 repeats.
for switchName = ["S1" "S2"]
    faultCode = 5 + double(switchName == "S2");
    switchOffset = 500 + 100*double(switchName == "S2");
    localIndex = 0;
    for r = cfg.HighResistanceLevels_mOhm
        for rep = 1:3
            localIndex = localIndex + 1;
            [idx, rows] = addCase(rows, idx, cfg, "high_resistance", ...
                switchName + "_high_resistance", switchName, r, ...
                faultCode, r, localIndex, switchOffset);
        end
    end
end

assert(idx == numel(rows));
manifest = struct2table(rows, "AsArray", true);
manifest.ActualBias = cell(height(manifest), 1);
manifest.ActualResistance_mOhm = nan(height(manifest), 1);
manifest.HDF5Group = strings(height(manifest), 1);
end

function [idx, rows] = addCase(rows, idx, cfg, faultGroup, subtype, ...
        location, magnitude, faultCode, resistance_mOhm, localIndex, offset)
idx = idx + 1;
combo = mod(localIndex-1, 12);
modeIndex = mod(combo, 2) + 1;
currentIndex = mod(floor(combo/2), 3) + 1;
tempIndex = mod(floor(combo/6), 2) + 1;
modeNames = ["charge" "discharge"];
modeSigns = [-1 1];

rows(idx).RunID = compose("RUN_%04d", idx);
rows(idx).FaultGroup = string(faultGroup);
rows(idx).FaultSubtype = string(subtype);
rows(idx).FaultLocation = string(location);
rows(idx).FaultCode = faultCode;
rows(idx).CommandedMagnitude = magnitude;
rows(idx).CommandedResistance_mOhm = resistance_mOhm;
rows(idx).Mode = modeNames(modeIndex);
rows(idx).ModeCode = modeSigns(modeIndex);
rows(idx).CurrentLevel_A = cfg.CurrentLevels_A(currentIndex);
rows(idx).Iref_A = modeSigns(modeIndex) * rows(idx).CurrentLevel_A;
rows(idx).Temperature_C = cfg.Temperatures_C(tempIndex);
rows(idx).SOCInit_pct = 55 + 10*modeSigns(modeIndex);
rows(idx).VbusRef_V = cfg.VbusRef_V;
rows(idx).Pload_W = 500 + 50*rows(idx).CurrentLevel_A;
rows(idx).SwitchFrequency_Hz = cfg.SwitchFrequency_Hz;
rows(idx).SampleInterval_s = cfg.SampleTime_s;
rows(idx).StopTime_s = cfg.StopTime_s;
rows(idx).RandomSeed = cfg.RandomSeedBase + offset + localIndex;

if faultCode == 0
    rows(idx).FaultStart_s = NaN;
    rows(idx).FaultEnd_s = NaN;
else
    startSample = round((0.0075 + 0.00015*mod(localIndex-1, 10)) / ...
        cfg.SampleTime_s);
    rows(idx).FaultStart_s = startSample * cfg.SampleTime_s;
    if subtype == "full_open"
        rows(idx).FaultEnd_s = min(rows(idx).FaultStart_s + 0.004, ...
            cfg.StopTime_s);
    elseif subtype == "partial_open"
        rows(idx).FaultEnd_s = min(rows(idx).FaultStart_s + 0.008, ...
            cfg.StopTime_s);
    else
        rows(idx).FaultEnd_s = cfg.StopTime_s;
    end
end
end

function row = emptyManifestRow()
row = struct( ...
    "RunID", "", "FaultGroup", "", "FaultSubtype", "", ...
    "FaultLocation", "", "FaultCode", 0, ...
    "CommandedMagnitude", 0, "CommandedResistance_mOhm", NaN, ...
    "Mode", "", "ModeCode", 0, "CurrentLevel_A", 0, ...
    "Iref_A", 0, "Temperature_C", 0, "SOCInit_pct", 0, ...
    "VbusRef_V", 0, "Pload_W", 0, "SwitchFrequency_Hz", 0, ...
    "SampleInterval_s", 0, "FaultStart_s", NaN, ...
    "FaultEnd_s", NaN, "StopTime_s", 0, "RandomSeed", 0);
end

function [data, metrics, realized] = synthesizeRun(row, tSingle, cfg)
t = double(tSingle);
dt = cfg.SampleTime_s;
rng(row.RandomSeed, "twister");
n = numel(t);
f = cfg.SwitchFrequency_Hz;
period = 1/f;
direction = row.ModeCode;
iref = row.Iref_A;
temperature = row.Temperature_C;

if isnan(row.FaultStart_s)
    faultWindow = false(n,1);
    trigger = false(n,1);
else
    faultWindow = t >= row.FaultStart_s & t <= row.FaultEnd_s;
    trigger = false(n,1);
    [~, triggerIndex] = min(abs(t-row.FaultStart_s));
    trigger(triggerIndex) = true;
end

actualVbusBias = 0;
actualILBias = 0;
actualResistance_mOhm = NaN;
if row.FaultGroup == "vbus_bias"
    actualVbusBias = row.CommandedMagnitude * (1 + 0.006*randn);
elseif row.FaultGroup == "il_bias"
    actualILBias = row.CommandedMagnitude * (1 + 0.008*randn);
elseif row.FaultGroup == "high_resistance"
    tempFactor = 1 + 0.004*(temperature-25);
    actualResistance_mOhm = row.CommandedResistance_mOhm * ...
        tempFactor * (1 + 0.008*randn);
end

faultEffective = faultWindow;
if row.FaultSubtype == "partial_open"
    cycleNumber = floor(t/period);
    pseudo = mod(sin((cycleNumber + row.RandomSeed)*12.9898)*43758.5453, 1);
    faultEffective = faultWindow & pseudo < row.CommandedMagnitude;
elseif row.FaultSubtype == "intermittent_open"
    burstPhase = mod(max(t-row.FaultStart_s, 0), 0.004);
    faultEffective = faultWindow & burstPhase < 0.002;
end

faultState = firstOrderState(double(faultEffective), dt, 0.00025);
sensorState = firstOrderState(double(faultWindow), dt, 0.00040);
startup = 1-exp(-t/0.00070);

impact = 0;
voltageImpact_V = 0;
if contains(row.FaultGroup, "_open")
    isMainSwitch = (row.FaultLocation == "S1" && direction > 0) || ...
        (row.FaultLocation == "S2" && direction < 0);
    directionFactor = 0.92*double(isMainSwitch) + 0.62*double(~isMainSwitch);
    if row.FaultSubtype == "full_open"
        impact = 0.84*directionFactor;
    elseif row.FaultSubtype == "partial_open"
        impact = (0.18 + 0.62*row.CommandedMagnitude)*directionFactor;
    else
        impact = 0.72*directionFactor;
    end
    voltageImpact_V = 13.0*impact;
elseif row.FaultGroup == "high_resistance"
    excessResistance = max(actualResistance_mOhm-1, 0)/1000;
    impact = min(0.10, 0.40*excessResistance*abs(iref));
    voltageImpact_V = 1.6*impact + excessResistance*abs(iref)*0.8;
end

iMean = iref .* startup .* (1-impact*faultState) ...
    - actualILBias*sensorState;
rippleAmplitude = 0.22 + 0.018*abs(iref) + 0.20*impact*faultState;
phase = mod(t, period)/period;
triangle = 2*abs(2*phase-1)-1;
ilTrue = iMean + direction*rippleAmplitude.*triangle ...
    + 0.025*sin(2*pi*850*t + 0.1*row.RandomSeed);

ibatTrue = 0.985*ilTrue + 0.018*sin(2*pi*f*t + 0.3);
soc = row.SOCInit_pct - cumtrapz(t, ibatTrue) / ...
    (3600*cfg.BatteryCapacity_Ah)*100;
ocv = 190 + 0.35*soc - 0.025*(temperature-25);
rbat = 0.50*(1 - 0.003*(temperature-25));
vbatTrue = ocv - rbat*ibatTrue + 0.035*sin(2*pi*f*t + 1.1);

vFault = -direction*voltageImpact_V*faultState;
vSensorControl = -0.85*actualVbusBias*sensorState;
vbusTrue = row.VbusRef_V + vFault + vSensorControl ...
    + 0.34*sin(2*pi*f*t + 0.5) ...
    + 0.08*sin(2*pi*900*t + 0.2*row.RandomSeed);

vbusMeas = quantize(vbusTrue + actualVbusBias*double(faultWindow) ...
    + 0.05*randn(n,1), 0.01);
vbatMeas = quantize(vbatTrue + 0.02*randn(n,1), 0.01);
ilMeas = quantize(ilTrue + actualILBias*double(faultWindow) ...
    + 0.02*randn(n,1), 0.001);
ibatMeas = quantize(ibatTrue + 0.01*randn(n,1), 0.001);

dutyFF = vbatTrue ./ max(vbusTrue, 1);
dutyCmd = min(max(dutyFF + 0.0022*(iref-ilMeas), 0.08), 0.92);
deadFraction = cfg.DeadTime_s/period;
s1GateCmd = phase >= deadFraction & phase < max(dutyCmd-deadFraction, 0);
s2GateCmd = phase >= min(dutyCmd+deadFraction, 1) & phase < 1-deadFraction;
s1GateActual = s1GateCmd;
s2GateActual = s2GateCmd;
if row.FaultLocation == "S1" && contains(row.FaultGroup, "_open")
    s1GateActual(faultEffective) = false;
elseif row.FaultLocation == "S2" && contains(row.FaultGroup, "_open")
    s2GateActual(faultEffective) = false;
end

nominalRon = cfg.NominalSwitchResistance_Ohm * ...
    (1 + 0.004*(temperature-25));
s1Ron = nominalRon*ones(n,1);
s2Ron = nominalRon*ones(n,1);
if row.FaultGroup == "high_resistance" && row.FaultLocation == "S1"
    s1Ron(faultWindow) = actualResistance_mOhm/1000;
elseif row.FaultGroup == "high_resistance" && row.FaultLocation == "S2"
    s2Ron(faultWindow) = actualResistance_mOhm/1000;
end

s1Current = ilTrue.*double(s1GateActual);
s2Current = -ilTrue.*double(s2GateActual);
s1Voltage = double(~s1GateActual).*vbusTrue + ...
    double(s1GateActual).*abs(s1Current).*s1Ron;
s2Voltage = double(~s2GateActual).*vbusTrue + ...
    double(s2GateActual).*abs(s2Current).*s2Ron;
s1EquivCurrent = abs(s1Current);
s2EquivCurrent = abs(s2Current);

loadCurrent = row.Pload_W ./ max(vbusTrue, 1);
sourceCurrent = loadCurrent - vbatTrue.*ibatTrue./max(vbusTrue, 1);

vbusBiasSignal = actualVbusBias*double(faultWindow);
ilBiasSignal = actualILBias*double(faultWindow);
faultCodeSignal = row.FaultCode*ones(n,1);
modeCodeSignal = row.ModeCode*ones(n,1);
temperatureSignal = temperature*ones(n,1);
vbusRefSignal = row.VbusRef_V*ones(n,1);
ilRefSignal = iref*ones(n,1);

data = single([ ...
    vbusRefSignal, vbusTrue, vbusMeas, vbatTrue, vbatMeas, ...
    ilRefSignal, ilTrue, ilMeas, ibatTrue, ibatMeas, ...
    loadCurrent, sourceCurrent, soc, dutyCmd, ...
    double(s1GateCmd), double(s1GateActual), ...
    double(s2GateCmd), double(s2GateActual), ...
    s1Voltage, s1Current, s1EquivCurrent, ...
    s2Voltage, s2Current, s2EquivCurrent, ...
    s1Ron, s2Ron, vbusBiasSignal, ilBiasSignal, ...
    double(trigger), double(faultWindow), double(faultEffective), ...
    faultCodeSignal, modeCodeSignal, temperatureSignal]);

preMask = t >= 0.004 & t < 0.007;
if isnan(row.FaultStart_s)
    postMask = t >= 0.014 & t <= 0.019;
else
    postStart = min(row.FaultStart_s + 0.001, 0.015);
    postMask = t >= postStart & t <= min(postStart + 0.004, cfg.StopTime_s);
end
metrics = emptySummaryRow();
metrics.RunID = row.RunID;
metrics.FaultGroup = row.FaultGroup;
metrics.FaultSubtype = row.FaultSubtype;
metrics.Mode = row.Mode;
metrics.CurrentLevel_A = row.CurrentLevel_A;
metrics.Temperature_C = row.Temperature_C;
metrics.VbusPreMean_V = mean(vbusMeas(preMask));
metrics.VbusPostMean_V = mean(vbusMeas(postMask));
metrics.VbusPostMin_V = min(vbusMeas(postMask));
metrics.VbusPostMax_V = max(vbusMeas(postMask));
metrics.ILPreMean_A = mean(ilMeas(preMask));
metrics.ILPostMean_A = mean(ilMeas(postMask));
metrics.ILPostRMS_A = rms(ilMeas(postMask));
metrics.DutyPostMean = mean(dutyCmd(postMask));
metrics.S1MismatchCount = nnz(s1GateCmd(postMask) ~= s1GateActual(postMask));
metrics.S2MismatchCount = nnz(s2GateCmd(postMask) ~= s2GateActual(postMask));
metrics.S1RonPost_mOhm = 1000*mean(s1Ron(postMask));
metrics.S2RonPost_mOhm = 1000*mean(s2Ron(postMask));

if row.FaultGroup == "vbus_bias"
    realized.ActualBias = compose("%.6f V", actualVbusBias);
elseif row.FaultGroup == "il_bias"
    realized.ActualBias = compose("%.6f A", actualILBias);
else
    realized.ActualBias = "0";
end
realized.ActualResistance_mOhm = actualResistance_mOhm;
end

function y = firstOrderState(u, dt, tau)
alpha = exp(-dt/tau);
y = filter(1-alpha, [1 -alpha], u);
end

function y = quantize(x, step)
y = step*round(x/step);
end

function row = emptySummaryRow()
row = struct( ...
    "RunID", "", "FaultGroup", "", "FaultSubtype", "", ...
    "Mode", "", "CurrentLevel_A", 0, "Temperature_C", 0, ...
    "VbusPreMean_V", 0, "VbusPostMean_V", 0, ...
    "VbusPostMin_V", 0, "VbusPostMax_V", 0, ...
    "ILPreMean_A", 0, "ILPostMean_A", 0, "ILPostRMS_A", 0, ...
    "DutyPostMean", 0, "S1MismatchCount", 0, ...
    "S2MismatchCount", 0, "S1RonPost_mOhm", 0, ...
    "S2RonPost_mOhm", 0);
end

function coverage = buildCoverage(manifest)
items = [ ...
    "健康运行"; "母线电压偏置"; "电感电流偏置"; ...
    "S1 开路/部分开路/间歇开路"; ...
    "S2 开路/部分开路/间歇开路"; "S1/S2 高阻合计"];
minimum = [30; 15; 15; 20; 20; 40];
actual = [ ...
    nnz(manifest.FaultGroup == "healthy"); ...
    nnz(manifest.FaultGroup == "vbus_bias"); ...
    nnz(manifest.FaultGroup == "il_bias"); ...
    nnz(manifest.FaultGroup == "S1_open"); ...
    nnz(manifest.FaultGroup == "S2_open"); ...
    nnz(manifest.FaultGroup == "high_resistance")];
coverage = table(items, minimum, actual, actual >= minimum, ...
    'VariableNames', ["Requirement", "MinimumCount", "ActualCount", "Pass"]);
end

function writeStringMatrix(h5File, path, values)
matrix = uint8(char(values));
h5create(h5File, path, size(matrix), "Datatype", "uint8");
h5write(h5File, path, matrix);
end

function writeReadme(fileName, cfg, signalNames, signalUnits, runCount)
fid = fopen(fileName, "w", "n", "UTF-8");
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "# 双向 DC-DC 故障实验参考数据\n\n");
fprintf(fid, "> 重要：本数据集为按工程规律生成的合成参考数据，不是真实台架测量值。\n\n");
fprintf(fid, "- Run 数：%d\n", runCount);
fprintf(fid, "- 单 Run 时长：%.3f s\n", cfg.StopTime_s);
fprintf(fid, "- 同步采样间隔：%.0f us（%.0f Hz）\n", ...
    cfg.SampleTime_s*1e6, 1/cfg.SampleTime_s);
fprintf(fid, "- PWM：%.0f kHz；母线参考：%.0f V\n", ...
    cfg.SwitchFrequency_Hz/1e3, cfg.VbusRef_V);
fprintf(fid, "- 电流方向：正值为电池向母线放电，负值为电池充电。\n");
fprintf(fid, "- HDF5：`bidirectional_dcdc_reference_raw.h5`。全局时间轴为 `/time_s`，每条实验为 `/runs/RUN_xxxx/signals`。\n");
fprintf(fid, "- 信号矩阵按行对应时间、按列对应 `/meta/signal_names`。\n\n");
fprintf(fid, "## 配套文件\n\n");
fprintf(fid, "- `run_manifest.csv`：每个 Run 的完整工况、指令值、实测/实现值与 HDF5 路径。\n");
fprintf(fid, "- `run_summary.csv`：故障前后均值、极值、RMS、门极失配和导通电阻摘要。\n");
fprintf(fid, "- `coverage_summary.csv`：对截图最低实验次数的自动核验。\n");
fprintf(fid, "- `qa_receipt.json`：机器可读的交付校验信息。\n\n");
fprintf(fid, "## 信号列\n\n");
fprintf(fid, "| 列号 | 字段 | 单位 |\n|---:|---|---|\n");
for k = 1:numel(signalNames)
    fprintf(fid, "| %d | `%s` | %s |\n", k, signalNames(k), signalUnits(k));
end
fprintf(fid, "\n## MATLAB 读取示例\n\n```matlab\n");
fprintf(fid, "t = h5read('bidirectional_dcdc_reference_raw.h5','/time_s');\n");
fprintf(fid, "x = h5read('bidirectional_dcdc_reference_raw.h5','/runs/RUN_0001/signals');\n");
fprintf(fid, "manifest = readtable('run_manifest.csv','TextType','string');\n");
fprintf(fid, "```\n\n");
fprintf(fid, "高阻覆盖的指令值为 5、8、10、12、15、20、50 mOhm；实际值考虑温度系数和小量测量误差，详见工况表。\n");
clear cleanup
end
