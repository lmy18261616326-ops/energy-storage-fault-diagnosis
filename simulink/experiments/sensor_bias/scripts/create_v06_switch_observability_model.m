%% Create a v06 copy that logs the built-in IGBT measurement vectors.
% The v05 source model is loaded read-only and saved under a new name before
% any structural change.  Each IGBT measurement output is [device current,
% device voltage], as established from the library block's internal Mux.

scriptFolder = string(fileparts(mfilename("fullpath")));
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
modelFolder = fullfile(projectRoot, "simulink", "models");
sourcePath = fullfile(modelFolder, "main_model_fd_v05_energyprotect.slx");
destinationPath = fullfile(modelFolder, ...
    "main_model_fd_v06_switchobservability.slx");

assert(isfile(sourcePath), "Source model not found: %s", sourcePath);
assert(~isfile(destinationPath), ...
    "Destination already exists; refusing to overwrite: %s", destinationPath);

addpath(modelFolder);
[~, sourceModel] = fileparts(sourcePath);
[~, destinationModel] = fileparts(destinationPath);
load_system(sourcePath);
sourceCleanup = onCleanup(@() closeIfLoaded(sourceModel)); %#ok<NASGU>
save_system(sourceModel, destinationPath);
close_system(sourceModel, 0);

load_system(destinationPath);
destinationCleanup = onCleanup(@() closeIfLoaded(destinationModel)); %#ok<NASGU>

s1Block = Simulink.ID.getFullName(destinationModel + ":3");
s2Block = Simulink.ID.getFullName(destinationModel + ":4");
assert(string(get_param(s1Block, "MaskType")) == "IGBT/Diode");
assert(string(get_param(s2Block, "MaskType")) == "IGBT/Diode");
assert(string(get_param(s1Block, "Measurements")) == "on");
assert(string(get_param(s2Block, "Measurements")) == "on");

addSwitchLogger(destinationModel, s1Block, "S1", 650);
addSwitchLogger(destinationModel, s2Block, "S2", 850);

set_param(destinationModel, "SimulationCommand", "update");
verifyLogger(destinationModel, s1Block, "S1");
verifyLogger(destinationModel, s2Block, "S2");
save_system(destinationModel);
close_system(destinationModel, 0);

fprintf("Created switch-observability model: %s\n", destinationPath);

function addSwitchLogger(model, switchBlock, switchName, yBase)
demuxPath = model + "/" + switchName + "_Measurement_Demux";
currentPath = model + "/log_" + switchName + "_device_current";
voltagePath = model + "/log_" + switchName + "_device_voltage";

add_block("simulink/Signal Routing/Demux", demuxPath, ...
    "Outputs", "2", "Position", [1840 yBase 1845 yBase + 80]);
add_block("simulink/Sinks/To Workspace", currentPath, ...
    "VariableName", "log_" + switchName + "_device_current", ...
    "SaveFormat", "Timeseries", "SampleTime", "-1", ...
    "MaxDataPoints", "inf", "Decimation", "20", ...
    "Position", [1920 yBase 2070 yBase + 30]);
add_block("simulink/Sinks/To Workspace", voltagePath, ...
    "VariableName", "log_" + switchName + "_device_voltage", ...
    "SaveFormat", "Timeseries", "SampleTime", "-1", ...
    "MaxDataPoints", "inf", "Decimation", "20", ...
    "Position", [1920 yBase + 55 2070 yBase + 85]);

switchPorts = get_param(switchBlock, "PortHandles");
demuxPorts = get_param(demuxPath, "PortHandles");
currentPorts = get_param(currentPath, "PortHandles");
voltagePorts = get_param(voltagePath, "PortHandles");
assert(numel(switchPorts.Outport) == 1, ...
    "%s must expose exactly one measurement port.", switchBlock);
add_line(model, switchPorts.Outport(1), demuxPorts.Inport(1), ...
    "autorouting", "smart");
add_line(model, demuxPorts.Outport(1), currentPorts.Inport(1), ...
    "autorouting", "smart");
add_line(model, demuxPorts.Outport(2), voltagePorts.Inport(1), ...
    "autorouting", "smart");
end

function verifyLogger(model, switchBlock, switchName)
demuxPath = model + "/" + switchName + "_Measurement_Demux";
currentPath = model + "/log_" + switchName + "_device_current";
voltagePath = model + "/log_" + switchName + "_device_voltage";
switchPorts = get_param(switchBlock, "PortHandles");
demuxPorts = get_param(demuxPath, "PortHandles");
currentPorts = get_param(currentPath, "PortHandles");
voltagePorts = get_param(voltagePath, "PortHandles");
assert(get_param(switchPorts.Outport(1), "Line") ~= -1);
assert(get_param(demuxPorts.Inport(1), "Line") ~= -1);
assert(all(arrayfun(@(port) get_param(port, "Line") ~= -1, ...
    demuxPorts.Outport)));
assert(get_param(currentPorts.Inport(1), "Line") ~= -1);
assert(get_param(voltagePorts.Inport(1), "Line") ~= -1);
assert(string(get_param(currentPath, "VariableName")) == ...
    "log_" + switchName + "_device_current");
assert(string(get_param(voltagePath, "VariableName")) == ...
    "log_" + switchName + "_device_voltage");
end

function closeIfLoaded(model)
if bdIsLoaded(model)
    close_system(model, 0);
end
end
