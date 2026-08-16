%% Read-only audit of switch observability and logging in the v05 model.

scriptFolder = string(fileparts(mfilename("fullpath")));
projectRoot = string(fileparts(fileparts(fileparts(fileparts(scriptFolder)))));
modelPath = fullfile(projectRoot, "simulink", "models", ...
    "main_model_fd_v05_energyprotect.slx");
outputFolder = fullfile(projectRoot, "ML", "results", ...
    "high_resistance_model_audit");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

[modelFolder, modelName] = fileparts(modelPath);
addpath(modelFolder);
load_system(modelPath);
cleanup = onCleanup(@() close_system(modelName, 0)); %#ok<NASGU>

allBlocks = string(find_system(modelName, ...
    "LookUnderMasks", "all", "FollowLinks", "on", "Type", "Block"));
names = arrayfun(@(path) string(get_param(path, "Name")), allBlocks);
blockTypes = arrayfun(@(path) string(get_param(path, "BlockType")), allBlocks);
maskTypes = arrayfun(@(path) safeGet(path, "MaskType"), allBlocks);
referenceBlocks = arrayfun(@(path) safeGet(path, "ReferenceBlock"), allBlocks);

switchPattern = "(?i)(^|[^a-z0-9])(s1|s2)([^a-z0-9]|$)|switch|mosfet|igbt|buck|boost|converter";
isRelevant = ~cellfun("isempty", regexp(cellstr(names), switchPattern, "once")) | ...
    ~cellfun("isempty", regexp(cellstr(maskTypes), switchPattern, "once")) | ...
    ~cellfun("isempty", regexp(cellstr(referenceBlocks), switchPattern, "once"));
relevantBlocks = allBlocks(isRelevant);

inventory = repmat(struct( ...
    "Path", "", "SID", "", "Name", "", "BlockType", "", ...
    "MaskType", "", "ReferenceBlock", "", "PortCounts", struct(), ...
    "CandidateParameters", struct(), "Connectivity", []), ...
    numel(relevantBlocks), 1);
for index = 1:numel(relevantBlocks)
    block = relevantBlocks(index);
    inventory(index).Path = block;
    inventory(index).SID = string(Simulink.ID.getSID(block));
    inventory(index).Name = string(get_param(block, "Name"));
    inventory(index).BlockType = string(get_param(block, "BlockType"));
    inventory(index).MaskType = safeGet(block, "MaskType");
    inventory(index).ReferenceBlock = safeGet(block, "ReferenceBlock");
    inventory(index).PortCounts = summarizePorts(block);
    inventory(index).CandidateParameters = summarizeParameters(block);
    inventory(index).Connectivity = summarizeConnectivity(block);
end

toWorkspaceBlocks = string(find_system(modelName, ...
    "LookUnderMasks", "all", "FollowLinks", "on", ...
    "BlockType", "ToWorkspace"));
logging = repmat(struct("Path", "", "SID", "", "VariableName", "", ...
    "SaveFormat", "", "SampleTime", ""), numel(toWorkspaceBlocks), 1);
for index = 1:numel(toWorkspaceBlocks)
    block = toWorkspaceBlocks(index);
    logging(index).Path = block;
    logging(index).SID = string(Simulink.ID.getSID(block));
    logging(index).VariableName = safeGet(block, "VariableName");
    logging(index).SaveFormat = safeGet(block, "SaveFormat");
    logging(index).SampleTime = safeGet(block, "SampleTime");
end

subsystems = string(find_system(modelName, ...
    "SearchDepth", 2, "BlockType", "SubSystem"));
report = struct();
report.ModelPath = modelPath;
report.ModelName = modelName;
report.TotalBlocks = numel(allBlocks);
report.TopLevelSubsystems = subsystems;
report.RelevantBlocks = inventory;
report.ToWorkspaceBlocks = logging;
report.GeneratedAt = string(datetime("now", "TimeZone", "Asia/Shanghai"));

jsonPath = fullfile(outputFolder, "model_inventory.json");
fid = fopen(jsonPath, "w", "n", "UTF-8");
assert(fid >= 0, "Could not open audit output: %s", jsonPath);
fileCleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, "%s", jsonencode(report, "PrettyPrint", true));

fprintf("Model: %s\n", modelName);
fprintf("Total blocks: %d\n", numel(allBlocks));
fprintf("Relevant switch/converter blocks: %d\n", numel(relevantBlocks));
fprintf("To Workspace blocks: %d\n", numel(toWorkspaceBlocks));
fprintf("Audit: %s\n", jsonPath);

function value = safeGet(block, parameter)
try
    value = string(get_param(block, parameter));
catch
    value = "";
end
end

function counts = summarizePorts(block)
handles = get_param(block, "PortHandles");
counts = struct();
fields = fieldnames(handles);
for index = 1:numel(fields)
    field = fields{index};
    counts.(field) = numel(handles.(field));
end
end

function values = summarizeParameters(block)
candidateNames = ["Ron", "Resistance", "Rs", "Vf", "ForwardVoltage", ...
    "SnubberResistance", "SnubberCapacitance", "Measurements", ...
    "SwitchingDevice", "DeviceType"];
objectParameters = get_param(block, "ObjectParameters");
values = struct();
for name = candidateNames
    if isfield(objectParameters, name)
        field = matlab.lang.makeValidName(name);
        values.(field) = safeGet(block, name);
    end
end
end

function rows = summarizeConnectivity(block)
connectivity = get_param(block, "PortConnectivity");
rows = repmat(struct("Type", "", "Position", [], "Source", "", ...
    "Destinations", strings(0, 1)), numel(connectivity), 1);
for index = 1:numel(connectivity)
    item = connectivity(index);
    rows(index).Type = string(item.Type);
    rows(index).Position = item.Position;
    rows(index).Source = blockNameFromHandle(item.SrcBlock);
    destinationHandles = item.DstBlock;
    destinations = strings(numel(destinationHandles), 1);
    for destinationIndex = 1:numel(destinationHandles)
        destinations(destinationIndex) = ...
            blockNameFromHandle(destinationHandles(destinationIndex));
    end
    rows(index).Destinations = destinations;
end
end

function name = blockNameFromHandle(handle)
if isempty(handle) || all(handle == -1)
    name = "";
    return;
end
try
    name = string(getfullname(handle(1)));
catch
    name = "";
end
end
