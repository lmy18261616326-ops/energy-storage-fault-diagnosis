clear; clc;

projectRoot = fileparts(mfilename('fullpath'));
modelsDir = fullfile(projectRoot, 'simulink', 'models');
cd(modelsDir);
addpath(modelsDir);

model = getenv('ENERGY_STORAGE_MODEL');
if isempty(model)
    model = 'main_model_fd_v05_energyprotect';
end
modelFile = fullfile(modelsDir, [model '.slx']);
assert(isfile(modelFile), 'Model file does not exist: %s', modelFile);
load_system(modelFile);

fprintf('== Top-level connection table ==\n');
rootLines = find_system(model, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
for k = 1:numel(rootLines)
    line = rootLines(k);
    srcPort = get_param(line, 'SrcPortHandle');
    dstPorts = get_param(line, 'DstPortHandle');
    srcText = portText(srcPort);
    if isempty(dstPorts)
        fprintf('%s -> <none>\n', srcText);
    else
        for d = 1:numel(dstPorts)
            fprintf('%s -> %s\n', srcText, portText(dstPorts(d)));
        end
    end
end

fprintf('\n== Block type counts ==\n');
blocks = find_system(model, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
    'MatchFilter', @Simulink.match.allVariants, 'Type', 'Block');
types = strings(numel(blocks), 1);
for k = 1:numel(blocks)
    types(k) = string(get_param(blocks{k}, 'BlockType'));
end
[uniqueTypes, ~, idx] = unique(types);
counts = accumarray(idx, 1);
[counts, order] = sort(counts, 'descend');
uniqueTypes = uniqueTypes(order);
for k = 1:numel(uniqueTypes)
    fprintf('%s: %d\n', uniqueTypes(k), counts(k));
end

fprintf('\n== Candidate algorithm / diagnosis blocks ==\n');
keywords = {'diagnos','fault','logic','control','controller','pi','pwm','soc','boost','buck','current','voltage'};
for k = 1:numel(blocks)
    nameLower = lower(blocks{k});
    maskLower = lower(string(get_param(blocks{k}, 'MaskType')));
    hit = false;
    for q = 1:numel(keywords)
        if contains(nameLower, keywords{q}) || contains(maskLower, keywords{q})
            hit = true;
            break;
        end
    end
    if hit
        fprintf('%s | BlockType=%s | MaskType=%s\n', blocks{k}, get_param(blocks{k}, 'BlockType'), get_param(blocks{k}, 'MaskType'));
    end
end

fprintf('\n== Logged signals and scopes ==\n');
for k = 1:numel(rootLines)
    line = rootLines(k);
    try
        name = get_param(line, 'Name');
        logging = get_param(line, 'DataLogging');
        if strcmp(logging, 'on') || ~isempty(name)
            fprintf('Line=%g Name="%s" DataLogging=%s Source=%s\n', line, name, logging, portText(get_param(line, 'SrcPortHandle')));
        end
    catch
    end
end
scopes = find_system(model, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
    'MatchFilter', @Simulink.match.allVariants, 'BlockType', 'Scope');
for k = 1:numel(scopes)
    fprintf('Scope: %s\n', scopes{k});
end

fprintf('\n== Main model variables found by Simulink.findVars ==\n');
try
    vars = Simulink.findVars(model);
    for k = 1:numel(vars)
        fprintf('%s | Source=%s | SourceType=%s | Users=%d\n', vars(k).Name, vars(k).Source, vars(k).SourceType, numel(vars(k).Users));
    end
catch ME
    fprintf('findVars failed: %s\n', ME.message);
end

bdclose(model);

function txt = portText(portHandle)
if isempty(portHandle) || portHandle == -1
    txt = '<unconnected>';
    return;
end
try
    blk = get_param(portHandle, 'Parent');
    portNumber = get_param(portHandle, 'PortNumber');
    portType = get_param(portHandle, 'PortType');
    txt = sprintf('%s.%s%d', blk, portType, portNumber);
catch
    txt = '<invalid-port>';
end
end
