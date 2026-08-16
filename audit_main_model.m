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
fprintf('AUDIT_MODEL=%s\n', model);
fprintf('AUDIT_MODEL_FILE=%s\n', modelFile);
fprintf('MATLAB_VERSION=%s\n', version);

try
    load_system(modelFile);
    fprintf('LOAD_SYSTEM=OK\n');
catch ME
    fprintf('LOAD_SYSTEM=ERROR\n%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    return;
end

fprintf('\n== Model configuration ==\n');
configParams = {'StopTime','Solver','SolverType','FixedStep','RelTol','AbsTol', ...
    'SignalLogging','SaveOutput','SaveTime','ReturnWorkspaceOutputs', ...
    'SimulationMode','InitFcn','PreLoadFcn','PostLoadFcn','StartFcn','StopFcn'};
for k = 1:numel(configParams)
    try
        fprintf('%s=%s\n', configParams{k}, string(get_param(model, configParams{k})));
    catch
        fprintf('%s=<unavailable>\n', configParams{k});
    end
end

fprintf('\n== Root blocks ==\n');
rootBlocks = find_system(model, 'SearchDepth', 1, 'Type', 'Block');
for k = 1:numel(rootBlocks)
    blk = rootBlocks{k};
    if strcmp(blk, model), continue; end
    fprintf('%s | BlockType=%s", MaskType="%s"\n', blk, get_param(blk, 'BlockType'), get_param(blk, 'MaskType'));
end

fprintf('\n== Subsystems, model refs, MATLAB functions, Stateflow candidates ==\n');
allBlocks = find_system(model, 'LookUnderMasks', 'all', ...
    'FollowLinks', 'on', 'Type', 'Block');
interestingKinds = {'SubSystem','ModelReference','MATLABSystem','S-Function'};
interesting = {};
for k = 1:numel(allBlocks)
    if ismember(get_param(allBlocks{k}, 'BlockType'), interestingKinds)
        interesting{end+1,1} = allBlocks{k}; %#ok<AGROW>
    end
end
for k = 1:numel(interesting)
    blk = interesting{k};
    fprintf('%s | BlockType=%s | MaskType=%s\n', blk, get_param(blk, 'BlockType'), get_param(blk, 'MaskType'));
end

fprintf('\n== Unconnected conventional signal lines (root level) ==\n');
lines = find_system(model, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
unconnectedCount = 0;
for k = 1:numel(lines)
    src = get_param(lines(k), 'SrcPortHandle');
    dst = get_param(lines(k), 'DstPortHandle');
    children = get_param(lines(k), 'LineChildren');
    portType = '';
    if src ~= -1
        try
            portType = get_param(src, 'PortType');
        catch
        end
    end
    % Specialized Power Systems electrical branches report connection-port
    % parents/children with -1 endpoints. They are not unconnected Simulink
    % signals, so only terminal conventional outport lines are audited here.
    if strcmp(portType, 'outport') && isempty(children) && ...
            (isempty(dst) || any(dst == -1))
        unconnectedCount = unconnectedCount + 1;
        try
            parent = get_param(lines(k), 'Parent');
        catch
            parent = '<unknown>';
        end
        fprintf('LineHandle=%g Parent=%s Src=%s Dst=%s\n', lines(k), parent, mat2str(src), mat2str(dst));
    end
end
fprintf('UNCONNECTED_SIGNAL_LINE_COUNT=%d\n', unconnectedCount);

fprintf('\n== Root blocks with unconnected conventional ports ==\n');
blocks = find_system(model, 'SearchDepth', 1, 'Type', 'Block');
unconnectedPortCount = 0;
for k = 1:numel(blocks)
    ph = get_param(blocks{k}, 'PortHandles');
    portGroups = {'Inport','Outport','Enable','Trigger','Ifaction','Reset','State'};
    missing = {};
    for g = 1:numel(portGroups)
        groupName = portGroups{g};
        if ~isfield(ph, groupName), continue; end
        handles = ph.(groupName);
        for p = 1:numel(handles)
            h = handles(p);
            if h == -1, continue; end
            try
                line = get_param(h, 'Line');
            catch
                line = [];
            end
            if isempty(line) || line == -1
                missing{end+1} = sprintf('%s%d', groupName, p); %#ok<AGROW>
            end
        end
    end
    if ~isempty(missing)
        unconnectedPortCount = unconnectedPortCount + numel(missing);
        fprintf('%s | %s\n', blocks{k}, strjoin(missing, ', '));
    end
end
fprintf('UNCONNECTED_PORT_COUNT=%d\n', unconnectedPortCount);

fprintf('\n== Variable references in block parameters ==\n');
try
    usages = Simulink.findVars(model);
    if isempty(usages)
        varNames = strings(0,1);
    else
        varNames = unique(string({usages.Name}));
    end
    fprintf('REFERENCED_VARIABLE_COUNT=%d\n', numel(varNames));
    for k = 1:numel(varNames)
        fprintf('%s\n', varNames(k));
    end
catch ME
    fprintf('VARIABLE_REFERENCE_AUDIT=UNAVAILABLE\n%s\n', ME.message);
end

fprintf('\n== Model update / compile check ==\n');
try
    set_param(model, 'SimulationCommand', 'update');
    fprintf('UPDATE=OK\n');
catch ME
    fprintf('UPDATE=ERROR\n%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
end

fprintf('\n== Short simulation check ==\n');
try
    in = Simulink.SimulationInput(model);
    in = in.setModelParameter('StopTime', '0.02', 'ReturnWorkspaceOutputs', 'on');
    out = sim(in);
    fprintf('SIM=OK\n');
    fprintf('SIM_OUTPUTS=%s\n', strjoin(who(out), ', '));
catch ME
    fprintf('SIM=ERROR\n%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
end

bdclose(model);
